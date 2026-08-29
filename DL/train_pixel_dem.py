"""
train_pixel_dem.py
===================

Pixelwise training script for DEM models. It mirrors the workflow from
`train.ipynb` but is adapted for long runs on a single machine using
small batches (via DataLoader) to avoid excessive memory use.

Features:
    - Model and loss are selectable by class name; constructor args are
        provided as JSON strings for flexibility.
    - Saves epoch-by-epoch training/validation loss to CSV.
    - Keeps the best model (by validation loss or training loss if no
        validation set is provided) and periodic "latest" checkpoints for
        resuming interrupted runs.

Run from a terminal. Example:

        python train_pixel_dem.py \
                --project-root .. \
                --data-dir ../data/DEMs/train_dl \
                --val-file 20240520_1200.npz \
                --model DemoModel \
                --loss BarrierLoss --loss-kwargs '{"alpha_l1": 1e-3, "mu": 1.0}' \
                --epochs 5000 --batch-size 128 --lr 1e-4 \
                --output-dir ./results/run_barrier_demo

See help (`-h`) for more options.
"""

import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from numcodecs import Blosc, BitRound

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.model as model_module
import src.losses as losses_module

# --------------------------------------------------------------------------
# Loading and decoding .npz files (same logic as in train.ipynb)
# ---------------------------------------------------------------------------

_dem_compressor = Blosc(cname="zstd", clevel=5, shuffle=2)
_aia_compressor = Blosc(cname="zstd", clevel=4, shuffle=2)
_dem_filter = BitRound(keepbits=12)


def load_and_align(npz_path):
    """Load a .npz, decode compressed arrays, and downsample AIA/errors
    so they match the DEMCube grid."""
    npz = np.load(npz_path)

    DEMCubeShape = tuple(npz["DEMCubeShape"])
    DEMCube = np.frombuffer(
        _dem_filter.decode(_dem_compressor.decode(npz["DEMCube"].tobytes())),
        dtype=np.float32,
    ).reshape(DEMCubeShape)

    AIACubeShape = tuple(npz["AIACubeShape"])
    AIACube = np.frombuffer(
        _aia_compressor.decode(npz["AIACube"].tobytes()), dtype=np.float32
    ).reshape(AIACubeShape)
    AIAErrors = np.frombuffer(
        _aia_compressor.decode(npz["AIAErrors"].tobytes()), dtype=np.float32
    ).reshape(AIACubeShape)

    decimate = AIACubeShape[1] // DEMCubeShape[1]
    if decimate > 1:
        AIACube = AIACube[:, ::decimate, ::decimate]
        AIAErrors = AIAErrors[:, ::decimate, ::decimate]

    assert AIACube.shape[1:] == DEMCube.shape[1:]
    return AIACube, AIAErrors, DEMCube


def cube_to_pixels(AIACube, AIAErrors, DEMCube):
    """Convert cubes to per-pixel arrays, removing pixels with NaN in AIA.
    Returns (aia, errors, dem) for valid pixels. DEM may contain NaNs; the
    loss functions handle those appropriately."""
    C, H, W = AIACube.shape
    nBins = DEMCube.shape[0]

    aia_flat = AIACube.reshape(C, -1).T       # (H*W, C)
    err_flat = AIAErrors.reshape(C, -1).T     # (H*W, C)
    dem_flat = DEMCube.reshape(nBins, -1).T   # (H*W, nBins)

    valid = ~np.isnan(aia_flat).any(axis=1)
    return aia_flat[valid], err_flat[valid], dem_flat[valid]


def build_pixel_dataset(npz_paths):
    """Concatenate valid pixels (AIA, error, inverted DEM) from multiple
    .npz files into a single dataset."""
    all_aia, all_err, all_dem = [], [], []
    for p in npz_paths:
        AIACube, AIAErrors, DEMCube = load_and_align(p)
        aia_px, err_px, dem_px = cube_to_pixels(AIACube, AIAErrors, DEMCube)
        all_aia.append(aia_px)
        all_err.append(err_px)
        all_dem.append(dem_px)
    aia = np.concatenate(all_aia, axis=0)
    err = np.concatenate(all_err, axis=0)
    dem = np.concatenate(all_dem, axis=0)
    return aia, err, dem


def resolve_data_files(args):
    """Return (train_paths, val_paths) from --data-dir. val_paths is a list
    of file paths or None if no validation set is used."""
    import glob

    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not all_files:
        raise ValueError(f"No .npz files found in {args.data_dir}")

    val_paths = None
    if args.val_file is not None:
        matches = [p for p in all_files if os.path.basename(p) == args.val_file]
        if not matches:
            raise ValueError(
                f"--val-file '{args.val_file}' is not in {args.data_dir}"
            )
        val_paths = [matches[0]]
        train_paths = [p for p in all_files if p not in val_paths]
    elif args.val_frac > 0:
        n_val = max(1, int(round(len(all_files) * args.val_frac)))
        train_paths = all_files[:-n_val]
        val_paths = all_files[-n_val:]
    else:
        train_paths = all_files

    return train_paths, val_paths


# ---------------------------------------------------------------------------
# Basis functions (copied from fullBP.py). Pure numpy implementation.
# ---------------------------------------------------------------------------

def getBasis(R, logT, alphas=(0.0, 0.1, 0.2), notrunc=False):
    """Construct a basis set (Gaussians in logT of various widths, plus
    deltas), matching the behavior in fullBP.py."""
    nBins = R.shape[1]
    nAlphas = len(alphas)
    assert nBins == logT.size

    basis = np.zeros((nBins, nBins * nAlphas))
    for ai in range(nBins):
        basis[ai, ai] = 1.0

    for ai in range(1, nAlphas):
        a = alphas[ai]
        for bi in range(nBins):
            col = ai * nBins + bi
            diffLogT = logT - logT[bi]
            basisResult = np.exp(-(diffLogT ** 2 / a ** 2))
            if notrunc and bi < 18:
                pass
            else:
                basisResult[basisResult < 0.04] = 0
            basis[:, col] = basisResult

    return basis


# ---------------------------------------------------------------------------
# Dynamic model and loss selection
# ---------------------------------------------------------------------------

def build_model(model_module, model_name, model_kwargs):
    if not hasattr(model_module, model_name):
        raise ValueError(
            f"Class '{model_name}' not found in src/model.py. "
            f"Available classes: {[n for n in dir(model_module) if n[0].isupper() ]}"
        )
    ModelClass = getattr(model_module, model_name)
    return ModelClass(**model_kwargs)


def build_loss(losses_module, loss_name, loss_kwargs, R_t, B_t, D_t):
    """Instantiate the requested loss class following src/losses.py conventions."""
    if not hasattr(losses_module, loss_name):
        raise ValueError(
            f"Class '{loss_name}' not found in src/losses.py. "
            f"Available classes: {[n for n in dir(losses_module) if n[0].isupper()] }"
        )
    LossClass = getattr(losses_module, loss_name)

    if loss_name in ("BarrierLoss", "BarrierLoss_SiLU"):
        defaults = dict(alpha_l2=0.0, alpha_l1=1.0, mu=1.0, alpha_fit=0.0)
        # support ElasticNet-style shorthand inside loss-kwargs if provided
        if "fitlinearalpha" in loss_kwargs and "alpha_l1" not in loss_kwargs and "alpha_l2" not in loss_kwargs:
            a = float(loss_kwargs.pop("fitlinearalpha"))
            l = float(loss_kwargs.pop("fitlinearl1ratio", 0.5))
            defaults["alpha_l1"] = a * l
            defaults["alpha_l2"] = 0.5 * a * (1.0 - l)
        defaults.update(loss_kwargs)
        barrier_args = SimpleNamespace(**defaults)
        return LossClass(D=D_t, R=R_t, B=B_t, args=barrier_args)

    if loss_name == "MaskedMSELoss":
        return LossClass(R=R_t, B=B_t, **loss_kwargs)

    if loss_name == "JointResynthLoss":
        transformed = loss_kwargs.pop("transformed", False)
        return LossClass(R=R_t, transformed=transformed, **loss_kwargs)

    raise NotImplementedError(
        f"'{loss_name}' is not wired into this script (additional logic required). Supported options: BarrierLoss, BarrierLoss_SiLU, MaskedMSELoss, JointResynthLoss."
    )


def forward_pixelwise(model, x):
    """Run the model on pixel-sized inputs. Input (B,C) -> (B,C,1,1).
    If the model returns a tuple (regression, classification), use the
    regression head only."""
    x4 = x.unsqueeze(-1).unsqueeze(-1)  # (B, C) -> (B, C, 1, 1)
    out4 = model(x4)
    if isinstance(out4, tuple):
        out4 = out4[0]
    return out4.squeeze(-1).squeeze(-1)  # (B, n_out)


def compute_loss(loss_name, loss_fn, model, aia_batch, err_batch, dem_batch,
                  R_t, B_t, requires_basis, tolfac, apply_softplus):
    """Run model on a batch of pixels and compute the selected loss. Handles
    coefficients->DEM conversion for basis models."""
    raw_output = forward_pixelwise(model, aia_batch)
    coef_or_dem = F.softplus(raw_output) if apply_softplus else raw_output

    if requires_basis:
        x_for_barrier = coef_or_dem                     # basis coefficients
        dem_pred = coef_or_dem @ B_t.T                   # -> DEM space
    else:
        x_for_barrier = coef_or_dem
        dem_pred = coef_or_dem

    if loss_name in ("BarrierLoss", "BarrierLoss_SiLU"):
        tol = err_batch * tolfac
        lb = aia_batch - tol
        ub = aia_batch + tol
        return loss_fn(x_for_barrier, aia_obs=aia_batch, lb=lb, ub=ub)

    if loss_name == "MaskedMSELoss":
        return loss_fn(dem_pred, dem_batch)

    if loss_name == "JointResynthLoss":
        pred4 = dem_pred.unsqueeze(-1).unsqueeze(-1)
        target4 = dem_batch.unsqueeze(-1).unsqueeze(-1)
        aia4 = aia_batch.unsqueeze(-1).unsqueeze(-1)
        return loss_fn(pred4, target4, aia4)

    raise NotImplementedError(loss_name)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate(model, loss_name, loss_fn, aia_t, err_t, dem_t, R_t, B_t,
             requires_basis, tolfac, apply_softplus, device, batch_size):
    """Evaluate loss over a full set in batches (no gradients)."""
    model.eval()
    ds = TensorDataset(aia_t, err_t, dem_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for aia_batch, err_batch, dem_batch in loader:
            aia_batch = aia_batch.to(device)
            err_batch = err_batch.to(device)
            dem_batch = dem_batch.to(device)
            loss = compute_loss(loss_name, loss_fn, model, aia_batch, err_batch,
                                 dem_batch, R_t, B_t, requires_basis, tolfac,
                                 apply_softplus)
            total_loss += loss.item() * aia_batch.shape[0]
            n += aia_batch.shape[0]
    return total_loss / n


def save_checkpoint(path, model, optimizer, epoch, best_loss, history, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "history": history,
            "model_name": args.model,
            "model_kwargs": json.loads(args.model_kwargs),
            "loss_name": args.loss,
            "loss_kwargs": json.loads(args.loss_kwargs),
        },
        path,
    )


def append_history_csv(csv_path, epoch, train_loss, val_loss, elapsed_s):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "elapsed_seconds"])
        writer.writerow([epoch, train_loss, val_loss, elapsed_s])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default="..",
                         help="Folder that contains 'src/' (model.py, losses.py). Default: '..'")
    parser.add_argument("--rdata", default="RData.npz",
                         help="Path to RData.npz (contains R and logT)")

    # Directory containing all .npz files (train + validation are resolved from here)
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing ALL .npz files. The script uses all files "
                            "for training except the one specified by --val-file, or reserves "
                            "the last --val-frac files for validation.")
    parser.add_argument("--val-file", default=None,
                        help="(Optional) filename inside --data-dir to use as validation; the rest is used for training.")
    parser.add_argument("--val-frac", type=float, default=0.0,
                        help="(Optional, used if --val-file is not set) fraction of files to reserve for validation, e.g. 0.2. Default 0 = no validation.")

    # Modelo (hiperparámetro)
    parser.add_argument("--model", default="DemoModel",
                        help="Name of the class in src/model.py to use")
    parser.add_argument("--model-kwargs", default="{}",
                        help="JSON with model constructor arguments, e.g. '{\"out_channels\": 54}'")
    parser.add_argument("--no-softplus", action="store_true",
                        help="Do not apply softplus to raw model output (useful if model ends with ReLU)")

    # Loss (hiperparámetro)
    parser.add_argument("--loss", default="BarrierLoss",
                        help="Name of the class in src/losses.py to use")
    parser.add_argument("--loss-kwargs", default="{}",
                        help="JSON with loss-specific arguments, e.g. '{\"alpha_l1\": 0.001, \"mu\": 1.0}'")
    parser.add_argument("--tolfac", type=float, default=1.4,
                        help="Used only by BarrierLoss: width factor for observation band [obs-tol, obs+tol]")
    parser.add_argument("--basis-alphas", default="0.0_0.1_0.2",
                        help="Widths for basis functions, separated by '_' (used only if model.requires_basis=True)")
    parser.add_argument("--notrunc", action="store_true",
                        help="Disable basis truncation (same as fullBP.py --notrunc)")

    parser.add_argument("--output-dir", default="./results/pixel_dem_run",
                         help="Directory to save checkpoints and loss history")
    parser.add_argument("--epochs", type=int, default=5000,
                         help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu",
                         help="'cpu', 'cuda' or 'mps'. Default 'cpu'")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                         help="Save 'latest' checkpoint every N epochs (for resuming)")
    parser.add_argument("--resume", action="store_true",
                         help="If set, resume from output-dir/latest.pth if present")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # `model_module` and `losses_module` are imported at top (from `src`),
    # so no need to re-import them here.

    device = torch.device(args.device)
    model_kwargs = json.loads(args.model_kwargs)
    loss_kwargs = json.loads(args.loss_kwargs)

    # -------------------- resolve which files to use --------------------
    train_paths, val_paths = resolve_data_files(args)
    print("Training files:")
    for p in train_paths:
        print(f"  - {p}")
    if val_paths:
        print("Validation files:")
        for p in val_paths:
            print(f"  - {p}")

    # -------------------- data --------------------
    print("Loading training data...")
    train_aia, train_err, train_dem = build_pixel_dataset(train_paths)
    print(f"  training pixels: {train_aia.shape}, DEM: {train_dem.shape}")

    train_aia_t = torch.from_numpy(train_aia).double()
    train_err_t = torch.from_numpy(train_err).double()
    train_dem_t = torch.from_numpy(train_dem).double()
    train_dataset = TensorDataset(train_aia_t, train_err_t, train_dem_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)

    have_val = val_paths is not None
    if have_val:
        print("Loading validation data...")
        val_aia, val_err, val_dem = build_pixel_dataset(val_paths)
        val_aia_t = torch.from_numpy(val_aia).double()
        val_err_t = torch.from_numpy(val_err).double()
        val_dem_t = torch.from_numpy(val_dem).double()
        print(f"  validation pixels: {val_aia.shape}")

    # -------------------- R (and B if required) --------------------
    RData_raw = np.load(args.rdata)
    R = RData_raw["R"]
    logT = RData_raw["logT"]
    scale = 10 ** 26
    R = (R * scale).astype(np.float64)
    R_t = torch.from_numpy(R).double().to(device)

    # -------------------- model --------------------
    model = build_model(model_module, args.model, model_kwargs).double().to(device)
    requires_basis = getattr(model, "requires_basis", False)
    apply_softplus = not args.no_softplus

    B_t = None
    D_t = R_t
    if requires_basis:
        alphas = [float(a) for a in args.basis_alphas.split("_")]
        B = getBasis(R, logT, alphas=alphas, notrunc=args.notrunc)
        B_t = torch.from_numpy(B).double().to(device)
        D_t = R_t @ B_t
        print(f"Model requires_basis=True: B shape {tuple(B_t.shape)} (alphas={alphas})")

    # -------------------- quick shape check --------------------
    with torch.no_grad():
        dummy = train_aia_t[:2].to(device)
        dummy_out = forward_pixelwise(model, dummy)
    expected_out = D_t.shape[1] if requires_basis else R_t.shape[1]
    if dummy_out.shape[1] != expected_out:
        raise ValueError(
            f"Model '{args.model}' output has {dummy_out.shape[1]} channels but {expected_out} were expected (according to R{'@B' if requires_basis else ''}). "
            f"Adjust --model-kwargs (e.g. out_channels/nOut) or --basis-alphas.")

    # -------------------- loss --------------------
    loss_fn = build_loss(losses_module, args.loss, dict(loss_kwargs), R_t, B_t, D_t)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)

    print(f"Model: {args.model}({model_kwargs})  |  Loss: {args.loss}({loss_kwargs})")

    # -------------------- optimizador --------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # No LR scheduler: keep learning rate constant.

    start_epoch = 0
    best_loss = float("inf")
    history = []

    latest_ckpt_path = os.path.join(args.output_dir, "latest.pth")
    best_ckpt_path = os.path.join(args.output_dir, "best.pth")
    csv_path = os.path.join(args.output_dir, "loss_history.csv")
    args_json_path = os.path.join(args.output_dir, "run_args.json")

    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"Resuming from {latest_ckpt_path} ...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        history = ckpt.get("history", [])
        print(f"  resumed at epoch {start_epoch}, best_loss so far: {best_loss:.6f}")
    else:
        with open(args_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # -------------------- training loop --------------------
    t0 = time.time()
    end_epoch = start_epoch + args.epochs
    print(f"Training epochs {start_epoch} to {end_epoch - 1} ...")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        epoch_loss = 0.0
        for aia_batch, err_batch, dem_batch in train_loader:
            aia_batch = aia_batch.to(device)
            err_batch = err_batch.to(device)
            dem_batch = dem_batch.to(device)

            optimizer.zero_grad()
            loss = compute_loss(args.loss, loss_fn, model, aia_batch, err_batch,
                                 dem_batch, R_t, B_t, requires_basis, args.tolfac,
                                 apply_softplus)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * aia_batch.shape[0]

        epoch_loss /= len(train_dataset)

        if have_val:
            val_loss = evaluate(model, args.loss, loss_fn, val_aia_t, val_err_t, val_dem_t,
                                 R_t, B_t, requires_basis, args.tolfac, apply_softplus,
                                 device, args.batch_size)
        else:
            val_loss = None

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        append_history_csv(csv_path, epoch, epoch_loss, val_loss, elapsed)

        monitored_loss = val_loss if have_val else epoch_loss
        is_best = monitored_loss < best_loss
        if is_best:
            best_loss = monitored_loss
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_loss, history, args)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == end_epoch - 1:
            save_checkpoint(latest_ckpt_path, model, optimizer, epoch, best_loss, history, args)

        if epoch % 20 == 0 or epoch == end_epoch - 1:
            msg = f"epoch {epoch}  train_loss: {epoch_loss:.4f}"
            if have_val:
                msg += f"  val_loss: {val_loss:.4f}"
            msg += f"  best: {best_loss:.4f}  ({elapsed/60:.1f} min)"
            print(msg)

    save_checkpoint(latest_ckpt_path, model, optimizer, end_epoch - 1, best_loss, history, args)
    print(f"Done. Best loss: {best_loss:.6f}")
    print(f"  Best model:  {best_ckpt_path}")
    print(f"  Latest checkpoint: {latest_ckpt_path}")
    print(f"  History CSV: {csv_path}")


if __name__ == "__main__":
    main()
