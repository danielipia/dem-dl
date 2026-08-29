"""
predict_dem.py
===============

Predict DEM from a trained model checkpoint on a full AIA image (6, H, W).
All model layers use Conv2d with kernel_size=1, so the model works identically
for single pixels (1,1) or full images (H,W). This enables efficient batch
inference on entire cubes.

Usage:
    python predict_dem.py --checkpoint ./results/run1/best.pth \
                           --input image.npz --project-root ..

Input .npz must contain an 'AIACube' array of shape (6, H, W) in the same units
as training (DN/s corrected for exposure and degradation, as produced by fullBP.py).
Raw FITS files must first be processed through the full calibration pipeline.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def load_model(checkpoint_path, project_root, device):
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import src.model as model_module

    ckpt = torch.load(checkpoint_path, map_location=device)

    model_name = ckpt["model_name"]
    model_kwargs = ckpt["model_kwargs"]
    ModelClass = getattr(model_module, model_name)
    model = ModelClass(**model_kwargs).double().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    requires_basis = getattr(model, "requires_basis", False)
    return model, requires_basis, ckpt


def build_basis_if_needed(requires_basis, rdata_path, basis_alphas, notrunc, device):
    if not requires_basis:
        return None
    from train_pixel_dem import getBasis

    RData_raw = np.load(rdata_path)
    R, logT = RData_raw["R"], RData_raw["logT"]
    alphas = [float(a) for a in basis_alphas.split("_")]
    B = getBasis(R, logT, alphas=alphas, notrunc=notrunc)
    return torch.from_numpy(B).double().to(device)


def predict_dem_cube(model, aia_cube, requires_basis, B_t, apply_softplus, device):
    """
    Args:
        aia_cube: numpy array (6, H, W) in training units.
        requires_basis: whether model outputs basis coefficients.
        B_t: basis matrix on device (or None).
        apply_softplus: whether to apply softplus activation.
        device: torch device.

    Returns:
        numpy array (n_bins, H, W) containing predicted DEM.
    """
    with torch.no_grad():
        aia_cube = np.ascontiguousarray(aia_cube)
        x = torch.from_numpy(aia_cube).double().unsqueeze(0).to(device)  # (1,6,H,W)
        raw_output = model(x)
        if isinstance(raw_output, tuple):
            raw_output = raw_output[0]

        coef_or_dem = F.softplus(raw_output) if apply_softplus else raw_output

        if requires_basis:
            coef = coef_or_dem.permute(0, 2, 3, 1)
            dem = torch.matmul(coef, B_t.T)
            dem = dem.permute(0, 3, 1, 2)
        else:
            dem = coef_or_dem

        return dem.squeeze(0).cpu().numpy()  # (n_bins, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth or latest.pth")
    parser.add_argument("--input", required=True, help=".npz with 'AIACube' array (6,H,W)")
    parser.add_argument("--project-root", default="..", help="Project root directory")
    parser.add_argument("--rdata", default="RData.npz", help="Path to RData.npz")
    parser.add_argument("--basis-alphas", default="0.0_0.1_0.2", help="Basis alphas")
    parser.add_argument("--notrunc", action="store_true", help="No truncation in basis")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--output", help="Output .npz path")
    parser.add_argument("--decimate", type=int, default=128, help="Spatial decimation factor")
    args = parser.parse_args()

    device = torch.device(args.device)

    model, requires_basis, ckpt = load_model(args.checkpoint, args.project_root, device)
    apply_softplus = "no-softplus" not in json.dumps(ckpt.get("model_kwargs", {}))

    B_t = build_basis_if_needed(
        requires_basis, args.rdata, args.basis_alphas, args.notrunc, device
    )

    from numcodecs import Blosc

    _aia_compressor = Blosc(cname="zstd", clevel=4, shuffle=2)
    npz = np.load(args.input)

    aia_cube_shape = tuple(npz["AIACubeShape"])
    aia_cube = np.frombuffer(
        _aia_compressor.decode(npz["AIACube"].tobytes()),
        dtype=np.float32,
    ).reshape(aia_cube_shape)   

    dec = args.decimate
    aia_decimated = aia_cube[:, ::dec, ::dec]

    dem_pred = predict_dem_cube(model, aia_decimated, requires_basis, B_t, apply_softplus, device)

    np.savez(args.output, DEMCube=dem_pred)
    print(f"Predicted DEM: shape {dem_pred.shape}, saved to {args.output}")


if __name__ == "__main__":
    main()
