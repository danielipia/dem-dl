"""
fast_predict.py — minimal, speed-first inference on a single AIA timestamp folder.

Usage:
    python scripts/fast_predict.py \
        --input  data/flares_hicad/20170910_sequence/20170910_154359 \
        --model  results/models/<run>/model_best.pth \
        --target results/fast_preds/ \
        [--pointing_file pointing.ecsv] \
        [--corr_table aia_corr.csv] \
        [--batch_size 32] \
        [--aia_only] \
        [--save_ci] \
        [--vmin 0] [--vmax 2000] [--bin_spacing linear]

Saves:
    <target>/<YYYYMMDD_HHMMSS>_reg.npy   — regression DEM [26, 4096, 4096] float32
    <target>/<YYYYMMDD_HHMMSS>_ci.npz    — ci_low, ci_high, ci_std [26, 4096, 4096] float32  (--save_ci only)
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import BasicNetworkFreqClass, BasicNetworkFreqClassConv, BasicNetworkFreqClassSmall, NoNormMixer, FreqClassNonReLU, FreqClassNonReLUNoPos
from src.data import SimpleAIAData
from src.utils import processIndAIAData, unfold_tensor, reconstruct_cube, create_dem_bins, quantiles_from_pmf

_models = {
    'BasicNetworkFreqClass':      BasicNetworkFreqClass,
    'BasicNetworkFreqClassConv':  BasicNetworkFreqClassConv,
    'BasicNetworkFreqClassSmall': BasicNetworkFreqClassSmall,
    'NoNormMixer':                NoNormMixer,
    'FreqClassNonReLU':           FreqClassNonReLU,
    'FreqClassNonReLUNoPos':      FreqClassNonReLUNoPos,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',    required=True,  help='path to a single AIA timestamp folder')
    parser.add_argument('--model',    required=True,  help='path to model .pth checkpoint')
    parser.add_argument('--target',   required=True,  help='output directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--corr_table',    default='aia_corr.csv')
    parser.add_argument('--pointing_file', default='', help='cached pointing table (.ecsv)')
    parser.add_argument('--aia_only', action='store_true',
                        help='zero out XRT bin weights (18-25) for AIA-only models')
    parser.add_argument('--deconvolve', type=str, default='none', choices=['none', 'hofmeister'],
                        help='deconvolution method applied to AIA inputs before inference')
    parser.add_argument('--save_ci', action='store_true',
                        help='also compute and save classification CI bounds (ci_low, ci_high, ci_std)')
    parser.add_argument('--vmin',        type=float, default=0.0,      help='DEM bin min (for --save_ci)')
    parser.add_argument('--vmax',        type=float, default=2000.0,   help='DEM bin max (for --save_ci)')
    parser.add_argument('--bin_spacing', type=str,   default='sqrt', choices=['linear', 'sqrt', 'log'],
                        help='DEM bin spacing (for --save_ci)')
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = os.path.basename(args.input.rstrip('/'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"timestamp: {timestamp}  device: {device}")

    # --- load model ---
    pack = torch.load(args.model, map_location=device)
    model_name = pack['model_name']
    model = _models[model_name](n_bins=pack['model_state_dict']['classification_head.weight'].shape[0] // 26)
    model.load_state_dict(pack['model_state_dict'])
    model.to(device)

    if args.aia_only:
        n_bins = pack['model_state_dict']['classification_head.weight'].shape[0] // 26
        print("zeroing XRT bin weights (18-25)")
        with torch.no_grad():
            model.regression_head[0].weight[18:26] = 0
            if model.regression_head[0].bias is not None:
                model.regression_head[0].bias[18:26] = 0
            for t in range(18, 26):
                s, e = t * n_bins, (t + 1) * n_bins
                model.classification_head.weight[s:e] = 0
                if model.classification_head.bias is not None:
                    model.classification_head.bias[s] = 10.0
                    model.classification_head.bias[s+1:e] = -1e10

    model.eval()
    print(f"model: {model_name}")

    # --- load + preprocess AIA ---
    AIACube, aia_errors, _ = processIndAIAData(args.input, args)
    AIACube = np.maximum(AIACube, 0).astype(np.float32)  # model applies sqrt internally

    # --- patch + dataloader ---
    patches = unfold_tensor(AIACube, 256, 256)
    dummy_err = np.zeros_like(patches)
    loader = DataLoader(SimpleAIAData((patches, dummy_err)),
                        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_patches = len(patches)
    reg_patches = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)

    if args.save_ci:
        n_bins = pack['model_state_dict']['classification_head.weight'].shape[0] // 26
        bins   = create_dem_bins(vmin=args.vmin, vmax=args.vmax, n_bins=n_bins, spacing=args.bin_spacing)
        ci_low_patches  = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)
        ci_high_patches = torch.empty((num_patches, 26, 256, 256), dtype=torch.float32)

    idx = 0
    infer_start = time.perf_counter()
    with torch.no_grad():
        for aia_batch, _ in loader:
            aia_batch = aia_batch.to(device, non_blocking=True)
            reg_out, cls_out = model(aia_batch)
            bs = reg_out.shape[0]
            reg_patches[idx:idx + bs] = reg_out.cpu()

            if args.save_ci:
                # cls_out: [B, 26, n_bins, H, W]
                for t in range(26):
                    logits_t = cls_out[:, t, :, :, :]   # [B, n_bins, H, W]
                    low, _, high = quantiles_from_pmf(logits_t, bins,
                                                      q_low=0.05, q_med=0.50, q_high=0.95,
                                                      use_bin_edges=True)
                    ci_low_patches[idx:idx + bs,  t] = low.cpu()
                    ci_high_patches[idx:idx + bs, t] = high.cpu()
                    del low, high, logits_t
                del cls_out

            idx += bs
    infer_elapsed = time.perf_counter() - infer_start
    total_pixels = num_patches * 256 * 256
    print(f"SPEED: {total_pixels:,} pixels in {infer_elapsed:.2f}s  |  {total_pixels / infer_elapsed:,.0f} px/s  |  {infer_elapsed / total_pixels * 1e6:.3f} us/px")

    reg_cube = reconstruct_cube(reg_patches, (26, 4096, 4096), numpy=True)

    os.makedirs(args.target, exist_ok=True)
    out_path = os.path.join(args.target, f'{timestamp}_reg.npy')
    np.save(out_path, reg_cube)
    print(f"saved → {out_path}  shape={reg_cube.shape}")

    if args.save_ci:
        ci_low_cube  = reconstruct_cube(ci_low_patches,  (26, 4096, 4096), numpy=True)
        ci_high_cube = reconstruct_cube(ci_high_patches, (26, 4096, 4096), numpy=True)
        ci_path = os.path.join(args.target, f'{timestamp}_ci.npz')
        np.savez(ci_path, ci_low=ci_low_cube, ci_high=ci_high_cube)
        print(f"saved → {ci_path}")


if __name__ == '__main__':
    main()
