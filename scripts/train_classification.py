"""
experimental training regression + classification models
"""

import argparse
import os
import re
import numpy as np
import h5py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append('../')

# inherit from original training infrastructure
from src.data import DEMdata, ZarrAIAData, ZarrDEMData, zarrDataset
from src.model import BasicNetworkFreq, BasicNetworkFreqClass, BasicNetworkFreqClassConv, BasicNetworkFreqClassSmall, NoNormMixer, FreqClassNonReLU, FreqClassNonReLUNoPos
from src.train import train_model, handle_loss_fn as original_handle_loss_fn
from src.losses import MaskedMSELoss, ClassificationLoss, RegressionClassificationLoss
from src.utils import getBasis, create_dem_bins, dem_values_to_bins
import time
import wandb

_losses = {
    'MaskedMSELoss': MaskedMSELoss,
    'ClassificationLoss': ClassificationLoss,
    'RegressionClassificationLoss': RegressionClassificationLoss,
}

_models = {
    'BasicNetworkFreq': BasicNetworkFreq,
    'BasicNetworkFreqClass': BasicNetworkFreqClass,
    'BasicNetworkFreqClassConv': BasicNetworkFreqClassConv,
    'BasicNetworkFreqClassSmall': BasicNetworkFreqClassSmall,
    'NoNormMixer': NoNormMixer,
    'FreqClassNonReLU': FreqClassNonReLU,
    'FreqClassNonReLUNoPos': FreqClassNonReLUNoPos,
}

_data_class = {
    'DEMdata': DEMdata,
    'ZarrAIAData': ZarrAIAData,
    'ZarrDEMData': ZarrDEMData,
    'zarrDataset': zarrDataset,
}

def load_pretrained_backbone(new_model, pretrained_path):
    """load backbone weights from pretrained BasicNetworkFreq"""
    print(f"Loading pretrained weights from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
    else:
        pretrained_dict = checkpoint  # direct state dict
    
    # map BasicNetworkFreq layers to BasicNetworkFreqClass backbone
    backbone_dict = {}
    for old_key, value in pretrained_dict.items():
        if old_key.startswith('layers.'):
            layer_idx = int(old_key.split('.')[1])
            if layer_idx < 12:  # skip final regression layer
                new_key = old_key.replace('layers.', 'backbone.')
                backbone_dict[new_key] = value
                print(f"  Mapping {old_key} -> {new_key}")

    # map last layer weights to regression head
    for old_key, value in pretrained_dict.items():
        if old_key.startswith('layers.12.'):
            new_key = old_key.replace('layers.12.', 'regression_head.0.')
            backbone_dict[new_key] = value
            print(f"  Mapping {old_key} -> {new_key}")
    
    missing, unexpected = new_model.load_state_dict(backbone_dict, strict=False)
    print(f"loaded {len(backbone_dict)} backbone tensors")
    if missing:    print("missing:", missing)
    if unexpected: print("unexpected:", unexpected)
    return new_model

def freeze_backbone(model):
    """freeze backbone parameters"""
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen")

def handle_classification_loss(loss_fn, output, im1, im2, bins):
    """extended loss handler for classification models"""
    
    if isinstance(output, tuple) and len(output) == 2:
        # dual-head model: (regression, classification)
        reg_out, cls_out = output
        dem = im2

        if loss_fn.__class__.__name__ == 'MaskedMSELoss':
            # only regression loss, ignore classification head
            loss = loss_fn(reg_out, dem)
            return loss, {}
        
        # convert dem values to bin indices (ensure bins on same device)
        bins_dev = bins.to(dem.device, non_blocking=True)
        bin_indices = dem_values_to_bins(dem, bins_dev)

        # handle NaN values, set to -1 (ignore_index for CrossEntropyLoss)
        bin_indices[torch.isnan(dem)] = -1
        
        # for AIA-only data, high-temp bins (18-25) have no physical signal
        # ignore classification loss for these to prevent hallucination
        # only applies when dem values are zero in these bins
        # B, n_temps, H, W = dem.shape
        # if n_temps == 26:  # full temperature range
        #     high_temp_mask = (dem[:, 18:26, :, :] == 0)
        #     bin_indices[:, 18:26, :, :][high_temp_mask] = -1

        bin_targets = bin_indices.long()

        if loss_fn.__class__.__name__ == 'ClassificationLoss':
            # only classification loss
            loss = loss_fn(cls_out, bin_targets)
        elif loss_fn.__class__.__name__ == 'RegressionClassificationLoss':
            # combined loss
            loss = loss_fn(reg_out, cls_out, dem, bin_targets)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn.__class__.__name__}")
            
        return loss, {'bin_targets': bin_targets, 'cls_out': cls_out}
    else:
        # fallback to original handler for single-head models
        return original_handle_loss_fn(loss_fn, output, im1, im2)

def parse_args():
    """Extended argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Folder with train/val/test patches')
    parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--crop', type=str, default="")
    parser.add_argument('--dev', type=str, default='cuda')
    parser.add_argument('--preprocess', type=str, default='')
    parser.add_argument('--loss', type=str, default='ClassificationLoss')
    parser.add_argument('--model', type=str, default='BasicNetworkFreqClass')
    parser.add_argument('--data_class', type=str, default='ZarrAIAData')
    parser.add_argument('--LR', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--log_image_interval', type=int, default=10000)
    parser.add_argument('--log_loss_interval', type=int, default=20)
    parser.add_argument('--run_name', type=str, default='classification_exp')
    
    # classification-specific args
    parser.add_argument('--pretrained_path', type=str, help='Path to pretrained BasicNetworkFreq model')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone')
    parser.add_argument('--n_bins', type=int, default=128, help='Number of classification bins')
    parser.add_argument('--bin_spacing', type=str, default='linear', choices=['linear', 'sqrt', 'log'])
    parser.add_argument('--vmin', type=float, default=0, help='Min DEM value for binning')
    parser.add_argument('--vmax', type=float, default=2000, help='Max DEM value for binning')
    parser.add_argument('--distributed', action='store_true', help='Enable DDP (torchrun/SLURM env required)')
    
    return parser.parse_args()

def setup_distributed(args):
    """
    Initialize distributed mode.
    
    Priority:
      1. torchrun/env:// (RANK, WORLD_SIZE, LOCAL_RANK)
      2. Slurm (SLURM_PROCID, SLURM_NTASKS) - only if torchrun env is absent
      3. Fallback to single-process
    """
    # case 1: torchrun (current setup with --standalone)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # sanity check GPU availability
        available_gpus = torch.cuda.device_count()
        if args.world_size > available_gpus:
            raise RuntimeError(
                f"Requested world_size={args.world_size} but only {available_gpus} GPUs are visible. "
                "Request more GPUs or reduce nproc_per_node."
            )
        
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
        return True
    
    # case 2: pure Slurm + srun (NYU doc style, no torchrun)
    if "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ.get("SLURM_NTASKS", 1))
        gpus_per_node = torch.cuda.device_count()
        args.local_rank = args.rank % max(gpus_per_node, 1)
        
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
        return True
    
    # case 3: single-process
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0
    return False

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def main():
    args = parse_args()
    distributed = setup_distributed(args)
    is_main_process = (not distributed) or args.rank == 0
    if distributed:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        gpu_count = torch.cuda.device_count()
        gpu_ids = []
        for i in range(gpu_count):
            try:
                gpu_ids.append(torch.cuda.get_device_properties(i).uuid)
            except Exception:
                gpu_ids.append(f"idx{i}")
        if is_main_process:
            print(f"CUDA_VISIBLE_DEVICES={visible}")
            print(f"Visible GPU count: {gpu_count}, UUIDs: {gpu_ids}")
        if len(set(gpu_ids)) < gpu_count:
            raise RuntimeError(f"Duplicate GPU UUIDs detected in visibility list: {gpu_ids}")
    
    # create bins
    bins = create_dem_bins(vmin=args.vmin, vmax=args.vmax, n_bins=args.n_bins, spacing=args.bin_spacing)
    if is_main_process:
        print(f"Created {len(bins)-1} bins from {bins[0]:.2f} to {bins[-1]:.2f}")
    bins = torch.tensor(bins, dtype=torch.float32)
    
    # setup
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if is_main_process:
        print(f"Using {num_workers} data loader workers")
    
    if args.wandb and is_main_process:
        wandb.init(
            project="demdemo",
            name=f"{args.model}-{args.loss}_{args.run_name}_{timestamp}",
            config=vars(args),
            tags=[args.model, args.loss, args.data_class, "classification"],
        )
    
    if distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device(args.dev)
    if is_main_process:
        print(f"Device: {device}")
        if distributed:
            print(f"Distributed initialized: world_size={args.world_size}, rank={args.rank}, local_rank={args.local_rank}")
    bins = bins.to(device)

    if distributed:
        # debug prints – leave them on for a test run
        print(
            f"[global rank {args.rank}] local_rank={args.local_rank}, "
            f"world_size={args.world_size}, "
            f"cuda.current_device={torch.cuda.current_device()}, "
            f"device={torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    
    # data loading 
    if is_main_process:
        print("Loading data...")
    dem_data = _data_class[args.data_class](args.data, split="train", args=args)
    if args.run_name == 'test':
        print("Test mode: using small subset")
        dem_data = torch.utils.data.Subset(dem_data, range(10))
    train_sampler = DistributedSampler(dem_data, num_replicas=args.world_size, rank=args.rank, shuffle=True) if distributed else None
    dem_loader = DataLoader(
        dem_data,
        batch_size=args.batch_size,
        shuffle=not distributed,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_data = _data_class[args.data_class](args.data, split="val", args=args)
    val_sampler = DistributedSampler(val_data, num_replicas=args.world_size, rank=args.rank, shuffle=False) if distributed else None
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )    
    
    # model setup
    if is_main_process:
        print(f"Creating model: {args.model}")
    model = _models[args.model](n_bins=args.n_bins)
    
    # load pretrained weights if provided
    if args.pretrained_path:
        model = load_pretrained_backbone(model, args.pretrained_path)
        
        if args.freeze_backbone:
            freeze_backbone(model)

    if distributed and is_main_process:
        print(f"Train dataset size: {len(dem_data)}")
        print(f"Samples per replica (approx): {len(train_sampler)}")

    model = model.to(device)
    if distributed and args.world_size > 1:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    # check requires_basis on underlying model (use .module if wrapped in DDP)
    underlying_model = model.module if isinstance(model, DDP) else model
    if underlying_model.requires_basis:
        print("Model requires basis, loading basis data...")
        # load RData for DEMBasisNet
        RData = np.load("../RData.npz")
        R, logT = RData['R'], RData['logT']
        wavelengths = [94, 131, 171, 193, 211, 335]
        scale = 10**26
        R = (R * scale).astype(np.float64)
        basisAlphas = list(map(float, "0.0_0.1_0.2".split("_")))
        B = getBasis(R, logT, basisAlphas)
        # total matrix is R (nObs x nLogTBins) @ B (nLogTBins x nBasis)
        D = R @ B
        D = torch.tensor(D, dtype=torch.float32).to(device)  # [N, n_temps]
        R = torch.tensor(R, dtype=torch.float32).to(device)  # [nObs, nLogTBins]
        B = torch.tensor(B, dtype=torch.float32).to(device)  # [nLogTBins, n_basis]
    else:
        D = None
        R = None
        B = None

    # loss function
    if args.loss == 'MaskedMSELoss':
        if is_main_process:
            print("Training regression only with BasicNetworkFreqClass")
        criterion = MaskedMSELoss(B=B)
    elif args.loss == 'ClassificationLoss':
        if is_main_process:
            print("Only training classification head")
        criterion = ClassificationLoss(bins, R=R, B=B)
    elif args.loss == 'RegressionClassificationLoss':
        if is_main_process:
            print("Training both regression and classification heads")
        criterion = RegressionClassificationLoss(bins, alpha=0.5)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")
    
    # optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.LR, weight_decay=1e-4)
    if is_main_process:
        print(f"Optimizing {len(trainable_params)} parameter groups")

    # scheduler: drop LR by 1e-2 after 100th epoch
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.01)

    # monkey-patch the loss handler
    import src.train
    src.train.handle_loss_fn = lambda loss_fn, output, im1, im2: handle_classification_loss(loss_fn, output, im1, im2, bins)

    # use existing training infrastructure
    if is_main_process:
        print("Starting training...")
    results_dir = f"../results/models/{args.model}-{args.loss}_{args.run_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # train (reuse existing train_model function)
    t0 = time.time()  # start time
    train_model(
        model,
        dem_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=args.epochs,
        device=device,
        args=args,
        timestamp=timestamp,
        train_sampler=train_sampler,
        world_size=args.world_size,
        rank=args.rank,
        is_main_process=is_main_process,
    )

    if is_main_process:
        print(f"Training complete. Results saved to {results_dir}")
    ttime = "Training time = {0} seconds".format(time.time() - t0)
    if is_main_process:
        print(ttime)
    if args.wandb and is_main_process:
        wandb.finish()
    cleanup_distributed()
if __name__ == '__main__':
    main()
