"""
Yale/UNC-CH - Geophysical Waveform Inversion
UNet Baseline Training Script

Expected data layout (Kaggle input or local):
  input_dir/
    FlatVel_A/data/*.npy  (500, 5, 1000, 70)
    FlatVel_A/model/*.npy (500, 70, 70)
    FlatFault_A/seis*.npy
    FlatFault_A/vel*.npy
    ...
    test/{oid}.npy        (5, 1000, 70) per file

Submission format: one row per oid/y position with odd x-columns only.
"""

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import Cfg, load_velocity_stats, resolve_device, select_families, stats_path_for_families
from data import SeisVelDataset, build_flat_indices, find_pairs
from model import UNet
from training import train_one_epoch, unwrap_model, validate, wrap_model_for_parallel
from utils import create_run_dir, log_memory_state, save_json, save_mae_curve

# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def main():
    """Train the U-Net using the configured dataset and command-line overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(Cfg.train_data_dir))
    parser.add_argument("--out_dir",  default=str(Cfg.output_dir))
    parser.add_argument("--epochs",   type=int, default=Cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=Cfg.num_workers,
        help="DataLoader worker count; use 0 to eliminate worker memory overhead.",
    )
    parser.add_argument(
        "--parallel_mode",
        choices=("single", "data_parallel", "ddp"),
        default=Cfg.parallel_mode,
        help="Parallel backend; data_parallel supports the current multi-GPU workflow.",
    )
    parser.add_argument(
        "--family",
        default=None,
        help="Case-insensitive family keyword(s), comma-separated, or 'all'.",
    )
    parser.add_argument(
        "--stats_path",
        default=None,
        help="Optional velocity-statistics JSON path; defaults to the selected families.",
    )
    parser.add_argument(
        "--log_memory",
        action="store_true",
        help="Print host and CUDA memory usage after each epoch.",
    )
    args = parser.parse_args()
    device = resolve_device()
    selected_families = select_families(args.family)
    stats_path = Path(args.stats_path) if args.stats_path else stats_path_for_families(
        selected_families
    )
    if stats_path.is_file():
        vel_mean, vel_std = load_velocity_stats(stats_path)
        print(f"[info] loaded velocity statistics: {stats_path}")
    else:
        vel_mean, vel_std = Cfg.vel_mean, Cfg.vel_std
        print(
            f"[warn] statistics file not found: {stats_path}; "
            "using Cfg.vel_mean and Cfg.vel_std"
        )
    if args.family and args.out_dir == str(Cfg.output_dir):
        output_name = args.family.strip().lower().replace(",", "_")
        args.out_dir = os.path.join(args.out_dir, output_name)
    run_dir = create_run_dir(args.out_dir)
    run_started_at = time.perf_counter()

    torch.manual_seed(Cfg.seed)
    np.random.seed(Cfg.seed)

    # Collect paired training files and build a file-level split.
    print(f"[info] selected families: {', '.join(selected_families)}")
    pairs = find_pairs(args.data_dir, selected_families)
    if len(pairs) == 0:
        raise RuntimeError("No paired data files found; check the --data_dir path")

    indices = build_flat_indices(pairs)
    # Split at file level to reduce leakage between samples from one file.
    file_ids = list(range(len(pairs)))
    if len(file_ids) < 2:
        raise ValueError("At least two data files are required for a train/validation split")
    tr_files, va_files = train_test_split(
        file_ids, test_size=Cfg.val_ratio, random_state=Cfg.seed
    )
    tr_set = set(tr_files); va_set = set(va_files)
    tr_idx = [idx for idx in indices if idx[0] in tr_set]
    va_idx = [idx for idx in indices if idx[0] in va_set]
    del indices, tr_set, va_set
    gc.collect()
    print(f"[info] train samples: {len(tr_idx)}, val samples: {len(va_idx)}")

    train_ds = SeisVelDataset(pairs, tr_idx, vel_mean=vel_mean, vel_std=vel_std)
    val_ds   = SeisVelDataset(pairs, va_idx, vel_mean=vel_mean, vel_std=vel_std)
    # Pinning only helps when workers copy batches; with num_workers=0 it is
    # pure host-memory overhead and a common source of pinned-pool growth.
    use_pin_memory = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=use_pin_memory)

    # Build the model, optimizer, scheduler, and loss function.
    model = UNet(in_ch=Cfg.n_src, base=Cfg.model_base_channels).to(device)
    model, resolved_parallel_mode = wrap_model_for_parallel(
        model, args.parallel_mode, str(device)
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()  # MAE in normalized target units.

    best_val = float("inf")
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        log_memory_state(epoch, args)
        gc.collect()
        train_mae_raw = tr_loss * vel_std
        val_mae_raw = va_loss * vel_std
        history.append({
            "epoch": epoch,
            "train_mae_norm": tr_loss,
            "val_mae_norm": va_loss,
            "train_mae_raw": train_mae_raw,
            "val_mae_raw": val_mae_raw,
        })
        print(
            f"epoch {epoch:03d}  "
            f"train_mae_norm={tr_loss:.4f}  "
            f"val_mae_norm={va_loss:.4f}  "
            f"val_mae_raw={val_mae_raw:.2f}"
        )
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            state = unwrap_model(model).state_dict()
            torch.save(state, run_dir / "best_unet.pth")
            del state
            gc.collect()
            print(f"  saved best (val_mae_raw={val_mae_raw:.2f})")

    elapsed_seconds = time.perf_counter() - run_started_at
    save_json(run_dir / "history.json", history)
    save_mae_curve(history, run_dir / "mae_curve.png")
    save_json(run_dir / "config.json", {
        "arguments": vars(args),
        "selected_families": selected_families,
        "config": {
            key: value for key, value in vars(Cfg).items()
            if not key.startswith("__") and not callable(value)
        },
        "velocity_statistics": {
            "path": stats_path,
            "mean": vel_mean,
            "std": vel_std,
        },
    })
    save_json(run_dir / "velocity_stats.json", {
        "families": selected_families,
        "mean": vel_mean,
        "std": vel_std,
        "source": stats_path,
    })
    save_json(run_dir / "results.json", {
        "run_dir": str(run_dir),
        "device": str(device),
        "parallel_mode_requested": args.parallel_mode,
        "parallel_mode_resolved": resolved_parallel_mode,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "model": "UNet",
        "model_base_channels": Cfg.model_base_channels,
        "parameter_count": n_params,
        "train_files": len(tr_files),
        "validation_files": len(va_files),
        "train_samples": len(tr_idx),
        "validation_samples": len(va_idx),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_mae_norm": best_val,
        "best_val_mae_raw": best_val * vel_std,
        "velocity_mean": vel_mean,
        "velocity_std": vel_std,
        "velocity_stats_path": stats_path,
        "elapsed_seconds": elapsed_seconds,
    })
    print(f"[done] best val_mae_raw = {best_val * vel_std:.2f}")
    print(f"[done] run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
