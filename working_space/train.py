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
import copy
import gc
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split

from config import Cfg, load_velocity_stats, resolve_device, select_families, stats_path_for_families
from data import SeisVelDataset, build_flat_indices, find_pairs
from model import MODEL_NAMES, ACT_NAMES, build_model
from training import (
    cleanup_ddp,
    get_rank,
    is_main_process,
    setup_ddp,
    train_one_epoch,
    unwrap_model,
    validate,
    wrap_model_for_parallel,
)
from utils import (
    create_run_dir,
    current_rss_mb,
    log_memory_state,
    memory_snapshot,
    save_json,
    save_mae_curve,
)

# Ask glibc to return freed host memory to the OS promptly instead of keeping it
# in the process heap. With num_workers=0 every per-sample numpy temporary is
# allocated in the rank process, so without this the heap (and thus host RSS)
# climbs across epochs under DDP even though nothing is truly leaked. These are
# inherited by mp.spawn'd children (fork inherits malloc tuning; spawn re-reads
# the env vars at startup), so they must be set before spawning.
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")   # trim heap when >=128KB is free
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "131072")   # large allocs via mmap -> returned on free
os.environ.setdefault("MALLOC_TOP_PAD_", "65536")           # keep ~64KB headroom after a trim
os.environ.setdefault("MALLOC_ARENA_MAX", "2")              # cap arenas so freed memory coalesces

# ---------------------------------------------------------------------------
# Project-root progress log
#
# Appends the current training/project progress (startup context, per-epoch
# metrics, best model, final summary) as timestamped lines to
# training_progress.log under the project root so it can be inspected anytime
# with tail. Console output is unchanged (each line is also echoed to stdout).
# Only the main process (rank 0) writes to the file; DDP workers do not.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROGRESS_LOG_PATH = PROJECT_ROOT / "training_progress.log"


def _log_timestamp():
    """Return the current local time as a log-friendly string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ProgressLogger:
    """Append timestamped progress lines to the project-root log file.

    Every line is written as ``[<time>] <message>`` to ``PROGRESS_LOG_PATH``
    (created if missing). When ``echo`` is True (default) the message is also
    printed to the console, so existing console progress output is preserved.
    """

    def __init__(self, path=PROGRESS_LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message, echo=True):
        """Append one progress line to the log file and optionally echo it."""
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(f"[{_log_timestamp()}] {message}\n")
        if echo:
            print(message)


def _load_resume_state(path, device):
    """Load a model state_dict from a checkpoint, normalizing common formats.

    Accepts a bare state_dict (as saved by this repo's ``best_unet.pth``) or a
    dict that wraps it under ``model`` / ``model_state_dict`` / ``state_dict``.
    Keys prefixed with ``module.`` (DataParallel) are stripped so the state
    matches a bare ``UNet``.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
    return checkpoint


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
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Run a 3-epoch smoke training on the flat families with memory monitoring.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of processes to spawn for DDP (multi-GPU).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a state_dict checkpoint (e.g. a previous run's "
             "best_unet.pth) used to initialize weights and continue training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=Cfg.lr,
        help="Peak learning rate; lower it (e.g. 1e-4) when resuming/fine-tuning "
             "an already-trained model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=Cfg.weight_decay,
        help="AdamW L2 weight decay for regularization.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=Cfg.dropout,
        help="UNet bottleneck/decoder Dropout2d rate (0 disables dropout).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=Cfg.early_stop_patience,
        help="Early stop after this many epochs without validation improvement "
             "(0 disables early stopping).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP (fp16 autocast + GradScaler) for training steps.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="EMA decay for weight averaging (0 disables EMA; e.g. 0.999).",
    )
    parser.add_argument(
        "--schedule",
        choices=("cosine", "ruby"),
        default="ruby",
        help="LR schedule: cosine anneals over all epochs; ruby keeps peak LR for "
             "the first 80 percent of epochs, then cosine-decays over the final "
             "20 percent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Cfg.seed,
        help="Random seed for model init, file-level split, and shuffling; use a "
             "different value per run to grow a multi-seed ensemble pool.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=Cfg.val_split_seed,
        help="Seed for the file-level train/val split. Keep the same value across "
             "seeds so every run shares one fixed holdout (comparable val + fair "
             "ensemble eval).",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=1.0,
        help="Data-scaling experiment knob: fraction of TRAIN files to keep "
             "(1.0 = all). The val/holdout split is untouched, so different "
             "fractions stay mutually comparable on the same holdout.",
    )
    parser.add_argument(
        "--frac_seed",
        type=int,
        default=0,
        help="Seed for the deterministic --train_frac file subsample.",
    )
    parser.add_argument(
        "--aug_json",
        default=None,
        help="Optional JSON file holding an {'augmentations': {...}} override. "
             "Use this for augmentation A/Bs so the global "
             "output/aug_explore/winner_aug.json stays untouched.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default=Cfg.model_name,
        help="Architecture: 'unet' (original, symmetric 3x3 kernels) or "
             "'seisunet' (asymmetric time-compression U-Net, see seisunet.md).",
    )
    parser.add_argument(
        "--act",
        choices=ACT_NAMES,
        default=Cfg.activation,
        help="Hidden activation. 'leaky_relu' is the current default; pass 'relu' "
             "to reproduce checkpoints trained before 2026-09-15.",
    )
    parser.add_argument(
        "--out_activation",
        choices=("none", "tanh"),
        default=Cfg.out_activation,
        help="Output activation. 'none' matches this project's z-scored targets; "
             "'tanh' assumes a MinMax[-1, 1] target.",
    )
    args = parser.parse_args()

    if args.test_run:
        # test_run only overrides family, epochs, and memory logging; all other
        # settings (batch_size, num_workers, parallel_mode, ...) follow the
        # user-supplied values or Cfg defaults.
        args.family = "flat"
        args.epochs = 3
        args.log_memory = True
        print("[info] test_run enabled: flat families, 3 epochs, memory monitoring on")

    if args.nproc_per_node > 1 and args.parallel_mode in ("data_parallel", "ddp"):
        # mp.spawn + env:// rendezvous requires MASTER_ADDR/MASTER_PORT; they are
        # not set by mp.spawn automatically. Default to the local node so the
        # spawned processes can form the NCCL process group.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        mp.spawn(
            train_worker,
            args=(args.nproc_per_node, args),
            nprocs=args.nproc_per_node,
            join=True,
        )
    else:
        train_worker(0, 1, args)


def train_worker(local_rank, world_size, args):
    """Run training in one process; multiple ranks use DistributedDataParallel."""
    distributed = world_size > 1
    if distributed:
        device = setup_ddp(local_rank, world_size)
    else:
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
    run_prefix = "test" if args.test_run else "model"
    run_dir = create_run_dir(args.out_dir, prefix=run_prefix) if is_main_process() else None
    run_started_at = time.perf_counter()

    # Project-root progress log: only the main process creates it and writes a
    # one-line startup context.
    progress = ProgressLogger() if is_main_process() else None
    if progress is not None:
        progress.write(
            f"[start] run_dir={run_dir}  device={device}  families="
            f"{', '.join(selected_families)}  epochs={args.epochs}  "
            f"batch_size={args.batch_size}  seed={args.seed}  "
            f"split_seed={args.split_seed}  train_frac={args.train_frac}  "
            f"parallel_mode={args.parallel_mode}",
            echo=False,
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Collect paired training files and build a file-level split.
    if is_main_process():
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
        file_ids, test_size=Cfg.val_ratio, random_state=args.split_seed
    )
    # Optional data-scaling: keep a deterministic subset of the TRAIN files only.
    # The split above is untouched, so every fraction is evaluated on exactly the
    # same val files and the fractions remain mutually comparable.
    n_train_files = len(tr_files)
    if args.train_frac < 1.0:
        frac_rng = np.random.default_rng(args.frac_seed)
        n_keep = max(1, int(round(n_train_files * args.train_frac)))
        tr_files = sorted(
            frac_rng.choice(np.asarray(tr_files), size=n_keep, replace=False).tolist()
        )
        if is_main_process() and progress is not None:
            progress.write(
                f"[info] train_frac={args.train_frac}: keeping {n_keep}/"
                f"{n_train_files} train files (frac_seed={args.frac_seed})"
            )
    tr_set = set(tr_files); va_set = set(va_files)
    tr_idx = [idx for idx in indices if idx[0] in tr_set]
    va_idx = [idx for idx in indices if idx[0] in va_set]
    del indices, tr_set, va_set
    gc.collect()
    if is_main_process():
        progress.write(f"[info] train samples: {len(tr_idx)}, val samples: {len(va_idx)}")

    # Augmentations for this run: the config default, optionally overridden by
    # --aug_json (keeps augmentation A/Bs away from the global winner_aug.json).
    augmentations = Cfg.augmentations
    if args.aug_json:
        with open(args.aug_json) as aug_file:
            augmentations = json.load(aug_file).get("augmentations", {})
    if is_main_process() and progress is not None:
        progress.write(
            f"[info] augmentations: {augmentations if augmentations else '{} (none)'}"
        )

    train_ds = SeisVelDataset(
        pairs, tr_idx, vel_mean=vel_mean, vel_std=vel_std,
        augmentations=augmentations, train=True, seed=args.seed,
    )
    val_ds = SeisVelDataset(pairs, va_idx, vel_mean=vel_mean, vel_std=vel_std,
                            train=False)
    # Pinning only helps when workers copy batches; with num_workers=0 it is
    # pure host-memory overhead and a common source of pinned-pool growth.
    use_pin_memory = args.num_workers > 0

    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=get_rank(), shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=get_rank(), shuffle=False
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=args.num_workers,
                                  pin_memory=use_pin_memory,
                                  persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                sampler=val_sampler, num_workers=args.num_workers,
                                pin_memory=use_pin_memory,
                                persistent_workers=args.num_workers > 0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=use_pin_memory,
                                  persistent_workers=args.num_workers > 0)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=use_pin_memory,
                                  persistent_workers=args.num_workers > 0)

    # Build the model; optionally initialize weights from a previous run.
    model = build_model(
        name=args.model,
        in_ch=Cfg.n_src,
        base=Cfg.model_base_channels,
        dropout=args.dropout,
        act=args.act,
        out_activation=args.out_activation,
    ).to(device)
    if is_main_process() and progress is not None:
        progress.write(
            f"[info] model: {args.model}  act: {args.act}  "
            f"out_activation: {args.out_activation}"
        )
    if args.resume:
        state_dict = _load_resume_state(args.resume, device)
        model.load_state_dict(state_dict)
        if progress is not None:
            progress.write(f"[info] resumed weights from {args.resume}")
    model, resolved_parallel_mode = wrap_model_for_parallel(
        model, args.parallel_mode, device, world_size
    )
    n_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        progress.write(f"[info] model params: {n_params/1e6:.2f}M")

    # A lower peak LR is recommended when resuming an already-trained model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        # Ruby-style schedule: constant peak LR for the first 80% of epochs,
        # then cosine decay over the final 20%.
        const_epochs = max(1, int(round(0.8 * args.epochs)))
        cos_epochs = max(1, args.epochs - const_epochs)

        def _ruby_lr_lambda(epoch_idx):  # 0-based epoch index
            if epoch_idx < const_epochs:
                return 1.0
            t = (epoch_idx - const_epochs) / cos_epochs
            return 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_ruby_lr_lambda
        )
    criterion = nn.L1Loss()  # MAE in normalized target units.

    # Mixed precision (AMP) + optional EMA weight averaging.
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema_state = None
    ema_model = None
    ema_best_val = float("inf")
    ema_best_epoch = None
    ema_history = []
    if args.ema_decay and args.ema_decay > 0 and not distributed:
        base_model = unwrap_model(model)
        ema_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        ema_model = copy.deepcopy(base_model)
        ema_model.eval()
        if progress is not None:
            progress.write(
                f"[info] EMA enabled (decay={args.ema_decay}) "
                f"with {len(ema_state)} state tensors"
            )

    def _ema_update():
        """Refresh EMA copies of every floating tensor in the model state."""
        if ema_state is None:
            return
        with torch.no_grad():
            decay = args.ema_decay
            for name, value in unwrap_model(model).state_dict().items():
                if name in ema_state and value.is_floating_point():
                    ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)

    best_val = float("inf")
    best_epoch = None
    no_improve_epochs = 0
    history = []
    memory_series = []
    rss_baseline = current_rss_mb()
    for epoch in range(1, args.epochs + 1):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_amp=args.amp, scaler=scaler,
            ema_update=(_ema_update if ema_state is not None else None),
        )
        va_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        if is_main_process():
            log_memory_state(epoch, args, rss_baseline)
        if args.log_memory and is_main_process():
            # Snapshot host RSS/uss/data and CUDA allocated/reserved/pinned so
            # results.json can attribute any RSS growth to a true leak (uss up),
            # malloc fragmentation (data up, uss flat), or page-cache warm-up.
            memory_series.append({"epoch": epoch, **memory_snapshot()})
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
        if is_main_process():
            progress.write(
                f"epoch {epoch:03d}  "
                f"train_mae_norm={tr_loss:.4f}  "
                f"val_mae_norm={va_loss:.4f}  "
                f"val_mae_raw={val_mae_raw:.2f}"
            )
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            no_improve_epochs = 0
            if is_main_process():
                state = unwrap_model(model).state_dict()
                torch.save(state, run_dir / "best_unet.pth")
                del state
                gc.collect()
                progress.write(f"  saved best (val_mae_raw={val_mae_raw:.2f})")
        elif args.patience > 0:
            no_improve_epochs += 1

        if ema_state is not None:
            with torch.no_grad():
                ema_model.load_state_dict(ema_state)
            ema_va_norm = validate(ema_model, val_loader, criterion, device)
            ema_va_raw = ema_va_norm * vel_std
            ema_history.append({"epoch": epoch, "val_mae_raw": ema_va_raw})
            if is_main_process():
                progress.write(f"ema  epoch {epoch:03d}  val_mae_raw={ema_va_raw:.2f}")
            if ema_va_raw < ema_best_val:
                ema_best_val = ema_va_raw
                ema_best_epoch = epoch
                if is_main_process():
                    torch.save(
                        {k: v.detach().cpu().clone() for k, v in ema_state.items()},
                        run_dir / "best_ema.pth",
                    )
                    progress.write(f"  saved best EMA (val_mae_raw={ema_va_raw:.2f})")

        # Early stopping: stop once val has not improved for `patience` epochs.
        stop_now = args.patience > 0 and no_improve_epochs >= args.patience
        if distributed:
            # Unanimous rank-0 decision so every DDP rank breaks together.
            flag = torch.tensor([1.0 if stop_now else 0.0], device=device)
            dist.broadcast(flag, src=0)
            stop_now = bool(flag[0].item() > 0.5)
        if stop_now:
            if is_main_process():
                progress.write(
                    f"[info] early stopping at epoch {epoch:03d} "
                    f"(no val improvement for {no_improve_epochs} epochs)"
                )
            break

    if not is_main_process():
        cleanup_ddp()
        return

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
        "resumed_from": args.resume,
        "peak_lr": args.lr,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_frac": args.train_frac,
        "frac_seed": args.frac_seed,
        "train_files_available": n_train_files,
        "amp": bool(args.amp),
        "schedule": args.schedule,
        "ema_decay": args.ema_decay,
        "ema_best_epoch": ema_best_epoch,
        "ema_best_val_mae_raw": (ema_best_val if ema_best_val < float("inf") else None),
        "device": str(device),
        "parallel_mode_requested": args.parallel_mode,
        "parallel_mode_resolved": resolved_parallel_mode,
        "world_size": world_size,
        "rank": get_rank(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "model_type": args.model,
        "act": args.act,
        "out_activation": args.out_activation,
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
        "memory_monitoring": memory_series,
        "ema_history": ema_history,
    })
    if progress is not None:
        progress.write(
            f"[done] best epoch={best_epoch}  "
            f"best val_mae_raw={best_val * vel_std:.2f}  "
            f"elapsed={elapsed_seconds:.1f}s"
        )
        progress.write(f"[done] run artifacts saved to {run_dir}")
        progress.write("==================== Training finished ====================", echo=False)
    cleanup_ddp()


if __name__ == "__main__":
    main()