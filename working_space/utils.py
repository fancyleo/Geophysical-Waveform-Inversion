"""Utility helpers: memory logging, run directories, and artifact saving."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    import psutil
except ImportError:
    psutil = None


def current_rss_mb():
    """Return current resident memory in MiB when psutil is available."""
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / 1024**2


def memory_snapshot():
    """Return a dict of host and CUDA memory metrics for leak diagnostics.

    ``host_uss`` (unique set size) excludes pages shared with other processes
    (mmap'd data files, NCCL shared segments), so its growth indicates a true
    per-process leak rather than shared page-cache warm-up. ``host_data`` is
    the malloc-managed heap/brk size, which is what glibc refuses to return to
    the OS when free lists fragment. ``cuda_host_pinned_mb`` tracks the CUDA
    host pinned pool used by DDP/NCCL, which is invisible to ``memory_allocated``.
    """
    snapshot = {}
    if psutil is not None:
        info = psutil.Process().memory_full_info()
        snapshot["host_rss_mb"] = info.rss / 1024**2
        uss = getattr(info, "uss", None)
        snapshot["host_uss_mb"] = uss / 1024**2 if uss is not None else None
        data = getattr(info, "data", None)
        snapshot["host_data_mb"] = data / 1024**2 if data is not None else None
    if torch.cuda.is_available():
        snapshot["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        snapshot["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        stats = torch.cuda.memory_stats()
        pinned = stats.get("host_allocated_bytes") or 0
        snapshot["cuda_host_pinned_mb"] = pinned / 1024**2
    return snapshot


def log_memory_state(epoch, args, rss_baseline=None):
    """Print host and CUDA memory usage for one epoch.

    When ``rss_baseline`` is given, also prints the per-epoch host RSS delta so
    slow leaks are visible without flooding the log with system-wide fields.
    Also prints ``uss`` (excludes shared mmap pages) and the CUDA host pinned
    pool so a growing RSS can be attributed to a true leak vs. malloc
    fragmentation vs. page-cache warm-up.
    """
    if not args.log_memory:
        return

    snapshot = memory_snapshot()
    rss_mb = snapshot.get("host_rss_mb")
    if rss_mb is not None:
        message = f"[mem] epoch {epoch:03d}  host_rss={rss_mb:.0f} MiB"
        if rss_baseline is not None:
            message += f"  delta={rss_mb - rss_baseline:+.0f} MiB"
        uss = snapshot.get("host_uss_mb")
        if uss is not None:
            message += f"  uss={uss:.0f} MiB"
        data = snapshot.get("host_data_mb")
        if data is not None:
            message += f"  data={data:.0f} MiB"
        pinned = snapshot.get("cuda_host_pinned_mb")
        if pinned is not None:
            message += f"  host_pinned={pinned:.0f} MiB"
        print(message)
    if torch.cuda.is_available():
        print(
            f"[mem] epoch {epoch:03d}  "
            f"cuda_allocated={snapshot['cuda_allocated_mb']:.0f} MiB  "
            f"cuda_reserved={snapshot['cuda_reserved_mb']:.0f} MiB"
        )


def create_run_dir(output_root, prefix="model"):
    """Create a timestamped directory for one run's artifacts.

    ``prefix`` controls the directory name (e.g. ``model_260824_1012`` or
    ``test_260824_1012`` for smoke runs).
    """
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    run_dir = Path(output_root) / f"{prefix}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(output_root) / f"{prefix}_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def save_json(path, data):
    """Write JSON data while converting pathlib values to strings."""
    with open(path, "w") as file:
        json.dump(data, file, indent=2, default=str)


def save_mae_curve(history, path):
    """Save normalized and raw train/validation MAE curves as a PNG."""
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [item["train_mae_norm"] for item in history], label="train")
    axes[0].plot(epochs, [item["val_mae_norm"] for item in history], label="validation")
    axes[0].set_title("Normalized MAE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [item["train_mae_raw"] for item in history], label="train")
    axes[1].plot(epochs, [item["val_mae_raw"] for item in history], label="validation")
    axes[1].set_title("Raw-unit MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
