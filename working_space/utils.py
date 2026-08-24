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


def log_memory_state(epoch, args):
    """Print host, system, and CUDA memory usage for one epoch."""
    if not args.log_memory:
        return

    rss_mb = current_rss_mb()
    if rss_mb is not None:
        print(f"[mem] epoch {epoch:03d}  host_rss={rss_mb:.0f} MiB")
    if psutil is not None:
        virtual = psutil.virtual_memory()
        print(
            f"[mem] epoch {epoch:03d}  "
            f"sys_available={virtual.available/1024**2:.0f} MiB  "
            f"sys_used={virtual.used/1024**2:.0f} MiB"
        )
    if torch.cuda.is_available():
        print(
            f"[mem] epoch {epoch:03d}  "
            f"cuda_allocated={torch.cuda.memory_allocated()/1024**2:.0f} MiB  "
            f"cuda_reserved={torch.cuda.memory_reserved()/1024**2:.0f} MiB"
        )


def create_run_dir(output_root):
    """Create a timestamped directory for one training run's artifacts."""
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    run_dir = Path(output_root) / f"model_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(output_root) / f"model_{timestamp}_{suffix:02d}"
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
