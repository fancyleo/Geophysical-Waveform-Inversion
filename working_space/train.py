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

import os, glob, json, argparse, time
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from config import Cfg, load_velocity_stats, select_families, stats_path_for_families

if Cfg.device == "auto":
    Cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------
def find_pairs(root, families=None):
    """Find paired seismic and velocity files for the selected families."""
    families = Cfg.families if families is None else families
    pairs = []
    for fam in families:
        fam_dir = os.path.join(root, fam)
        if not os.path.isdir(fam_dir):
            print(f"[warn] missing family dir: {fam_dir}")
            continue

        # Velocity and style families store data and models in separate folders.
        data_dir = os.path.join(fam_dir, "data")
        model_dir = os.path.join(fam_dir, "model")
        if os.path.isdir(data_dir) and os.path.isdir(model_dir):
            seis_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
            for sf in seis_files:
                base = os.path.basename(sf)
                # Convert data1.npy to model1.npy.
                mf = os.path.join(model_dir, base.replace("data", "model"))
                if os.path.exists(mf):
                    pairs.append((sf, mf))
            continue

        pairs.extend(find_fault_pairs(fam_dir))

    print(f"[info] total paired files: {len(pairs)}")
    return pairs


def find_fault_pairs(family_dir):
    """Find recursive same-directory pairs such as ``seis2.npy`` and ``vel2.npy``."""
    pairs = []
    pattern = os.path.join(family_dir, "**", "seis*.npy")
    for seismic_path in sorted(glob.glob(pattern, recursive=True)):
        filename = os.path.basename(seismic_path)
        velocity_path = os.path.join(
            os.path.dirname(seismic_path), "vel" + filename[4:]
        )
        if os.path.isfile(velocity_path):
            pairs.append((seismic_path, velocity_path))
    return pairs


class SeisVelDataset(Dataset):
    """Lazily load individual seismic and velocity samples from NumPy files."""

    def __init__(self, pairs, idx_list, vel_mean=Cfg.vel_mean, vel_std=Cfg.vel_std):
        """Initialize the dataset with file pairs and flattened sample indices."""
        self.pairs = pairs
        self.idx_list = idx_list
        self.vel_mean = vel_mean
        self.vel_std = vel_std
        # Cache opened memory-mapped arrays within each worker process.
        self._seis_cache = {}
        self._vel_cache = {}

    def __len__(self):
        """Return the number of indexed samples."""
        return len(self.idx_list)

    def _open(self, fi):
        """Open and cache the seismic and velocity arrays for one file pair."""
        if fi not in self._seis_cache:
            sp, vp = self.pairs[fi]
            self._seis_cache[fi] = np.load(sp, mmap_mode="r")
            self._vel_cache[fi] = np.load(vp, mmap_mode="r")
        return self._seis_cache[fi], self._vel_cache[fi]

    def __getitem__(self, i):
        """Load, preprocess, and return one seismic/velocity sample pair."""
        fi, si = self.idx_list[i]
        seis_arr, vel_arr = self._open(fi)

        seis = seis_arr[si].astype(np.float32)   # Shape: (5, 1000, 70).
        vel  = vel_arr[si].astype(np.float32)    # Shape: (70, 70) or (1, 70, 70).

        # Normalize the velocity target.
        vel = (vel - self.vel_mean) / self.vel_std

        # Compress the seismic dynamic range.
        seis = np.log1p(np.abs(seis))

        # Treat the five sources as channels in a 2D convolutional input.
        seis = seis.reshape(Cfg.n_src, Cfg.n_steps, Cfg.n_recv)

        return torch.from_numpy(seis), torch.from_numpy(vel)


def build_flat_indices(pairs):
    """Build flattened ``(file_index, sample_index)`` pairs for all files."""
    indices = []
    for fi, (sp, _) in enumerate(pairs):
        arr = np.load(sp, mmap_mode="r")
        n = arr.shape[0]
        for si in range(n):
            indices.append((fi, si))
    return indices


def resolve_parallel_mode(mode, device, gpu_count):
    """Resolve the requested parallel mode against the available hardware."""
    if mode not in {"single", "data_parallel", "ddp"}:
        raise ValueError(f"Unsupported parallel mode: {mode}")
    if mode == "ddp":
        raise NotImplementedError(
            "DDP is reserved for the distributed training entry point. "
            "Use data_parallel for the current notebook workflow."
        )
    if mode == "data_parallel" and device.startswith("cuda") and gpu_count > 1:
        return "data_parallel"
    return "single"


def wrap_model_for_parallel(model, mode, device):
    """Wrap a model for the selected parallel backend."""
    resolved_mode = resolve_parallel_mode(mode, device, torch.cuda.device_count())
    if resolved_mode == "data_parallel":
        print(f"[info] using DataParallel on {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model), resolved_mode
    return model, resolved_mode


def unwrap_model(model):
    """Return the underlying model from a parallel wrapper."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ---------------------------------------------------------------------------
# U-Net model: input (5, 1000, 70) -> output (1, 70, 70)
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Apply two convolution, batch-normalization, and ReLU blocks."""

    def __init__(self, in_ch, out_ch):
        """Create a double-convolution block with the requested channel sizes."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        """Apply the convolutional block to an input tensor."""
        return self.net(x)

class UNet(nn.Module):
    """Map five-source seismic measurements to a 70 x 70 velocity map."""

    def __init__(self, in_ch=Cfg.n_src, base=Cfg.model_base_channels):
        """Build the encoder, decoder, and output projection layers."""
        super().__init__()
        # Encoder: progressively reduce the time and receiver dimensions.
        self.enc1 = DoubleConv(in_ch, base)        # 1000x70 -> 1000x70
        self.pool1 = nn.MaxPool2d(2, 2)            # -> 500x35
        self.enc2 = DoubleConv(base, base*2)       # 500x35
        self.pool2 = nn.MaxPool2d(2, 2)            # -> 250x17 (pad to 250x18)
        self.enc3 = DoubleConv(base*2, base*4)     # 250x18
        self.pool3 = nn.MaxPool2d(2, 2)            # -> 125x9
        self.enc4 = DoubleConv(base*4, base*8)     # 125x9
        self.pool4 = nn.MaxPool2d(2, 2)             # -> 62x4 (pad to 62x5)
        self.enc5 = DoubleConv(base*8, base*16)    # 62x5

        # Project the bottleneck back to the first decoder resolution.
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base*16, base*8, kernel_size=(9, 14),
                                stride=(2, 2)),    # 62x5 -> 125x10 (approx)
            nn.BatchNorm2d(base*8),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks with skip connections.
        self.dec1 = DoubleConv(base*16, base*8)     # concat with enc4
        self.up2 = nn.ConvTranspose2d(base*8, base*4, kernel_size=4, stride=2, padding=1)
        self.dec2 = DoubleConv(base*8, base*4)
        self.up3 = nn.ConvTranspose2d(base*4, base*2, kernel_size=4, stride=2, padding=1)
        self.dec3 = DoubleConv(base*4, base*2)
        self.up4 = nn.ConvTranspose2d(base*2, base,   kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec4 = DoubleConv(base*2, base)
        self.dec5 = DoubleConv(base, base)

        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        """Run the U-Net and return a tensor with shape ``(B, 70, 70)``."""
        # Input shape: (B, 5, 1000, 70).
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        # Pad the receiver dimension to make the next pooling operation even.
        e2p = nn.functional.pad(e2, (0, 1))         # 17->18
        e3 = self.enc3(self.pool2(e2p))
        e4 = self.enc4(self.pool3(e3))
        e4p = nn.functional.pad(e4, (0, 1))         # 4->5
        e5 = self.enc5(self.pool4(e4p))

        u = self.up(e5)                             # -> 125x10
        # Align the decoder feature map with the skip connection.
        u = nn.functional.interpolate(u, size=(125, 9), mode="nearest")
        d1 = self.dec1(torch.cat([u, e4], dim=1))
        d2 = self.up2(d1)
        d2 = nn.functional.interpolate(d2, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))
        d3 = self.up3(d2)
        d3 = nn.functional.interpolate(d3, size=e2p.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2p], dim=1))
        d4 = self.up4(d3)
        d4 = nn.functional.interpolate(d4, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e1], dim=1))
        d5 = nn.functional.interpolate(d4, size=(Cfg.img_size, Cfg.img_size),
                        mode="bilinear", align_corners=False)
        d5 = self.dec5(d5)
        out = self.head(d5)
        return out.squeeze(1)                       # (B, 70, 70)


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch and return the sample-weighted mean loss."""
    model.train()
    total, n = 0.0, 0
    progress = tqdm(loader, desc="train", leave=False)
    for seis, vel in progress:
        seis, vel = seis.to(device), vel.to(device)
        # seis: (B,5,1000,70)  vel: (B,70,70) or (B,1,70,70)
        vel = vel.squeeze(1) if vel.dim() == 4 else vel

        pred = model(seis)
        loss = criterion(pred, vel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = seis.size(0)
        total += loss.item() * bs
        n += bs
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate the model and return the sample-weighted mean validation loss."""
    model.eval()
    total, n = 0.0, 0
    progress = tqdm(loader, desc="valid", leave=False)
    for seis, vel in progress:
        seis, vel = seis.to(device), vel.to(device)
        vel = vel.squeeze(1) if vel.dim() == 4 else vel
        pred = model(seis)
        loss = criterion(pred, vel)
        bs = seis.size(0)
        total += loss.item() * bs
        n += bs
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


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
    args = parser.parse_args()
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
    print(f"[info] train samples: {len(tr_idx)}, val samples: {len(va_idx)}")

    train_ds = SeisVelDataset(pairs, tr_idx, vel_mean=vel_mean, vel_std=vel_std)
    val_ds   = SeisVelDataset(pairs, va_idx, vel_mean=vel_mean, vel_std=vel_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=Cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=Cfg.num_workers, pin_memory=True)

    # Build the model, optimizer, scheduler, and loss function.
    model = UNet(in_ch=Cfg.n_src, base=Cfg.model_base_channels).to(Cfg.device)
    model, resolved_parallel_mode = wrap_model_for_parallel(
        model, args.parallel_mode, Cfg.device
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
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, Cfg.device)
        va_loss = validate(model, val_loader, criterion, Cfg.device)
        scheduler.step()
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
            torch.save(unwrap_model(model).state_dict(), run_dir / "best_unet.pth")
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
        "run_dir": run_dir,
        "device": Cfg.device,
        "parallel_mode_requested": args.parallel_mode,
        "parallel_mode_resolved": resolved_parallel_mode,
        "gpu_count": torch.cuda.device_count() if Cfg.device.startswith("cuda") else 0,
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
