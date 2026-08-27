"""
Yale/UNC-CH - Geophysical Waveform Inversion
Inference and submission generation script.

Usage (local default paths):
    python infer.py

For Kaggle, override the default paths with --ckpt, --test_dir, and --out.

Output format (matching sample_submission.csv):
  oid_ypos,x_1,x_3,...,x_69
  000039dca2_y_0,3000.0,3000.0,...,3000.0
  ...
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import UNet
from config import Cfg, load_velocity_stats, resolve_device

# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------
class TestDataset(Dataset):
    """Load one complete seismic sample from each NumPy file."""

    def __init__(self, test_dir):
        """Index test files in the given directory and extract their object IDs."""
        self.files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {test_dir}")
        # Extract object IDs from filenames.
        self.oids = [os.path.splitext(os.path.basename(f))[0] for f in self.files]
        print(f"[info] test files: {len(self.files)}")

    def __len__(self):
        """Return the number of test files."""
        return len(self.files)

    def __getitem__(self, i):
        """Load and preprocess one seismic sample."""
        seis = np.load(self.files[i]).astype(np.float32)   # Shape: (5, 1000, 70).
        seis = np.log1p(np.abs(seis))                     # Match the training transform.
        return self.oids[i], torch.from_numpy(seis)


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
    """Run inference and write predictions in Kaggle submission format."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default=str(Cfg.checkpoint_path))
    parser.add_argument("--test_dir", default=str(Cfg.test_data_dir))
    parser.add_argument("--out",      default=str(Cfg.submission_path))
    parser.add_argument("--batch_size", type=int, default=Cfg.infer_batch_size)
    parser.add_argument("--vel_mean", type=float, default=Cfg.vel_mean)
    parser.add_argument("--vel_std",  type=float, default=Cfg.vel_std)
    parser.add_argument(
        "--stats_path",
        default=None,
        help="Optional velocity-statistics JSON path; defaults to the checkpoint directory.",
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    stats_path = (
        Path(args.stats_path)
        if args.stats_path
        else Path(args.ckpt).resolve().parent / "velocity_stats.json"
    )
    if stats_path.is_file():
        args.vel_mean, args.vel_std = load_velocity_stats(stats_path)
        print(f"[info] loaded velocity statistics: {stats_path}")

    device = resolve_device()

    # Load the trained model.
    model = UNet(in_ch=Cfg.n_src, base=Cfg.model_base_channels).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[info] loaded checkpoint: {args.ckpt}")

    # Prepare the test data loader.
    ds = TestDataset(args.test_dir)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=Cfg.num_workers
    )

    # Generate velocity predictions.
    oid_list = []
    preds = []   # Store denormalized predictions with shape (B, 70, 70).
    for oids, seis in tqdm(loader, desc="inference"):
        seis = seis.to(device)                     # (B,5,1000,70)
        pred = model(seis)                         # (B,70,70)
        pred = pred.cpu().numpy() * args.vel_std + args.vel_mean
        preds.append(pred)
        oid_list.extend(oids)

    preds = np.concatenate(preds, axis=0)          # (N, 70, 70)
    print(f"[info] predictions shape: {preds.shape}")

    # Write the submission using only odd x-columns.
    odd_cols = preds[:, :, Cfg.submission_x_start:Cfg.submission_x_stop:Cfg.submission_x_step]
    out_path = args.out
    with open(out_path, "w") as f:
        f.write("oid_ypos,x_1,x_3,x_5,x_7,x_9,x_11,x_13,x_15,x_17,x_19,"
                "x_21,x_23,x_25,x_27,x_29,x_31,x_33,x_35,x_37,x_39,"
                "x_41,x_43,x_45,x_47,x_49,x_51,x_53,x_55,x_57,x_59,"
                "x_61,x_63,x_65,x_67,x_69\n")
        for i, oid in enumerate(oid_list):
            for y in range(odd_cols.shape[1]):     # 70 rows
                row_vals = ",".join(f"{v:.1f}" for v in odd_cols[i, y])
                f.write(f"{oid}_y_{y},{row_vals}\n")
    print(f"[done] saved → {out_path}")


if __name__ == "__main__":
    main()
