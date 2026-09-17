"""
Yale/UNC-CH - Geophysical Waveform Inversion
Inference and submission generation script.

Usage (local default paths):
    python infer.py

For Kaggle, override the default paths with --ckpt, --test_dir, and --out.

Equal-weight multi-checkpoint ensemble (the repo's validated best recipe --
121.10 m/s single vs 111.49 m/s with 6 checkpoints on the shared holdout):
    python infer.py --ckpt run_a/best_ema.pth run_a/best_unet.pth \
                           run_b/best_ema.pth run_b/best_unet.pth \
                           run_c/best_ema.pth run_c/best_unet.pth \
                    --test_dir D:/data/test --out submission.csv

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
from model import MODEL_NAMES, ACT_NAMES, build_model, resolve_model_spec
from config import Cfg, load_velocity_stats, resolve_device
from tta import FLIP_VARIANTS, ensemble_predict

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
        seis = np.sign(seis) * np.log1p(np.abs(seis))     # sign·log1p: keep waveform polarity.
        return self.oids[i], torch.from_numpy(seis)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_state(path, device):
    """Load a state_dict, normalizing the wrapper formats train.py may save."""
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
    """Run inference and write predictions in Kaggle submission format."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", nargs="+", default=[str(Cfg.checkpoint_path)],
        help="One or more checkpoint files. Multiple checkpoints are averaged "
             "with equal weight (validated best recipe: 6 checkpoints).",
    )
    parser.add_argument("--test_dir", default=str(Cfg.test_data_dir))
    parser.add_argument("--out",      default=str(Cfg.submission_path))
    parser.add_argument("--batch_size", type=int, default=Cfg.infer_batch_size)
    parser.add_argument("--vel_mean", type=float, default=Cfg.vel_mean)
    parser.add_argument("--vel_std",  type=float, default=Cfg.vel_std)
    parser.add_argument("--device", default=Cfg.device,
                        choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num_workers", type=int, default=Cfg.num_workers,
                        help="DataLoader workers; use 0 on Windows if it errors.")
    parser.add_argument(
        "--model", choices=("auto",) + MODEL_NAMES, default="auto",
        help="Architecture override. 'auto' (default) reads model_type from each "
             "checkpoint's results.json / config.json.",
    )
    parser.add_argument(
        "--act", choices=("auto",) + ACT_NAMES, default="auto",
        help="Hidden-activation override. 'auto' reads it from the run metadata, "
             "defaulting to 'relu' for runs saved before 2026-09-15.",
    )
    parser.add_argument(
        "--tta", choices=tuple(FLIP_VARIANTS), default="none",
        help="Flip test-time augmentation per model before ensembling "
             "(see tta.py; the geometry-consistent variants are src_recv / "
             "time_recv / time_src_recv).",
    )
    parser.add_argument(
        "--stats_path",
        default=None,
        help="Optional velocity-statistics JSON path; defaults to the directory "
             "of the first checkpoint (velocity_stats.json is stored there by "
             "train.py).",
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    stats_path = (
        Path(args.stats_path)
        if args.stats_path
        else Path(args.ckpt[0]).resolve().parent / "velocity_stats.json"
    )
    if stats_path.is_file():
        args.vel_mean, args.vel_std = load_velocity_stats(stats_path)
        print(f"[info] loaded velocity statistics: {stats_path}")
    else:
        print(f"[warn] no statistics JSON at {stats_path}; using --vel_mean/--vel_std "
              f"({args.vel_mean:.2f}/{args.vel_std:.2f})")

    device = resolve_device(args.device)
    print(f"[info] device: {device}")

    # Load one or more trained models (equal-weight ensemble when several).
    # Denormalisation constants are resolved PER checkpoint: runs launched
    # without WAVEFORM_OUTPUT_ROOT trained against Cfg.vel_mean/Cfg.vel_std
    # instead of the shared stats file, so a single global constant would
    # mis-scale those members. An explicit --stats_path still forces one pair.
    models = []
    inverse_scales = []
    for ckpt in args.ckpt:
        spec_model, spec_act, spec_out, spec_base = resolve_model_spec(ckpt)
        if args.model != "auto":
            spec_model = args.model
        if args.act != "auto":
            spec_act = args.act
        model = build_model(name=spec_model, in_ch=Cfg.n_src,
                            base=spec_base,
                            act=spec_act, out_activation=spec_out).to(device)
        model.load_state_dict(load_state(ckpt, device))
        model.eval()
        models.append(model)

        ckpt_mean, ckpt_std = args.vel_mean, args.vel_std
        if not args.stats_path:
            run_stats = Path(ckpt).resolve().parent / "velocity_stats.json"
            if run_stats.is_file():
                ckpt_mean, ckpt_std = load_velocity_stats(run_stats)
        inverse_scales.append((ckpt_mean, ckpt_std))
        print(f"[info] loaded checkpoint: {ckpt} "
              f"(model={spec_model} act={spec_act} base={spec_base} "
              f"norm=(mean={ckpt_mean:.2f}, std={ckpt_std:.2f}))")
    print(f"[info] ensemble size: {len(models)} (equal-weight average)")
    print(f"[info] tta: {args.tta}")

    # Prepare the test data loader.
    ds = TestDataset(args.test_dir)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Generate velocity predictions.
    oid_list = []
    preds = []   # Store denormalized predictions with shape (B, 70, 70).
    for oids, seis in tqdm(loader, desc="inference"):
        seis = seis.to(device)                     # (B,5,1000,70)
        # ensemble_predict with scales averages in RAW m/s (see tta.py).
        pred = ensemble_predict(seis, models, args.tta,
                                scales=inverse_scales)   # (B,70,70) raw m/s
        preds.append(pred.cpu().numpy())
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
