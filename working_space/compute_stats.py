"""
compute_stats.py
Compute velocity-map statistics for target normalization.
After running, copy the reported mean and standard deviation to config.py.

Usage:
    python compute_stats.py
    python compute_stats.py --data_dir /kaggle/input/waveform-inversion/train_samples
"""

import os, glob, argparse, json
import numpy as np
from tqdm.auto import tqdm
from config import Cfg, select_families, stats_path_for_families

def main():
    """Compute summary statistics for all velocity files under the data root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(Cfg.train_data_dir))
    parser.add_argument(
        "--family",
        default=None,
        help="Case-insensitive family keyword(s), comma-separated, or 'all'.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path for the computed statistics.",
    )
    args = parser.parse_args()
    selected_families = select_families(args.family)

    files = []
    for root, _, fs in os.walk(args.data_dir):
        for f in fs:
            if f.endswith(".npy") and (
                "model" in f.lower() or f.lower().startswith("vel")
            ):
                files.append(os.path.join(root, f))
    files = sorted(set(files))
    files = [
        path for path in files
        if any(os.sep + family.lower() + os.sep in path.lower() for family in selected_families)
    ]
    print(f"[info] selected families: {', '.join(selected_families)}")
    print(f"[info] found {len(files)} velocity .npy files")

    if not files:
        raise RuntimeError("No velocity files found for the selected families")

    vals = []
    for fp in tqdm(files, desc="velocity statistics"):
        arr = np.load(fp)                # Shape: (500, 70, 70) or (500, 1, 70, 70).
        arr = arr.reshape(arr.shape[0], -1)
        vals.append(arr.ravel())
    vals = np.concatenate(vals)
    print(f"[stats] count   = {vals.size}")
    print(f"[stats] mean    = {vals.mean():.2f}")
    print(f"[stats] std     = {vals.std():.2f}")
    print(f"[stats] min     = {vals.min():.2f}")
    print(f"[stats] max     = {vals.max():.2f}")
    print(f"[stats] median  = {np.median(vals):.2f}")
    stats_path = (
        os.path.abspath(args.out)
        if args.out
        else str(stats_path_for_families(selected_families))
    )
    stats = {
        "families": selected_families,
        "data_dir": os.path.abspath(args.data_dir),
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median": float(np.median(vals)),
    }
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    with open(stats_path, "w") as file:
        json.dump(stats, file, indent=2)
    print(f"[done] saved velocity statistics to {stats_path}")

if __name__ == "__main__":
    main()
