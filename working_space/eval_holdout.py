"""Unified file-level holdout evaluation & greedy ensemble selection.

Protocol: the holdout is built with the SAME file-level split used at training
time (train.py --split_seed, default Cfg.val_split_seed = 777), so models
trained with that split never saw the holdout files. It reports per-model raw
MAE on the shared holdout and runs greedy forward ensemble selection (equal
weight mean of predictions), which is the repo's validated integration lever.

Usage (run from working_space; WAVEFORM_OUTPUT_ROOT should point at repo/output
so the shared velocity-statistics JSON is found):
    python eval_holdout.py \\
        --ckpts output/all/model_260909_xxxx output/all/model_260909_yyyy ... \\
        [--weight ema|online] [--batch_size 32] [--split_seed 777]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (Cfg, load_velocity_stats, select_families,  # noqa: E402
                    stats_path_for_families)
from data import SeisVelDataset, build_flat_indices, find_pairs  # noqa: E402
from model import UNet  # noqa: E402


def load_state(path, device):
    """Load a state_dict from a checkpoint, normalizing common formats."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if isinstance(checkpoint, dict) and any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(Cfg.train_data_dir))
    parser.add_argument(
        "--ckpts", nargs="+", default=None,
        help="Run directories (best_ema.pth/best_unet.pth inside) or direct "
             "checkpoint paths. If omitted, auto-discover runs under "
             "--discover_root whose results.json split_seed matches --split_seed.",
    )
    parser.add_argument(
        "--discover_root",
        default=str(Path(Cfg.project_root) / "output" / "all"),
        help="Directory scanned for runs when --ckpts is not given.",
    )
    parser.add_argument("--weight", choices=("ema", "online"), default="ema",
                        help="Which checkpoint to load from each run dir.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--split_seed", type=int, default=Cfg.val_split_seed,
        help="Must match --split_seed used to train the models (default 777).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    families = select_families("all")
    stats_path = stats_path_for_families(families)
    if stats_path.is_file():
        vel_mean, vel_std = load_velocity_stats(stats_path)
    else:
        vel_mean, vel_std = Cfg.vel_mean, Cfg.vel_std
        print(f"[warn] stats not found at {stats_path}; using Cfg defaults")
    print(f"[holdout] split_seed={args.split_seed}  vel_mean={vel_mean:.2f}  "
          f"vel_std={vel_std:.2f}")

    # Build the shared file-level holdout exactly like train.py.
    pairs = find_pairs(args.data_dir, families)
    file_ids = list(range(len(pairs)))
    _, va_files = train_test_split(file_ids, test_size=Cfg.val_ratio,
                                   random_state=args.split_seed)
    va_set = set(va_files)
    indices = build_flat_indices(pairs)
    va_idx = [idx for idx in indices if idx[0] in va_set]
    del indices
    print(f"[holdout] files={len(pairs)}  holdout files={len(va_files)}  "
          f"holdout samples={len(va_idx)}")

    holdout_ds = SeisVelDataset(pairs, va_idx, vel_mean=vel_mean, vel_std=vel_std,
                                train=False)
    loader = torch.utils.data.DataLoader(holdout_ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)

    # Resolve checkpoints: explicit --ckpts OR auto-discover runs whose
    # results.json split_seed matches (i.e. they were trained on the same holdout).
    if args.ckpts:
        entries = args.ckpts
    else:
        root = Path(args.discover_root)
        if not root.is_dir():
            raise SystemExit(f"Discovery root not found: {root}")
        entries = []
        for run_dir in sorted(root.glob("model_*")):
            results_path = run_dir / "results.json"
            if not results_path.is_file():
                continue
            try:
                meta = json.loads(results_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if meta.get("split_seed") != args.split_seed:
                continue
            entries.append(str(run_dir))
        print(f"[discover] {len(entries)} run(s) with split_seed={args.split_seed} "
              f"under {root}")

    # Load models.
    ckpt_paths = []
    for entry in entries:
        p = Path(entry)
        path = p if p.is_file() else (p / ("best_ema.pth" if args.weight == "ema"
                                           else "best_unet.pth"))
        if not path.is_file():
            print(f"[skip] checkpoint not found: {path}")
            continue
        model = UNet(in_ch=Cfg.n_src, base=Cfg.model_base_channels).to(device)
        model.load_state_dict(load_state(str(path), device))
        model.eval()
        ckpt_paths.append((str(path), model))
    if not ckpt_paths:
        raise SystemExit("No usable checkpoints were provided.")
    print(f"[holdout] models to evaluate: {len(ckpt_paths)}")

    # One deterministic pass: collect targets + per-model normalized predictions.
    targets = []
    per_model = {path: [] for path, _ in ckpt_paths}
    with torch.no_grad():
        for seismic, velocity in loader:
            velocity = velocity.squeeze(1) if velocity.dim() == 4 else velocity
            targets.append(velocity.float())
            for path, model in ckpt_paths:
                per_model[path].append(model(seismic.to(device)).float().cpu())
    targets = torch.cat(targets, dim=0)
    preds = {path: torch.cat(tensors, dim=0) for path, tensors in per_model.items()}
    del per_model

    def raw_mae(tensor):
        return (tensor - targets).abs().mean().item() * vel_std

    print("\n=== Single-model holdout MAE (m/s) ===")
    single = {}
    for path, pred in preds.items():
        single[path] = raw_mae(pred)
        print(f"  {Path(path).parent.name if not Path(path).is_file() else path}: {single[path]:.2f}")

    def ensemble_pred(keys):
        return torch.stack([preds[k] for k in keys], dim=0).mean(dim=0)

    print("\n=== Greedy forward ensemble selection (equal-weight mean) ===")
    chosen = []
    base_mae = float("inf")
    while len(chosen) < len(preds):
        current = ensemble_pred(chosen) if chosen else None
        current_mae = raw_mae(current) if current is not None else float("inf")
        best_gain = (current_mae, None)
        for path in preds:
            if path in chosen:
                continue
            if chosen:
                trial = torch.stack([ensemble_pred(chosen), preds[path]], dim=0).mean(dim=0)
            else:
                trial = preds[path]
            trial_mae = raw_mae(trial)
            if trial_mae < best_gain[0]:
                best_gain = (trial_mae, path)
        if best_gain[1] is None or best_gain[0] >= current_mae:
            break  # no model improves the current ensemble
        chosen.append(best_gain[1])
        name = Path(best_gain[1]).parent.name
        print(f"  + {name:<22s} ensemble MAE = {best_gain[0]:.2f}  (n={len(chosen)})")

    if chosen:
        final_name = " + ".join(Path(p).parent.name for p in chosen)
        print(f"\nBest ensemble ({len(chosen)} models): {final_name}")
        print(f"  holdout MAE = {raw_mae(ensemble_pred(chosen)):.2f} m/s")
    else:
        print("\nNo ensemble improved over the best single model.")


if __name__ == "__main__":
    main()
