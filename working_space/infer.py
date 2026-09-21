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

Multi-GPU (one OS process per GPU; the test set is sharded BY FILE and the parts
are merged back into the original file order, so the result matches a
single-process run -- mirrors train.py's flag):
    python infer.py --ckpt run/best_ema.pth --test_dir D:/data/test \
                    --out submission.csv --nproc_per_node 2

The same sharding can be driven by hand (e.g. one shell per GPU), in which case
each process writes its own file and nothing is merged automatically:
    CUDA_VISIBLE_DEVICES=0 python infer.py ... --shard_index 0 --shard_count 2 \
                              --out part0.csv

Output format (matching sample_submission.csv):
  oid_ypos,x_1,x_3,...,x_69
  000039dca2_y_0,3000.0,3000.0,...,3000.0
  ...
"""

import os
import glob
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
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
# Submission writing
# ---------------------------------------------------------------------------
SUBMISSION_HEADER = (
    "oid_ypos,x_1,x_3,x_5,x_7,x_9,x_11,x_13,x_15,x_17,x_19,"
    "x_21,x_23,x_25,x_27,x_29,x_31,x_33,x_35,x_37,x_39,"
    "x_41,x_43,x_45,x_47,x_49,x_51,x_53,x_55,x_57,x_59,"
    "x_61,x_63,x_65,x_67,x_69\n"
)


def write_submission(out_path, oid_list, preds):
    """Write predictions in the Kaggle format (odd x-columns only).

    ``preds`` has shape (N, 70, 70); the first axis is ``y`` (depth) and the
    second is ``x`` (receiver position), matching submission row ``<oid>_y_<y>``.
    """
    odd_cols = preds[:, :, Cfg.submission_x_start:Cfg.submission_x_stop:Cfg.submission_x_step]
    with open(out_path, "w") as f:
        f.write(SUBMISSION_HEADER)
        for i, oid in enumerate(oid_list):
            for y in range(odd_cols.shape[1]):     # 70 rows
                row_vals = ",".join(f"{v:.1f}" for v in odd_cols[i, y])
                f.write(f"{oid}_y_{y},{row_vals}\n")


def merge_submission_parts(out_path, part_paths, ordered_oids):
    """Concatenate the per-rank part files into one submission file.

    Rows are emitted in ``ordered_oids`` order (the test directory's sorted file
    order), NOT in shard order, so the merged file is indistinguishable from a
    single-process run. Each object is predicted by exactly one rank, so a
    missing key means a shard died -- that raises instead of writing a file that
    would silently score badly.
    """
    rows = {}
    for part in part_paths:
        if not os.path.isfile(part):
            continue
        with open(part) as f:
            next(f, None)                          # skip the header
            for line in f:
                oid = line.split(",", 1)[0].rsplit("_y_", 1)[0]
                rows.setdefault(oid, []).append(line)
    missing = [oid for oid in ordered_oids if oid not in rows]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(ordered_oids)} test objects were not "
            f"predicted (e.g. {missing[:3]}); refusing to write a partial "
            f"submission -- check the shard logs above."
        )
    with open(out_path, "w") as f:
        f.write(SUBMISSION_HEADER)
        for oid in ordered_oids:
            f.writelines(rows[oid])


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------
def main():
    """Parse arguments, then infer on one process (or one process per GPU)."""
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
        help="Optional velocity-statistics JSON. When given it FORCES that single "
             "mean/std pair onto every checkpoint. When omitted (recommended) "
             "each checkpoint is denormalised with the velocity_stats.json of its "
             "own run directory, falling back to --vel_mean/--vel_std.",
    )
    parser.add_argument(
        "--nproc_per_node", type=int, default=1,
        help="Number of processes, one per GPU (mirrors train.py's flag). "
             "1 (default) = the original single-process run. N > 1 shards the "
             "test set by file across N GPUs and merges the part files back into "
             "the original order, so predictions match the single-process run. "
             "Falls back to 1 when CUDA is unavailable or fewer GPUs are visible.",
    )
    parser.add_argument(
        "--shard_index", type=int, default=-1,
        help="Run only THIS shard of the test set (0-based). Normally set by "
             "--nproc_per_node, which launches one process per GPU; you can also "
             "run the shards yourself (see the module docstring). Each shard "
             "writes its own --out file and nothing is merged for you.",
    )
    parser.add_argument(
        "--shard_count", type=int, default=1,
        help="Total number of shards the test set is split into (with "
             "--shard_index).",
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.shard_index >= 0:
        # Worker mode: one shard, launched by --nproc_per_node or by hand.
        count = max(1, args.shard_count)
        if not 0 <= args.shard_index < count:
            raise SystemExit(f"--shard_index {args.shard_index} is outside "
                             f"--shard_count {count}")
        infer_worker(args.shard_index, count, args)
        return

    nproc = max(1, int(args.nproc_per_node))
    if nproc > 1 and resolve_device(args.device).type != "cuda":
        print(f"[warn] --nproc_per_node {nproc} needs CUDA (got --device "
              f"{args.device}); running a single process.")
        nproc = 1
    if nproc > 1:
        available = torch.cuda.device_count()
        if nproc > available:
            print(f"[warn] --nproc_per_node {nproc} exceeds the {available} "
                  f"visible GPU(s); using {available}.")
            nproc = max(1, available)

    if nproc == 1:
        infer_worker(0, 1, args)
        return

    # One OS process per GPU, each pinned with CUDA_VISIBLE_DEVICES=<rank>. Rank r
    # predicts every nproc-th test file into its own part file; main() merges the
    # parts in the original file order, so the submission matches the
    # single-process run (unlike DataParallel, no sample is split across
    # processes).
    #
    # subprocess rather than torch.multiprocessing.spawn(): spawn switches the
    # multiprocessing start method, so the DataLoader workers are recreated with a
    # fresh interpreter every time and the loader starves the loop -- measured
    # 6.2 it/s per rank that way vs 21 it/s for the same shard run as an
    # independent process. A forked mp.spawn is also unsafe here because the
    # parent has already initialised CUDA (resolve_device / device_count).
    part_paths = [f"{args.out}.part{rank}" for rank in range(nproc)]
    script = str(Path(__file__).resolve())
    children = []
    for rank in range(nproc):
        cmd = [sys.executable, script, *sys.argv[1:],
               "--shard_index", str(rank), "--shard_count", str(nproc),
               "--out", part_paths[rank]]
        print(f"[info] shard {rank + 1}/{nproc} on GPU {rank}")
        children.append(subprocess.Popen(cmd, env=dict(os.environ,
                                                       CUDA_VISIBLE_DEVICES=str(rank))))
    codes = [child.wait() for child in children]
    if any(codes):
        raise SystemExit(f"shard processes failed (exit codes {codes}); part "
                         f"files kept for debugging, nothing was merged")

    ordered_oids = TestDataset(args.test_dir).oids
    merge_submission_parts(args.out, part_paths, ordered_oids)
    for path in part_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    print(f"[done] saved → {args.out} (merged {nproc} shards, "
          f"{len(ordered_oids)} objects)")


@torch.no_grad()
def infer_worker(local_rank, world_size, args):
    """Run inference over this rank's shard of the test set.

    ``world_size == 1`` is the plain single-process path. ``world_size > 1`` runs
    one shard (files ``local_rank``, ``local_rank + world_size``, ...) on the GPU
    named by ``CUDA_VISIBLE_DEVICES`` and writes it to this shard's --out file;
    the launcher merges the parts afterwards.
    """
    sharded = world_size > 1
    device = resolve_device(args.device)
    if sharded and device.type == "cuda":
        # The launcher pins this process with CUDA_VISIBLE_DEVICES=<rank>, so its
        # only visible device is 0.
        torch.cuda.set_device(0)
    tag = f"[r{local_rank}] " if sharded else ""
    print(f"{tag}[info] device: {device} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

    # Denormalisation constants, resolved PER checkpoint with this priority:
    #   1. an explicit --stats_path, which forces ONE pair onto every member;
    #   2. velocity_stats.json in the checkpoint's own run directory;
    #   3. the --vel_mean/--vel_std given on the command line (Cfg defaults).
    # The CLI pair is captured BEFORE any file is read: otherwise the first
    # checkpoint's file would silently become the fallback for all the others.
    cli_mean, cli_std = args.vel_mean, args.vel_std
    if args.stats_path:
        forced = load_velocity_stats(args.stats_path)
        print(f"{tag}[info] forcing one statistics pair ({args.stats_path}) on every "
              f"checkpoint: mean={forced[0]:.2f} std={forced[1]:.2f}")
    else:
        forced = None
        print(f"{tag}[info] per-checkpoint velocity statistics "
              f"(fallback mean={cli_mean:.2f} std={cli_std:.2f})")

    # Load one or more trained models (equal-weight ensemble when several).
    # Each member is denormalised with its own constants and the ensemble
    # averages in RAW m/s (see tta.py::ensemble_predict), so members trained in
    # different normalisation spaces still combine correctly.
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
                            act=spec_act, out_activation=spec_out,
                            pretrained=False).to(device)
        model.load_state_dict(load_state(ckpt, device))
        model.eval()
        models.append(model)

        run_stats = Path(ckpt).resolve().parent / "velocity_stats.json"
        if forced is not None:
            ckpt_mean, ckpt_std, norm_src = forced[0], forced[1], "forced"
        elif run_stats.is_file():
            ckpt_mean, ckpt_std = load_velocity_stats(run_stats)
            norm_src = "run"
        else:
            ckpt_mean, ckpt_std, norm_src = cli_mean, cli_std, "fallback"
        inverse_scales.append((ckpt_mean, ckpt_std))
        print(f"{tag}[info] loaded checkpoint: {ckpt} "
              f"(model={spec_model} act={spec_act} base={spec_base} "
              f"norm=(mean={ckpt_mean:.2f}, std={ckpt_std:.2f}) from {norm_src})")
    print(f"{tag}[info] ensemble size: {len(models)} (equal-weight average)")
    print(f"{tag}[info] tta: {args.tta}")

    # Shard the test set: rank r takes files r, r+world_size, r+2*world_size, ...
    full = TestDataset(args.test_dir)
    if sharded:
        index = list(range(local_rank, len(full), world_size))
        print(f"{tag}[info] shard {local_rank + 1}/{world_size}: {len(index)} "
              f"of {len(full)} test files")
        ds = Subset(full, index)
    else:
        ds = full

    # The launcher passes this process its own --out (a part file that the parent
    # merges); running a shard by hand keeps whatever --out the user gave.
    out_path = args.out
    if len(ds) == 0:                       # fewer test files than processes
        write_submission(out_path, [],
                         np.zeros((0, Cfg.img_size, Cfg.img_size), dtype=np.float32))
        print(f"{tag}[done] empty shard → {out_path}")
        return

    # Prepare the test data loader.
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Generate velocity predictions.
    oid_list = []
    preds = []   # Store denormalized predictions with shape (B, 70, 70).
    desc = f"inference r{local_rank}" if sharded else "inference"
    for oids, seis in tqdm(loader, desc=desc):
        seis = seis.to(device)                     # (B,5,1000,70)
        # ensemble_predict with scales averages in RAW m/s (see tta.py).
        pred = ensemble_predict(seis, models, args.tta,
                                scales=inverse_scales)   # (B,70,70) raw m/s
        preds.append(pred.cpu().numpy())
        oid_list.extend(oids)

    preds = np.concatenate(preds, axis=0)          # (N, 70, 70)
    print(f"{tag}[info] predictions shape: {preds.shape}")

    write_submission(out_path, oid_list, preds)
    print(f"{tag}[done] saved → {out_path}")


if __name__ == "__main__":
    main()
