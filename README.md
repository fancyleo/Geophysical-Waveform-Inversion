# Geophysical Waveform Inversion

A PyTorch U-Net baseline for the Yale/UNC-CH Geophysical Waveform Inversion
Kaggle competition. The model maps five-source seismic measurements to a
`70 x 70` velocity model.

## Project Layout

```text
.
├── input/waveform-inversion/
│   ├── train_samples/
│   ├── test/
│   └── sample_submission.csv
├── working_space/
│   ├── config.py       # Shared paths, hyperparameters, and path resolution
│   ├── data.py         # File pairing, lazy dataset, and sample indices
│   ├── data_aug.py     # Augmentation shim (re-exports pretrain)
│   ├── pretrain.py     # Data augmentation transforms (registry pattern)
│   ├── aug_config.py   # Augmentation config registry + winner write-back
│   ├── model.py        # U-Net architecture
│   ├── training.py     # Train/validate loops and parallelization helpers
│   ├── utils.py        # Memory logging and artifact persistence
│   ├── train.py        # Training entry point
│   ├── infer.py        # Test inference and submission generation
│   ├── forward_model.py# Forward modeling: velocity -> seismic (2D acoustic FDTD)
│   ├── compute_stats.py# Velocity normalization statistics
│   ├── test_unet.py    # U-Net shape/gradient smoke test
│   ├── smoke_test.py   # Data pairing and submission format test
│   ├── unet/           # Exploration notebooks + experiment writeups
│   └── 1st_place_analysis.md
├── output/             # Training run artifacts (model_*/best_unet.pth, submissions)
└── kaggle_output/      # Kaggle-side scripts and artifacts
```

## Data Format

The training data is expected to contain paired seismic and velocity files:

```text
train_samples/
├── FlatVel_A/data/*.npy
├── FlatVel_A/model/*.npy
├── FlatFault_A/seis*.npy
├── FlatFault_A/vel*.npy
└── ...
```

Typical tensor shapes are:

```text
Seismic input:   (N, 5, 1000, 70)
Velocity target: (N, 70, 70) or (N, 1, 70, 70)
Test sample:     (5, 1000, 70)
```

## Configuration

Edit [working_space/config.py](working_space/config.py) to change shared
paths, dataset dimensions, model width, training settings, normalization
statistics, data families, or submission columns.

Defaults are project-relative and resolve to:

```text
Training data: input/waveform-inversion/train_samples
Test data:     input/waveform-inversion/test
Outputs:       output
```

The data root is resolved in this order:

1. `WAVEFORM_DATA_ROOT`, when set
2. The local project path `input/waveform-inversion`
3. A valid dataset directory under `/kaggle/input`
4. A valid dataset directory under `/kaggle/competition`
5. The local project path as a deferred fallback for model-only tests

`WAVEFORM_OUTPUT_ROOT` can be used to override the output directory. Otherwise
`/kaggle/working` is used when available, and local runs use `output`.
Command-line arguments override the relevant defaults, which makes the same
scripts usable with Kaggle paths.

## Data Augmentation

Augmentation lives in its own module,
[working_space/data_aug.py](working_space/data_aug.py), using a registry
pattern so new transforms can be added without touching the dispatch logic.

Each augmentation is a pure function with the signature:

```python
fn(seismic, velocity, rng, **params) -> (seismic, velocity)
```

applied to the *raw* physical traces — before `abs`/`log1p` and before the
velocity normalization — so physics-based transforms (time shift, amplitude
scaling, noise) remain valid. Augmentation runs on the training set only; the
validation set is never augmented.

The active set is controlled by `Cfg.augmentations` in `config.py`:

```python
augmentations = {
    "xflip": {"prob": 0.5},
    "time_shift": {"prob": 0.5, "max_shift": 100},
    # "noise": {"prob": 0.5, "sigma": 0.01},
    # "receiver_dropout": {"prob": 0.3, "drop_ratio": 0.15},
    # "amplitude_scale": {"prob": 0.5, "low": 0.85, "high": 1.15},
    # "source_dropout": {"prob": 0.3},
}
```

Each key maps to the parameters passed to the matching registered function;
`prob` is the per-sample application probability. Set `prob` to `0` or remove
a key to disable that augmentation.

Registered transforms:

- `xflip` — mirror the receiver axis and the velocity model horizontally.
- `time_shift` — shift the time axis with zero padding (source excitation
  delay); uses pad + slice instead of `roll` to avoid non-physical wrap-around.
- `noise` — additive Gaussian observation noise (`sigma` is relative to the
  trace's own standard deviation).
- `receiver_dropout` — zero random receiver traces (dead-channel simulation).
- `amplitude_scale` — scale trace amplitudes (source-strength variation).
- `source_dropout` — zero one random source channel (multi-source redundancy).

To add a new augmentation:

1. Write a function in `data_aug.py` with the signature above.
2. Decorate it with `@register_aug("name")`.
3. Enable it under `Cfg.augmentations` in `config.py`.

> **Note on `xflip`**: it currently flips `velocity[:, ::-1]`, assuming the
> second dimension of the velocity model runs along the receiver (horizontal)
> axis. If a visualization shows the horizontal axis is the first dimension,
> change that line to `velocity[::-1, :]`.

**Final augmentation conclusion**: after a full exploration sweep (15ep / 20ep
/ 30% held-out / 60ep 2-Fold, all file-level), **no augmentation beats plain
training**. The production config is `augmentations = {}`, frozen by writing
`output/aug_explore/winner_aug.json`. `xflip` is the most harmful (the problem
has no left-right symmetry — the 5 sources sit at fixed asymmetric positions),
and xflip TTA is inapplicable because the model is not flip-equivariant.

`aug_config.py` keeps the candidate registry (`AUG_CONFIGS`) and the
`write_winner_aug()` / `clear_winner_aug()` helpers that toggle the override
file consumed by `train.py`.

## Experiments & Findings (UNet case close)

All exploration uses a **file-level split + multi-fold** protocol (2-Fold CV,
60 epochs) — sample-level K-Fold overstates gains due to within-file leakage.
Full write-up: `working_space/unet/unet_experiment.md`.

### Production configuration (finalized)

| Item | Value |
|---|---|
| Model | U-Net base=32 (~24.3M params, `model.py`) |
| Input features | **`sign(x)·log(1+|x|)`** (5-source, `(5,1000,70)`) — adopted after 14th-place A/B |
| Augmentation | **none** (frozen via `winner_aug.json`) |
| Training | 60 epochs, AdamW lr=1e-3, CosineAnnealingLR, batch 16, AMP fp16 |
| Target norm | `(v - 2916.82) / 817.36` |
| EMA | decay=0.999 on full-data formal training (EMA best 211.0) |

### Best submission (3-model ensemble)

`formal_1042 + cloud_0501 + formal_ema` → file-level holdout (5000 samples)
**val MAE = 139.1** → `output/submission_ensformal_1042+cloud_0501+formal_ema.csv`
(4,607,260 rows, validated).

### Verified dead ends (protocol-confirmed)

- **Augmentation** (xflip / time_shift / noise / amp / all): none best.
- **xflip TTA**: inapplicable (no left-right symmetry).
- **Gradient / envelope features**: no consistent gain.
- **Hard-family data weighting**: invalid (+4.3 m/s).
- **Regularization** (dropout / WD): dropout harmful, WD marginal.
- **Larger base_channels** (48/64): unstable on small folds.
- **Half-data EMA models in ensemble**: too weak.

### Effective directions

- **Multi-model ensemble**: 2 models -7.7, 3 models -6.0 (~4%).
- **EMA on full-data training**: best single model (211.0).

### ⚠️ Data scale caveat

Local `train_samples/` is a **partial download (~1/47)**: ~10k samples per
run vs. the competition's ~470k (test 65,818 matches). All experiments ran on
this subset; full-data retraining is a large open lever.

### Borrowed from the 14th-place writeup (Ruby) — verified ✅

Low-cost A/B in `working_space/unet/improve_ruby.ipynb` (60ep × file-level
2-Fold, baseline = none 271.2±2.0). **`sign·log1p` is the largest single win
so far and is now the production preprocessing.**

| Config | best_mae | vs none | note |
|---|---|---|---|
| **`sign(x)·log(1+\|x\|)`** | **250.0±1.9** | **-21.2 (7.8%)** | keep waveform polarity → **adopted** |
| `sign_depth` (`sign·log1p` + depth L1) | 250.8±1.1 | -20.4 | depth adds nothing on 5ch sign |
| `sign_cmp_depth` (10ch, + depth) | 253.0±2.0 | -18.2 | depth helps on 10ch (-6.9 vs sign_cmp) |
| `sign_cmp` (10ch channel-split) | 259.9±1.2 | -11.3 | 10ch is worse than 5ch on sign |
| `cmp_reorder` (raw 10ch channel-split) | 262.8±0.2 | -8.4 | see geometry note below |
| Depth-weighted L1 (1 -> 1/4) | 265.7±0.2 | -5.5 | deeper rows are noisier |

> **Conclusions**: channel-split (10ch) is **harmful on top of `sign·log1p`**
> (+9.9 vs 5ch), and depth-weighting adds nothing on 5ch sign; the best
> config is plain **`sign·log1p`** (5ch, no depth, no channel-split).
>
> **Geometry note**: source positions were verified from the data (first-arrival
> V-shape) = receiver indices `[0,17,34,52,69]`, receivers uniform `[0..69]`.
> Because `CMP = (src+recv)/2` is strictly monotonic per source, the CMP
> permutation is the **identity** — Ruby's reorder degenerates to an
> even/odd receiver channel-split (masked, keeps receiver dim 70).

High-ROI, powered by the **forward simulator** (now in repo):
- `working_space/forward_model.py` — 2D acoustic FDTD (24th-order space, 2nd-order
  time + ABC), ported from jaewook704's `vel-to-seis` notebook. `vel_to_seis(vel)
  -> (5,1000,70)`. **Validated**: corr 0.995 vs real FlatVel_A gather, ~3.3 s/sample.
- **Reconstruction-error optimization** at inference `x -= λ·(F(M(x)) - x)` (~21%)
- Forward-simulated self-training data (addresses the 1/47 data bottleneck)

## Local Usage

Run these commands from the repository root after installing the dependencies:

```powershell
python working_space/compute_stats.py
python working_space/train.py
python working_space/infer.py
```

### 1. Compute normalization statistics

```powershell
python working_space/compute_stats.py
python working_space/compute_stats.py --family flatfault_a
```

The statistics command saves a JSON file under `output/stats/`. Training loads
it automatically for the selected families, so `Cfg.vel_mean` and `Cfg.vel_std`
are only fallback values. If the file is missing, training prints a warning and
falls back to the configured values.

### 2. Train

```powershell
python working_space/train.py
python working_space/train.py --family flatfault_a
python working_space/train.py --family all --epochs 30 --batch_size 8
```

The `--family` option performs case-insensitive substring matching against the
configured family names:

```powershell
python working_space/train.py --family fault                # all fault families
python working_space/train.py --family flatfault_a,curvevel_a
python working_space/train.py --family all                  # every family
```

When a family is selected and no `--out_dir` is provided, results are written
to a matching subdirectory under `output` to avoid overwriting another run.

### 3. Run inference

```powershell
python working_space/infer.py \
    --ckpt output/model_YYMMDD_HHMM/best_unet.pth
```

### Recommended memory-safe training settings

On Kaggle with limited host memory, prefer:

```powershell
python working_space/train.py \
    --family all \
    --batch_size 4 \
    --num_workers 0 \
    --parallel_mode single
```

- `--num_workers 0` removes DataLoader worker host-memory overhead.
- `--parallel_mode single` uses one GPU and is stable on host RSS. The
  `data_parallel` mode uses multiple GPUs but can grow host RSS on multi-GPU T4
  via scatter/gather buffers, so it should be verified before a long run.
- `--log_memory` prints host and CUDA memory after each epoch so leaks can be
  detected early.

### Quick smoke run

```powershell
python working_space/train.py --test_run
```

`--test_run` overrides only three settings for a fast validation run: it
selects the flat families (`FlatVel_*`, `FlatFault_*`), trains 3 epochs, and
turns memory monitoring on. All other arguments (e.g. `--batch_size`,
`--num_workers`, `--parallel_mode`) follow the user-supplied values or `Cfg`
defaults. Run artifacts are written to a `test_YYMMDD_HHMM` directory instead
of `model_YYMMDD_HHMM`. Per-epoch host RSS and CUDA usage are printed, and the
per-epoch memory series is stored in `results.json` under `memory_monitoring`.
A stable `host_rss` delta across epochs indicates no host memory leak. Formal
training keeps memory monitoring off by default (`--log_memory` is only enabled
when requested).

## Training Run Artifacts

Each training command creates a timestamped run directory. The name is
`model_YYMMDD_HHMM`; a numeric suffix is added if the same minute is used more
than once. A run contains:

```text
output/model_YYMMDD_HHMM/
├── best_unet.pth       # Weights from the best validation epoch
├── config.json         # Effective arguments and shared configuration
├── history.json        # Per-epoch normalized and raw-unit MAE
├── mae_curve.png       # Normalized and raw-unit MAE curves
├── results.json        # Dataset, model, timing, and best-result summary
└── velocity_stats.json # mean/std actually used for normalization
```

The output directory can be changed with `--out_dir` or
`WAVEFORM_OUTPUT_ROOT`. Pass the resulting checkpoint explicitly to inference.

## Parallel Modes

The training entry point accepts `single`, `data_parallel`, and `ddp`:

- `single` (default): one device; stable on host memory.
- `data_parallel` / `ddp`: run through PyTorch `DistributedDataParallel` when
  `--nproc_per_node` is greater than one; otherwise they degrade to `single`.
  `data_parallel` is kept as an alias for `ddp` for CLI compatibility.

To launch a multi-GPU DDP run:

```powershell
python working_space/train.py \
    --family all \
    --parallel_mode data_parallel \
    --nproc_per_node 2
```

DDP spawns one process per GPU. Only the main process (rank 0) writes run
artifacts; the others participate in training and exit. This avoids the host
memory growth seen with `nn.DataParallel`, since gradient synchronization goes
through NCCL instead of host-side scatter/gather. `single` mode is never
changed and remains the safe default for local runs.

## Kaggle Usage

When the competition data is mounted under `/kaggle/input` or
`/kaggle/competition`, the configuration attempts to detect it automatically.
For a non-standard mount, set `WAVEFORM_DATA_ROOT` or pass Kaggle paths
explicitly:

```python
!python working_space/compute_stats.py \
    --data_dir /kaggle/input/waveform-inversion/train_samples

!python working_space/train.py \
    --data_dir /kaggle/input/waveform-inversion/train_samples \
    --out_dir /kaggle/working

!python working_space/infer.py \
    --ckpt /kaggle/working/best_unet.pth \
    --test_dir /kaggle/input/waveform-inversion/test \
    --out /kaggle/working/submission.csv
```

The training and validation loops display progress through `tqdm.auto`,
which works in both notebooks and terminals. Set `TQDM_DISABLE=1` to suppress
progress bars in long subprocess runs.

## Tests

The U-Net smoke test checks the expected output shape and a complete backward
pass:

```powershell
python working_space/test_unet.py
```

The data smoke test checks file pairing, normalization-statistic handling,
and the submission schema:

```powershell
python working_space/smoke_test.py
```

## Submission Format

For each test object, the inference script writes 70 rows named
`<oid>_y_<row>`. Each row contains the 35 odd x-columns:
`x_1, x_3, ..., x_69`.

## Notes

- The training loss is L1 loss on normalized velocity maps. Training logs also
  report `val_mae_raw` in the original velocity units.
- The seismic input is transformed with `sign(x)·log(1+|x|)` (keeps waveform
  polarity; adopted after the 14th-place A/B in `unet/improve_ruby.ipynb` —
  -21 m/s vs `log1p(abs(x))` on file-level 2-Fold).
- Training uses a file-level train/validation split to reduce leakage between
  samples from the same file.
- **Data volume caveat**: the local `train_samples/` is a partial download
  (~1/47 of the competition set), so experiments run on ~10k samples.
- Reduce `batch_size` or `num_workers` if memory is limited.
