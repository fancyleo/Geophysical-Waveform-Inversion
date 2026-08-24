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
│   ├── model.py        # U-Net architecture
│   ├── training.py     # Train/validate loops and parallelization helpers
│   ├── utils.py        # Memory logging and artifact persistence
│   ├── train.py        # Training entry point
│   ├── infer.py        # Test inference and submission generation
│   ├── compute_stats.py# Velocity normalization statistics
│   ├── test_unet.py    # U-Net shape/gradient smoke test
│   └── smoke_test.py   # Data pairing and submission format test
└── output/             # Training run artifacts
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
- `data_parallel`: uses `nn.DataParallel` across available CUDA GPUs. When only
  one GPU is available it degrades to `single`. Checkpoints are saved without
  the `module.` prefix, so they load fine with a single-GPU model.
- `ddp`: reserved as a future distributed-training backend; it currently
  reports an explicit error instead of running a different configuration.

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
- The seismic input is transformed with `log1p(abs(x))`.
- Training uses a file-level train/validation split to reduce leakage between
  samples from the same file.
- Reduce `batch_size` or `num_workers` if memory is limited.
