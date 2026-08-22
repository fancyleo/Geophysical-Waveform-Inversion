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
│   ├── config.py         # Shared paths and hyperparameters
│   ├── train.py          # Dataset, U-Net, training, and validation
│   ├── infer.py          # Test inference and submission generation
│   ├── compute_stats.py  # Velocity normalization statistics
│   ├── test_unet.py      # U-Net forward/backward smoke test
│   └── smoke_test.py     # Data pairing and submission format test
└── output/               # Training run artifacts
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
Seismic input:  (N, 5, 1000, 70)
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

The statistics command reports and saves the velocity mean and standard
deviation. Training automatically loads the JSON statistics file for the
selected family, so `Cfg.vel_mean` and `Cfg.vel_std` are only fallback values.

To override selected settings:

```powershell
python working_space/train.py --epochs 30 --batch_size 8
python working_space/infer.py --batch_size 4
python working_space/train.py --family flatfault_a
python working_space/train.py --family fault
python working_space/train.py --family flatfault_a,curvevel_a
```

The `--family` option performs case-insensitive substring matching against the
configured family names. Omitting it or using `all` trains on every family.
When a family is selected and no `--out_dir` is provided, results are written
to a matching subdirectory under `output` to avoid overwriting another run.

Each training command creates a timestamped run directory under the selected
output root. The directory name is `model_YYMMDD_HHMM`; a numeric suffix is
added if the same minute is used more than once. A run contains:

```text
output/model_YYMMDD_HHMM/
├── best_unet.pth       # Weights from the best validation epoch
├── config.json         # Effective arguments and shared configuration
├── history.json        # Per-epoch normalized and raw-unit MAE
├── mae_curve.png       # Normalized and raw-unit MAE curves
└── results.json        # Dataset, model, timing, and best-result summary
```

The output directory can be changed with `--out_dir` or
`WAVEFORM_OUTPUT_ROOT`. Pass the resulting checkpoint explicitly to inference:

```powershell
python working_space/infer.py \
    --ckpt output/model_YYMMDD_HHMM/best_unet.pth
```

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
which works in both notebooks and terminals.

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
