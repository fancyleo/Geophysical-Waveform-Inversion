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
└── output/               # Default checkpoints and submissions
```

## Data Format

The training data is expected to contain paired seismic and velocity files:

```text
train_samples/
├── FlatVel_A/data/*.npy
├── FlatVel_A/model/*.npy
├── FlatFault_A/seis_*.npy
├── FlatFault_A/vel_*.npy
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

Command-line arguments override the relevant defaults, which makes the same
scripts usable with Kaggle paths.

## Local Usage

Run these commands from the repository root after installing the dependencies:

```powershell
python working_space/compute_stats.py
python working_space/train.py
python working_space/infer.py
```

The statistics command reports the velocity mean and standard deviation. Set
`Cfg.vel_mean` and `Cfg.vel_std` in `working_space/config.py` before training.

To override selected settings:

```powershell
python working_space/train.py --epochs 30 --batch_size 8
python working_space/infer.py --batch_size 4
```

The default outputs are:

```text
output/best_unet.pth
output/history.json
output/submission.csv
```

## Kaggle Usage

When the competition data is mounted under `/kaggle/input`, pass Kaggle paths
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

- The training loss is L1 loss on normalized velocity maps.
- The seismic input is transformed with `log1p(abs(x))`.
- Training uses a file-level train/validation split to reduce leakage between
  samples from the same file.
- Reduce `batch_size` or `num_workers` if memory is limited.
