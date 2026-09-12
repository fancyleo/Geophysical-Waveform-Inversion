# Windows Inference Guide — generate `submission.csv`

How to run `working_space/infer.py` on a Windows machine to produce a Kaggle
submission from the trained checkpoints.

`infer.py` is self-contained: it only needs `config.py`, `model.py`, the
`.pth` checkpoints, and the test `.npy` files. **No training data is required.**

---

## 1. What you need

| Item | Notes |
|---|---|
| Repo | this repository (`working_space/infer.py`, `config.py`, `model.py`) |
| Checkpoints | `best_ema.pth` and/or `best_unet.pth` per run, **plus `velocity_stats.json` in the same folder** |
| Test data | a folder containing flat `*.npy` files, one per test object |
| Python | **3.12** (`.python-version`), PyTorch **2.5.1** |
| GPU | optional but strongly recommended (CUDA 12.1 build) |

Extract the shipped bundle to get all six weights:

```
output/exports/stage_a_seed777_3models_20260912.zip
  └── model_260909_0834/  best_ema.pth  best_unet.pth  velocity_stats.json  ...
      model_260910_1213/  ...
      model_260911_2049/  ...
```

| Recipe | Holdout MAE (m/s) |
|---|---|
| `model_260910_1213/best_ema.pth` alone | 121.10 |
| 3 × `best_ema` (equal weight) | 112.73 |
| **6 checkpoints (3 × `best_ema` + 3 × `best_unet`)** | **111.49** ← best |
| 3 × `best_ema` is the minimum acceptable; 1 checkpoint is the weakest | |

Always ship the submission built from the **6-checkpoint ensemble**.

---

## 2. Test data layout

```
D:\kaggle\waveform-inversion\test\
    <oid1>.npy      # shape (5, 1000, 70), float32, raw amplitudes
    <oid2>.npy
    ...
```

- Point `--test_dir` at the folder that **directly contains** the `.npy` files.
- The file stem is used as the object id (`<oid>_y_0` … `<oid>_y_69` in the CSV).
- Preprocessing (`sign(x) * log1p(|x|)`) is applied inside `infer.py` — store the
  **raw** arrays, do not preprocess them yourself.

---

## 3. Environment setup (PowerShell)

```powershell
cd C:\path\to\Geophysical-Waveform-Inversion

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# GPU build (CUDA 12.1) — needs a driver >= 530
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm scikit-learn

# --- OR CPU-only build (no GPU) ---
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
# pip install numpy tqdm scikit-learn
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.5.1 True` (with a GPU) or `2.5.1 False` (CPU build).

Alternatives: `uv sync` also works (install `uv` first, `pip install uv`), and
`pip install -r requirements.txt` installs the CPU/CUDA-default torch wheel.

---

## 4. Run inference

```powershell
cd working_space

python infer.py `
  --ckpt "D:\models\model_260910_1213\best_ema.pth" `
         "D:\models\model_260911_2049\best_ema.pth" `
         "D:\models\model_260909_0834\best_ema.pth" `
         "D:\models\model_260910_1213\best_unet.pth" `
         "D:\models\model_260911_2049\best_unet.pth" `
         "D:\models\model_260909_0834\best_unet.pth" `
  --test_dir "D:\kaggle\waveform-inversion\test" `
  --out submission.csv `
  --batch_size 16 `
  --num_workers 0
```

- The ensemble is an **equal-weight average**, so the order of `--ckpt` does not matter.
- Single-model variant: pass just one `--ckpt`.

### Arguments

| Flag | Default | Purpose |
|---|---|---|
| `--ckpt` | `output/best_unet.pth` | one or more checkpoints; ≥2 = equal-weight ensemble |
| `--test_dir` | `Cfg.test_data_dir` | folder with the test `*.npy` files |
| `--out` | `output/submission.csv` | output CSV path |
| `--batch_size` | 4 | raise to 16–32 on a GPU, lower to 2 if you hit OOM |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--num_workers` | 2 | **use `0` on Windows** if you hit DataLoader errors |
| `--stats_path` | checkpoint dir | velocity-normalization JSON (see §6) |

### Expected console output

```
[info] loaded velocity statistics: D:\models\model_260910_1213\velocity_stats.json
[info] device: cuda
[info] loaded checkpoint: D:\models\model_260910_1213\best_ema.pth
...
[info] ensemble size: 6 (equal-weight average)
[info] test files: <N>
inference: 100%|####################| <N>/<N>
[info] predictions shape: (N, 70, 70)
[done] saved → submission.csv
```

GPU inference is fast — for a few hundred test files expect seconds to a couple
of minutes, dominated by reading the `.npy` files. The CPU path works too but is
far slower (roughly one to two orders of magnitude).

---

## 5. Validate the output

```powershell
python -c "import csv; rows=list(csv.reader(open('submission.csv',newline=''))); print('data rows', len(rows)-1); print('header fields', len(rows[0])); print('all rows 36 fields', all(len(r)==36 for r in rows[1:])); print('unique oids', len({r[0].rsplit('_y_',1)[0] for r in rows[1:]}))"
```

Expected:

- `data rows` = `70 × N` (N = number of test files)
- `header fields` = 36 (`oid_ypos,x_1,x_3,...,x_69`)
- `all rows 36 fields` = `True`
- `unique oids` = `N`, each with exactly 70 rows

If the competition `sample_submission.csv` is available, also check that the key
set matches exactly:

```powershell
python -c "a=[l.split(',')[0] for l in open('submission.csv').read().splitlines()[1:]]; b=[l.split(',')[0] for l in open(r'D:\kaggle\waveform-inversion\sample_submission.csv').read().splitlines()[1:]]; print(len(a), len(b), set(a)==set(b), a==b)"
```

`set(a)==set(b)` must be `True`. If `a==b` is also `True`, the row order matches
the sample file, which is the safest thing to submit.

---

## 6. Velocity statistics — do not skip this

`infer.py` denormalizes with `pred * vel_std + vel_mean`, loaded from
`velocity_stats.json` **in the directory of the first `--ckpt`**:

```
mean = 2905.50
std  = 792.22
```

- If that file is missing you will see
  `[warn] no statistics JSON at ...; using --vel_mean/--vel_std`, and the
  predictions will be wrong (the built-in defaults are stale, from the old
  6-family dataset).
- Fix: keep `velocity_stats.json` next to the `.pth` files (as shipped), or pass
  `--stats_path D:\models\model_260910_1213\velocity_stats.json`.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `[warn] no statistics JSON ...` | stats file missing → see §6 |
| `RuntimeError: No .npy files found in ...` | `--test_dir` points at the wrong level; it must be the folder holding the `.npy` files |
| `CUDA out of memory` | lower `--batch_size` (e.g. `2`); each extra checkpoint costs VRAM |
| DataLoader / worker crash on Windows | `--num_workers 0` |
| Predictions look like noise when submitted | wrong velocity stats (§6) or the checkpoints are not the Stage A runs |
| Errors reading paths with spaces / OneDrive | quote paths; prefer a short ASCII path such as `D:\models` |
| `torch.load` raises `weights_only` / unknown-format errors | requires `torch >= 2.5`; use the pinned 2.5.1 wheel |
| Want to rerun only the last stage | the script is a single deterministic pass — simply rerun; there is no cache |

---

## 8. Reproducing the reported numbers

`infer.py` writes a submission; it does not score anything (the test labels are
hidden). Reproduce the MAE numbers on the shared holdout on a machine that has
`train_samples` available:

```bash
# Linux / training machine, from working_space
WAVEFORM_OUTPUT_ROOT=/kaggle/working/Geophysical-Waveform-Inversion/output \
python eval_holdout.py \
  --ckpts output/all/model_260910_1213/best_ema.pth \
          output/all/model_260911_2049/best_unet.pth \
          ...
```

That yields 121.10 (best single), 112.73 (3 × `best_ema`) and 111.49 (all six
checkpoints). The holdout is the file-level 10% split with seed 777, so these
numbers are comparable across runs — see
`working_space/unet/experiment_plan_20260906.md` §8.
