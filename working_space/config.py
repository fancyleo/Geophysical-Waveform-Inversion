"""Central configuration for training, inference, and data preparation."""

import os
from pathlib import Path


def _is_waveform_root(path):
    """Return whether a directory contains the expected waveform layout."""
    return (
        path.is_dir()
        and (path / "train_samples").is_dir()
        and (path / "test").is_dir()
    )


def resolve_data_root():
    """Resolve the waveform dataset root across local and Kaggle environments."""
    project_root = Path(__file__).resolve().parents[1]
    configured_root = os.getenv("WAVEFORM_DATA_ROOT")
    candidates = []

    if configured_root:
        candidates.append(Path(configured_root).expanduser())

    candidates.append(project_root / "input" / "waveform-inversion")

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(path for path in kaggle_input.iterdir() if path.is_dir())

    kaggle_competition = Path("/kaggle/competition")
    if kaggle_competition.is_dir():
        candidates.extend(
            path for path in kaggle_competition.iterdir() if path.is_dir()
        )

    for candidate in candidates:
        if _is_waveform_root(candidate):
            return candidate.resolve()

    # Keep configuration imports usable for model-only tests without datasets.
    return project_root / "input" / "waveform-inversion"


def resolve_output_root(project_root):
    """Resolve a writable output directory for local and Kaggle execution."""
    configured_root = os.getenv("WAVEFORM_OUTPUT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working
    return project_root / "output"


class Cfg:
    """Shared paths, dataset settings, model hyperparameters, and test options."""

    # Resolve paths from environment variables, Kaggle mounts, or the project root.
    project_root = Path(__file__).resolve().parents[1]
    input_root = resolve_data_root()
    train_data_dir = input_root / "train_samples"
    test_data_dir = input_root / "test"
    output_dir = resolve_output_root(project_root)
    checkpoint_path = output_dir / "best_unet.pth"
    history_path = output_dir / "history.json"
    submission_path = output_dir / "submission.csv"
    sample_submission_path = input_root / "sample_submission.csv"

    # Reproducibility and runtime
    seed = 42
    device = "auto"                 # auto, cpu, or cuda
    num_workers = 2

    # Dataset dimensions
    img_size = 70
    n_src = 5
    n_steps = 1000
    n_recv = 70

    # Training
    batch_size = 8
    epochs = 30
    lr = 1e-3
    val_ratio = 0.1
    model_base_channels = 32

    # Target normalization; replace these values with compute_stats.py output.
    vel_mean = 3500.0
    vel_std = 500.0

    # Training data families
    families = [
        "FlatVel_A", "FlatVel_B",
        "CurveVel_A", "CurveVel_B",
        "FlatFault_A", "FlatFault_B",
        "CurveFault_A", "CurveFault_B",
        "Style_A", "Style_B",
    ]

    # Inference and submission
    infer_batch_size = 4
    submission_x_start = 1
    submission_x_stop = 70
    submission_x_step = 2
    # Shape smoke-test settings
    test_batch_size = 2
    test_base_channels = 4
