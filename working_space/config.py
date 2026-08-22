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


def select_families(query=None):
    """Select configured data families using a case-insensitive keyword."""
    if query is None or not query.strip() or query.strip().lower() == "all":
        return list(Cfg.families)

    requested = [item.strip().lower() for item in query.split(",") if item.strip()]
    selected = [
        family
        for family in Cfg.families
        if any(keyword in family.lower() for keyword in requested)
    ]
    if not selected:
        available = ", ".join(Cfg.families)
        raise ValueError(
            f"No family matched '{query}'. Available families: {available}"
        )
    return selected


def family_slug(families):
    """Create a stable filesystem-safe name for a family selection."""
    return "_".join(family.lower() for family in families)


def stats_path_for_families(families, output_root=None):
    """Return the shared JSON path used for a family selection's statistics."""
    root = Cfg.output_dir if output_root is None else Path(output_root)
    return root / "stats" / f"velocity_stats_{family_slug(families)}.json"


def load_velocity_stats(path):
    """Load and validate mean/std values from a computed statistics JSON file."""
    import json

    with open(path) as file:
        stats = json.load(file)
    mean = float(stats["mean"])
    std = float(stats["std"])
    if std <= 0:
        raise ValueError(f"Velocity standard deviation must be positive: {path}")
    return mean, std


class Cfg:
    """Shared paths, dataset settings, model hyperparameters, and test options."""

    # Resolve paths from environment variables, Kaggle mounts, or the project root.
    project_root = Path(__file__).resolve().parents[1]
    input_root = resolve_data_root()
    train_data_dir = input_root / "train_samples"
    test_data_dir = input_root / "test"
    output_dir = resolve_output_root(project_root)
    stats_dir = output_dir / "stats"
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
