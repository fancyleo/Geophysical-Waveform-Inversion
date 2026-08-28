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

    # Kaggle mounts the dataset under a nested layout, e.g.
    #   /kaggle/input/waveform-inversion
    #   /kaggle/input/competitions/waveform-inversion
    #   /kaggle/competition/waveform-inversion
    for base in (Path("/kaggle/input"), Path("/kaggle/competition")):
        if not base.is_dir():
            continue
        candidates.extend(path for path in base.iterdir() if path.is_dir())
        competitions = base / "competitions"
        if competitions.is_dir():
            candidates.extend(path for path in competitions.iterdir() if path.is_dir())

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


def resolve_device(device=None):
    """Return a ``torch.device`` from 'auto', 'cpu', or 'cuda'."""
    import torch

    device = Cfg.device if device is None else device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _load_augmentations(path, default):
    """Load augmentations from an optional JSON override file; fall back to default.

    The override file is written by ``aug_config.write_winner_aug`` after the
    augmentation exploration (aug_explore.ipynb), so formal training
    (train.py / preflight.ipynb) can directly use the winning augmentation.
    """
    import json

    try:
        if path.is_file():
            with open(path) as file:
                data = json.load(file)
            if isinstance(data, dict) and isinstance(data.get("augmentations"), dict):
                return data["augmentations"]
    except Exception as exc:  # defensive: never break training on a bad override
        print(f"[warn] 读取增强覆盖文件失败({exc})，使用默认增强")
    return default


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
    # single is stable and leak-free on host memory; data_parallel uses all
    # GPUs but can grow host RSS on multi-GPU T4 via scatter/gather buffers.
    parallel_mode = "single"        # single, data_parallel, or ddp
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

    # Data augmentation (train only; applied to raw physical traces before
    # abs/log1p and velocity normalization). Each key maps to params passed to
    # the matching function in pretrain.py; "prob" is the apply probability.
    # Set prob to 0 or remove a key to disable that augmentation. To add a new
    # one, write it in pretrain.py with @register_aug("name") and enable here.
    # NOTE: 若 output/aug_explore/winner_aug.json 存在，将自动覆盖以下默认值
    #（由 aug_config.write_winner_aug 写回，供 train.py / preflight.ipynb 直接使用）。
    _aug_override_path = project_root / "output" / "aug_explore" / "winner_aug.json"
    augmentations = _load_augmentations(_aug_override_path, {
        "xflip": {"prob": 0.5},
        "time_shift": {"prob": 0.5, "max_shift": 100},
    })

    # Target normalization; values computed for all 10 families via compute_stats.py.
    # train.py prefers the stats JSON in output/stats/ when it exists.
    vel_mean = 2916.82
    vel_std = 817.36

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
