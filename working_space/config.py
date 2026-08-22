"""Central configuration for training, inference, and data preparation."""

from pathlib import Path


class Cfg:
    # Project-relative paths. Override these with CLI arguments for Kaggle paths.
    project_root = Path(__file__).resolve().parents[1]
    input_root = project_root / "input" / "waveform-inversion"
    train_data_dir = input_root / "train_samples"
    test_data_dir = input_root / "test"
    output_dir = project_root / "output"
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

    # Target normalization; replace with values from compute_stats.py
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
    # Shape smoke test
    test_batch_size = 2
    test_base_channels = 4
