"""Validate data pairing and submission formatting without running training."""

import csv
import glob
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg
from data import find_pairs


FAMILIES = {
    "FlatVel_A": ("data", "model", "data{}.npy", "model{}.npy"),
    "CurveVel_B": ("data", "model", "data{}.npy", "model{}.npy"),
    "FlatFault_A": (None, None, "seis_5_1_{}.npy", "vel_5_1_{}.npy"),
}
SAMPLES_PER_FILE = 16  # Keep files small so the test uses minimal disk space.


def make_fake_data(tmp):
    """Create a small synthetic dataset mirroring the real directory layout."""
    for family, (data_dir, model_dir, seismic_pattern, velocity_pattern) in FAMILIES.items():
        family_dir = os.path.join(tmp, family)
        if data_dir and model_dir:
            os.makedirs(os.path.join(family_dir, data_dir), exist_ok=True)
            os.makedirs(os.path.join(family_dir, model_dir), exist_ok=True)
        else:
            os.makedirs(family_dir, exist_ok=True)

        for i in range(2):
            seismic = np.random.randn(SAMPLES_PER_FILE, 5, 1000, 70).astype(np.float32) * 100
            velocity = np.random.rand(SAMPLES_PER_FILE, 70, 70).astype(np.float32) * 2000 + 2500
            if data_dir:
                seismic_path = os.path.join(family_dir, data_dir, seismic_pattern.format(i + 1))
                velocity_path = os.path.join(family_dir, model_dir, velocity_pattern.format(i + 1))
            else:
                seismic_path = os.path.join(family_dir, seismic_pattern.format(i))
                velocity_path = os.path.join(family_dir, velocity_pattern.format(i))
            np.save(seismic_path, seismic)
            np.save(velocity_path, velocity)

    test_dir = os.path.join(tmp, "test")
    os.makedirs(test_dir)
    for i in range(2):
        np.save(os.path.join(test_dir, f"oid{i:03d}.npy"),
                np.random.randn(5, 1000, 70).astype(np.float32))
    print(f"[smoke] fake data at: {tmp}")


def write_submission(tmp, test_files, vel_mean, vel_std):
    """Write a submission CSV following the competition schema."""
    oids = [os.path.splitext(os.path.basename(path))[0] for path in test_files]
    preds = np.stack(
        [np.random.randn(70, 70).astype(np.float32) * vel_std + vel_mean
         for _ in test_files],
        axis=0,
    )
    odd_cols = preds[:, :, Cfg.submission_x_start:Cfg.submission_x_stop:Cfg.submission_x_step]

    sub_path = os.path.join(tmp, "submission.csv")
    header = "oid_ypos," + ",".join(f"x_{i}" for i in range(1, 70, 2))
    with open(sub_path, "w") as file:
        file.write(header + "\n")
        for i, oid in enumerate(oids):
            for y in range(odd_cols.shape[1]):
                row = ",".join(f"{value:.1f}" for value in odd_cols[i, y])
                file.write(f"{oid}_y_{y},{row}\n")
    return sub_path


def main():
    """Run the full smoke test and clean up after itself."""
    tmp = tempfile.mkdtemp(prefix="wi_smoke_")
    try:
        make_fake_data(tmp)

        pairs = find_pairs(tmp, list(FAMILIES.keys()))
        print(f"[smoke] paired files: {len(pairs)}  (expected 6)")
        assert len(pairs) == 6, "Unexpected number of file pairs"

        all_vel = np.concatenate([np.load(velocity_path).ravel() for _, velocity_path in pairs])
        vel_mean, vel_std = float(all_vel.mean()), float(all_vel.std())
        print(f"[smoke] vel_mean={vel_mean:.1f}  vel_std={vel_std:.1f}")

        test_files = sorted(glob.glob(os.path.join(tmp, "test", "*.npy")))
        sub_path = write_submission(tmp, test_files, vel_mean, vel_std)

        with open(sub_path) as file:
            lines = file.readlines()
        print(f"[smoke] submission lines: {len(lines)} (expected 1+2*70=141)")
        assert len(lines) == 141, "Unexpected number of submission rows"
        assert lines[0].startswith("oid_ypos,x_1,x_3"), "Invalid submission header"

        row = next(csv.reader([lines[1]]))
        assert len(row) == 36, f"Expected 36 fields per row, got {len(row)}"
        print("[smoke] first data line (truncated):", ",".join(row[:4]), "...,", row[-1])
        print("Smoke test passed: data loading and submission format are valid")
    finally:
        shutil.rmtree(tmp)
        print(f"[smoke] cleaned up {tmp}")


if __name__ == "__main__":
    main()
