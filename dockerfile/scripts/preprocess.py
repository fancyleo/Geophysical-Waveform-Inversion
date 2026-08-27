"""
CPU-side preprocessing script (preprocess.py)
Role: preprocess the raw data inside the CPU container into a training-friendly
      format, writing to /data/kaggle/working for the GPU container to use.

Typical steps:
1. Build a data checksum manifest (manifest.csv)
2. Convert to Parquet shards (faster GPU reads)
3. Generate train.csv / val.csv indices

Usage (CPU container):
  docker compose --profile downloader run --rm downloader \
      python /scripts/preprocess.py
"""
import os
import glob
import hashlib
import pandas as pd
from pathlib import Path

INPUT_DIR = os.environ.get("DOWNLOAD_DIR", "/data/kaggle/input")
WORKING_DIR = os.environ.get("WORKING_DIR", "/data/kaggle/working")

os.makedirs(WORKING_DIR, exist_ok=True)


def file_md5(filepath, chunk_size=1024 * 1024):
    """Compute the file MD5 (streaming, suitable for large files)."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest():
    """Scan the input directory and build manifest.csv (path + size + md5)."""
    print(f">>> Building manifest from {INPUT_DIR}")
    rows = []
    for filepath in glob.glob(os.path.join(INPUT_DIR, "**", "*"), recursive=True):
        if os.path.isfile(filepath):
            rel = os.path.relpath(filepath, INPUT_DIR)
            size = os.path.getsize(filepath)
            md5 = file_md5(filepath)
            rows.append({"path": rel, "size": size, "md5": md5})

    manifest = pd.DataFrame(rows)
    out_path = os.path.join(WORKING_DIR, "manifest.csv")
    manifest.to_csv(out_path, index=False)
    print(f"✓ manifest.csv: {len(manifest)} files -> {out_path}")
    return manifest


def convert_to_parquet(manifest: pd.DataFrame, max_rows_per_file: int = 200_000):
    """
    Example: convert CSV shards to Parquet (faster reads and lower memory on GPU).
    Actual columns need to be adapted to your competition data.
    """
    print(">>> Converting CSV -> Parquet (example)")
    csv_files = manifest[manifest["path"].str.endswith(".csv")]

    parquet_dir = os.path.join(WORKING_DIR, "parquet")
    os.makedirs(parquet_dir, exist_ok=True)

    for _, row in csv_files.iterrows():
        csv_path = os.path.join(INPUT_DIR, row["path"])
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  skip {row['path']}: {e}")
            continue

        # write shards to parquet
        base = os.path.splitext(os.path.basename(row["path"]))[0]
        for i, start in enumerate(range(0, len(df), max_rows_per_file)):
            chunk = df.iloc[start:start + max_rows_per_file]
            out = os.path.join(parquet_dir, f"{base}_{i:03d}.parquet")
            chunk.to_parquet(out, index=False)
        print(f"  {row['path']}: {len(df)} rows -> parquet/")


def make_split(manifest: pd.DataFrame, val_ratio: float = 0.1):
    """Generate train / val split indices."""
    print(">>> Making train/val split")
    df = manifest.copy()
    # Simple file-level split; adjust to your actual business logic
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = int(len(df) * val_ratio)
    df["split"] = ["val"] * n_val + ["train"] * (len(df) - n_val)

    train_csv = df[df["split"] == "train"]
    val_csv = df[df["split"] == "val"]
    train_csv.to_csv(os.path.join(WORKING_DIR, "train.csv"), index=False)
    val_csv.to_csv(os.path.join(WORKING_DIR, "val.csv"), index=False)
    print(f"✓ train.csv: {len(train_csv)}, val.csv: {len(val_csv)}")


if __name__ == "__main__":
    print("=" * 50)
    print("CPU Preprocessing Pipeline")
    print(f"INPUT : {INPUT_DIR}")
    print(f"OUTPUT: {WORKING_DIR}")
    print("=" * 50)

    manifest = build_manifest()
    convert_to_parquet(manifest)
    make_split(manifest)

    print("\n===== Preprocessing Complete =====")
    print(f"Working dir contents:")
    for f in sorted(os.listdir(WORKING_DIR)):
        print(f"  - {f}")
