"""
compute_stats.py
统计训练集速度图的均值/标准差，用于归一化。
跑完后把结果填回 config.py 的 Cfg.vel_mean / Cfg.vel_std。

用法:
    python compute_stats.py
    python compute_stats.py --data_dir /kaggle/input/waveform-inversion/train_samples
"""

import os, glob, argparse
import numpy as np
from tqdm.auto import tqdm
from config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(Cfg.train_data_dir))
    args = parser.parse_args()

    files = []
    for root, _, fs in os.walk(args.data_dir):
        for f in fs:
            if f.endswith(".npy") and ("model" in f or f.startswith("vel_")):
                files.append(os.path.join(root, f))
    files = sorted(set(files))
    print(f"[info] found {len(files)} velocity .npy files")

    vals = []
    for fp in tqdm(files, desc="velocity statistics"):
        arr = np.load(fp)                # (500,70,70) or (500,1,70,70)
        arr = arr.reshape(arr.shape[0], -1)
        vals.append(arr.ravel())
    vals = np.concatenate(vals)
    print(f"[stats] count   = {vals.size}")
    print(f"[stats] mean    = {vals.mean():.2f}")
    print(f"[stats] std     = {vals.std():.2f}")
    print(f"[stats] min     = {vals.min():.2f}")
    print(f"[stats] max     = {vals.max():.2f}")
    print(f"[stats] median  = {np.median(vals):.2f}")
    print("\n→ 把上面 mean / std 填到 config.py 的 Cfg.vel_mean / Cfg.vel_std")

if __name__ == "__main__":
    main()
