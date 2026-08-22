"""
Yale/UNC-CH - Geophysical Waveform Inversion
推理 + 生成 submission.csv

用法（本地默认路径）:
    python infer.py

Kaggle 使用时通过 --ckpt、--test_dir 和 --out 覆盖默认路径。

输出格式（与 sample_submission.csv 一致）:
  oid_ypos,x_1,x_3,...,x_69
  000039dca2_y_0,3000.0,3000.0,...,3000.0
  ...
"""

import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from working_space.train import UNet, Cfg   # 复用模型定义和配置

# ─────────────────────────────────────────────
# 1. 测试数据集
# ─────────────────────────────────────────────
class TestDataset(Dataset):
    """
    每个 .npy 文件是一个完整 sample，shape (5, 1000, 70)
    文件名为 oid（不含扩展名）
    """
    def __init__(self, test_dir):
        self.files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
        if len(self.files) == 0:
            raise RuntimeError(f"未在 {test_dir} 找到任何 .npy 文件")
        # 提取 oid
        self.oids = [os.path.splitext(os.path.basename(f))[0] for f in self.files]
        print(f"[info] test files: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        seis = np.load(self.files[i]).astype(np.float32)   # (5, 1000, 70)
        seis = np.log1p(np.abs(seis))                     # 同训练预处理
        return self.oids[i], torch.from_numpy(seis)


# ─────────────────────────────────────────────
# 2. 主推理
# ─────────────────────────────────────────────
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default=str(Cfg.checkpoint_path))
    parser.add_argument("--test_dir", default=str(Cfg.test_data_dir))
    parser.add_argument("--out",      default=str(Cfg.submission_path))
    parser.add_argument("--batch_size", type=int, default=Cfg.infer_batch_size)
    parser.add_argument("--vel_mean", type=float, default=Cfg.vel_mean)
    parser.add_argument("--vel_std",  type=float, default=Cfg.vel_std)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device_name = Cfg.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # 2.1 加载模型
    model = UNet(in_ch=Cfg.n_src, base=Cfg.model_base_channels).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[info] loaded checkpoint: {args.ckpt}")

    # 2.2 数据
    ds = TestDataset(args.test_dir)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=Cfg.num_workers
    )

    # 2.3 推理
    oid_list = []
    preds = []   # 存 (B, 70, 70) 反归一化后的速度图
    for oids, seis in tqdm(loader, desc="inference"):
        seis = seis.to(device)                     # (B,5,1000,70)
        pred = model(seis)                         # (B,70,70)
        pred = pred.cpu().numpy() * args.vel_std + args.vel_mean
        preds.append(pred)
        oid_list.extend(oids)

    preds = np.concatenate(preds, axis=0)          # (N, 70, 70)
    print(f"[info] predictions shape: {preds.shape}")

    # 2.4 写 submission
    # 只取奇数列: x_1, x_3, ..., x_69  → 索引 1,3,...,69
    odd_cols = preds[:, :, Cfg.submission_x_start:Cfg.submission_x_stop:Cfg.submission_x_step]
    out_path = args.out
    with open(out_path, "w") as f:
        f.write("oid_ypos,x_1,x_3,x_5,x_7,x_9,x_11,x_13,x_15,x_17,x_19,"
                "x_21,x_23,x_25,x_27,x_29,x_31,x_33,x_35,x_37,x_39,"
                "x_41,x_43,x_45,x_47,x_49,x_51,x_53,x_55,x_57,x_59,"
                "x_61,x_63,x_65,x_67,x_69\n")
        for i, oid in enumerate(oid_list):
            for y in range(odd_cols.shape[1]):     # 70 行
                row_vals = ",".join(f"{v:.1f}" for v in odd_cols[i, y])
                f.write(f"{oid}_y_{y},{row_vals}\n")
    print(f"[done] saved → {out_path}")


if __name__ == "__main__":
    main()
