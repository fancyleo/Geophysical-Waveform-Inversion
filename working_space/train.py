"""
Yale/UNC-CH - Geophysical Waveform Inversion
UNet Baseline Training Script

数据假设结构（Kaggle input 或本地）：
  input_dir/
    FlatVel_A/data/*.npy  (500, 5, 1000, 70)
    FlatVel_A/model/*.npy (500, 70, 70)
    FlatFault_A/seis_*.npy
    FlatFault_A/vel_*.npy
    ...
    test/{oid}.npy        (5, 1000, 70) per file

提交格式：每 oid 一行 y_0..y_69，每行只有奇数列 x_1,x_3,...,x_69
"""

import os, glob, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from config import Cfg

if Cfg.device == "auto":
    Cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# 2. 数据加载工具
# ─────────────────────────────────────────────
def find_pairs(root):
    """
    扫描 root 下所有家族目录，返回 [(seis_path, vel_path), ...]
    Vel/Style 家族: dataN.npy + modelN.npy
    Fault 家族:    seis_*.npy + vel_*.npy
    """
    pairs = []
    for fam in Cfg.families:
        fam_dir = os.path.join(root, fam)
        if not os.path.isdir(fam_dir):
            print(f"[warn] missing family dir: {fam_dir}")
            continue

        # Vel / Style
        data_dir = os.path.join(fam_dir, "data")
        model_dir = os.path.join(fam_dir, "model")
        if os.path.isdir(data_dir) and os.path.isdir(model_dir):
            seis_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
            for sf in seis_files:
                base = os.path.basename(sf)
                # data1.npy -> model1.npy
                mf = os.path.join(model_dir, base.replace("data", "model"))
                if os.path.exists(mf):
                    pairs.append((sf, mf))
            continue

        # Fault
        seis_files = sorted(glob.glob(os.path.join(fam_dir, "seis_*.npy")))
        for sf in seis_files:
            base = os.path.basename(sf)
            vf = os.path.join(fam_dir, base.replace("seis_", "vel_"))
            if os.path.exists(vf):
                pairs.append((sf, vf))

    print(f"[info] total paired files: {len(pairs)}")
    return pairs


class SeisVelDataset(Dataset):
    """
    懒加载版：每个 item 只读一个 sample（从预加载的文件中按索引取）
    如果内存够，可以把所有数据一次性读进内存（见下方 InMemDataset）
    """
    def __init__(self, pairs, idx_list, vel_mean=Cfg.vel_mean, vel_std=Cfg.vel_std):
        """
        pairs:    [(seis_path, vel_path), ...]
        idx_list: 扁平索引列表，每个元素是 (file_idx, sample_idx)
        """
        self.pairs = pairs
        self.idx_list = idx_list
        self.vel_mean = vel_mean
        self.vel_std = vel_std
        # 缓存已打开的 memmap
        self._seis_cache = {}
        self._vel_cache = {}

    def __len__(self):
        return len(self.idx_list)

    def _open(self, fi):
        if fi not in self._seis_cache:
            sp, vp = self.pairs[fi]
            self._seis_cache[fi] = np.load(sp, mmap_mode="r")
            self._vel_cache[fi] = np.load(vp, mmap_mode="r")
        return self._seis_cache[fi], self._vel_cache[fi]

    def __getitem__(self, i):
        fi, si = self.idx_list[i]
        seis_arr, vel_arr = self._open(fi)

        seis = seis_arr[si].astype(np.float32)   # (5, 1000, 70)
        vel  = vel_arr[si].astype(np.float32)    # (70, 70) or (1, 70, 70)

        # 速度图归一化
        vel = (vel - self.vel_mean) / self.vel_std

        # 地震波形预处理：log(1+|x|) 压缩动态范围
        seis = np.log1p(np.abs(seis))

        # 转成 (C, H, W) 形式：把 5 个源 × 70 接收器当通道维
        # 形状变为 (5*70, 1000) -> 后面用 1D/2D Conv 处理
        # 更常用：把 (5, 1000, 70) 视为 5 张 (1000, 70) 的"图像"
        seis = seis.reshape(Cfg.n_src, Cfg.n_steps, Cfg.n_recv)

        return torch.from_numpy(seis), torch.from_numpy(vel)


def build_flat_indices(pairs):
    """返回 [(file_idx, sample_idx), ...] 共 len(pairs)*500 条"""
    indices = []
    for fi, (sp, _) in enumerate(pairs):
        arr = np.load(sp, mmap_mode="r")
        n = arr.shape[0]
        for si in range(n):
            indices.append((fi, si))
    return indices


# ─────────────────────────────────────────────
# 3. 模型：UNet（输入 5×1000×70 → 输出 1×70×70）
# ─────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    """
    输入: (B, 5, 1000, 70)  —— 5 个源当成通道
    输出: (B, 1, 70, 70)
    思路：时间维 1000 通过多次 stride-2 下采样压到 70 左右
    """
    def __init__(self, in_ch=Cfg.n_src, base=Cfg.model_base_channels):
        super().__init__()
        # 下采样
        self.enc1 = DoubleConv(in_ch, base)        # 1000x70 -> 1000x70
        self.pool1 = nn.MaxPool2d(2, 2)            # -> 500x35
        self.enc2 = DoubleConv(base, base*2)        # 500x35
        self.pool2 = nn.MaxPool2d(2, 2)            # -> 250x17  (pad 到 250x18)
        self.enc3 = DoubleConv(base*2, base*4)      # 250x18
        self.pool3 = nn.MaxPool2d(2, 2)            # -> 125x9
        self.enc4 = DoubleConv(base*4, base*8)      # 125x9
        self.pool4 = nn.MaxPool2d(2, 2)            # -> 62x4   (pad 到 62x5)
        self.enc5 = DoubleConv(base*8, base*16)     # 62x5

        # 中间上采样到 70x70
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base*16, base*8, kernel_size=(9, 14),
                                stride=(2, 2)),    # 62x5 -> 125x10 (approx)
            nn.BatchNorm2d(base*8),
            nn.ReLU(inplace=True),
        )

        # 解码器
        self.dec1 = DoubleConv(base*16, base*8)     # concat with enc4
        self.up2 = nn.ConvTranspose2d(base*8, base*4, kernel_size=4, stride=2, padding=1)
        self.dec2 = DoubleConv(base*8, base*4)
        self.up3 = nn.ConvTranspose2d(base*4, base*2, kernel_size=4, stride=2, padding=1)
        self.dec3 = DoubleConv(base*4, base*2)
        self.up4 = nn.ConvTranspose2d(base*2, base,   kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec4 = DoubleConv(base*2, base)
        self.dec5 = DoubleConv(base, base)

        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # x: (B, 5, 1000, 70)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        # pad 时间维到偶数
        e2p = nn.functional.pad(e2, (0, 1))         # 17->18
        e3 = self.enc3(self.pool2(e2p))
        e4 = self.enc4(self.pool3(e3))
        e4p = nn.functional.pad(e4, (0, 1))         # 4->5
        e5 = self.enc5(self.pool4(e4p))

        u = self.up(e5)                             # -> 125x10
        # 裁剪/对齐
        u = nn.functional.interpolate(u, size=(125, 9), mode="nearest")
        d1 = self.dec1(torch.cat([u, e4], dim=1))
        d2 = self.up2(d1)
        d2 = nn.functional.interpolate(d2, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))
        d3 = self.up3(d2)
        d3 = nn.functional.interpolate(d3, size=e2p.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2p], dim=1))
        d4 = self.up4(d3)
        d4 = nn.functional.interpolate(d4, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e1], dim=1))
        d5 = nn.functional.interpolate(d4, size=(Cfg.img_size, Cfg.img_size),
                        mode="bilinear", align_corners=False)
        d5 = self.dec5(d5)
        out = self.head(d5)
        return out.squeeze(1)                       # (B, 70, 70)


# ─────────────────────────────────────────────
# 4. 训练循环
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    progress = tqdm(loader, desc="train", leave=False)
    for seis, vel in progress:
        seis, vel = seis.to(device), vel.to(device)
        # seis: (B,5,1000,70)  vel: (B,70,70) or (B,1,70,70)
        vel = vel.squeeze(1) if vel.dim() == 4 else vel

        pred = model(seis)
        loss = criterion(pred, vel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = seis.size(0)
        total += loss.item() * bs
        n += bs
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    progress = tqdm(loader, desc="valid", leave=False)
    for seis, vel in progress:
        seis, vel = seis.to(device), vel.to(device)
        vel = vel.squeeze(1) if vel.dim() == 4 else vel
        pred = model(seis)
        loss = criterion(pred, vel)
        bs = seis.size(0)
        total += loss.item() * bs
        n += bs
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


# ─────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(Cfg.train_data_dir))
    parser.add_argument("--out_dir",  default=str(Cfg.output_dir))
    parser.add_argument("--epochs",   type=int, default=Cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(Cfg.seed)
    np.random.seed(Cfg.seed)

    # 5.1 收集数据
    pairs = find_pairs(args.data_dir)
    if len(pairs) == 0:
        raise RuntimeError("未找到任何数据对，请检查 --data_dir 路径")

    indices = build_flat_indices(pairs)
    # 按文件级别划分 train/val（避免同文件泄漏）
    file_ids = list(range(len(pairs)))
    tr_files, va_files = train_test_split(
        file_ids, test_size=Cfg.val_ratio, random_state=Cfg.seed
    )
    tr_set = set(tr_files); va_set = set(va_files)
    tr_idx = [idx for idx in indices if idx[0] in tr_set]
    va_idx = [idx for idx in indices if idx[0] in va_set]
    print(f"[info] train samples: {len(tr_idx)}, val samples: {len(va_idx)}")

    train_ds = SeisVelDataset(pairs, tr_idx)
    val_ds   = SeisVelDataset(pairs, va_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=Cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=Cfg.num_workers, pin_memory=True)

    # 5.2 模型
    model = UNet(in_ch=Cfg.n_src, base=Cfg.model_base_channels).to(Cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()  # MAE，与比赛指标一致

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, Cfg.device)
        va_loss = validate(model, val_loader, criterion, Cfg.device)
        scheduler.step()
        history.append({"epoch": epoch, "train_mae": tr_loss, "val_mae": va_loss})
        print(f"epoch {epoch:03d}  train_mae={tr_loss:.3f}  val_mae={va_loss:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_unet.pth"))
            print(f"  ✓ saved best (val_mae={va_loss:.3f})")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best val_mae = {best_val:.3f}")


if __name__ == "__main__":
    main()
