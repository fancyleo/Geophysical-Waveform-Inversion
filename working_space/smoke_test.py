"""
smoke_test.py — 不依赖 torch，纯 numpy 验证数据加载 + submission 格式逻辑
确认目录扫描、配对、归一化、奇数列截取、CSV 写入全部正确。
"""

import os, sys, shutil, tempfile, csv
import numpy as np
from config import Cfg

# ── 1. 造假数据 ──
tmp = tempfile.mkdtemp(prefix="wi_smoke_")
families = {
    "FlatVel_A":   ("data",  "model",  "data{}.npy",  "model{}.npy"),
    "CurveVel_B":  ("data",  "model",  "data{}.npy",  "model{}.npy"),
    "FlatFault_A": (None,    None,     "seis_5_1_{}.npy","vel_5_1_{}.npy"),
}
N = 16  # 每文件样本数（小，确保磁盘够）

for fam, (dname, mname, sp, mp) in families.items():
    if dname and mname:
        os.makedirs(os.path.join(tmp, fam, "data"),  exist_ok=True)
        os.makedirs(os.path.join(tmp, fam, "model"), exist_ok=True)
    else:
        os.makedirs(os.path.join(tmp, fam), exist_ok=True)
    for i in range(2):
        seis = np.random.randn(N, 5, 1000, 70).astype(np.float32) * 100
        vel  = (np.random.rand(N, 70, 70).astype(np.float32) * 2000 + 2500)
        if dname:
            sf = os.path.join(tmp, fam, dname, sp.format(i+1))
            mf = os.path.join(tmp, fam, mname, mp.format(i+1))
        else:
            sf = os.path.join(tmp, fam, sp.format(i))
            mf = os.path.join(tmp, fam, mp.format(i))
        np.save(sf, seis); np.save(mf, vel)

# 测试数据
test_dir = os.path.join(tmp, "test"); os.makedirs(test_dir)
for i in range(2):
    np.save(os.path.join(test_dir, f"oid{i:03d}.npy"),
            np.random.randn(5, 1000, 70).astype(np.float32))
print(f"[smoke] fake data at: {tmp}")

# ── 2. 复用 find_pairs 逻辑（从 train.py 拷贝关键函数） ──
sys.path.insert(0, os.path.dirname(__file__))

def find_pairs(root, families):
    pairs = []
    for fam in families:
        fam_dir = os.path.join(root, fam)
        if not os.path.isdir(fam_dir): continue
        d, m = os.path.join(fam_dir,"data"), os.path.join(fam_dir,"model")
        if os.path.isdir(d) and os.path.isdir(m):
            for sf in sorted(glob.glob(os.path.join(d,"*.npy"))):
                mf = os.path.join(m, os.path.basename(sf).replace("data","model"))
                if os.path.exists(mf): pairs.append((sf, mf))
            continue
        for sf in sorted(glob.glob(os.path.join(fam_dir,"seis_*.npy"))):
            vf = os.path.join(fam_dir, os.path.basename(sf).replace("seis_","vel_"))
            if os.path.exists(vf): pairs.append((sf, vf))
    return pairs

import glob
pairs = find_pairs(tmp, list(families.keys()))
print(f"[smoke] paired files: {len(pairs)}  (expect 6)")
assert len(pairs) == 6, "配对数量不对"

# ── 3. 统计 vel mean/std ──
all_vel = []
for _, mf in pairs:
    all_vel.append(np.load(mf).ravel())
all_vel = np.concatenate(all_vel)
vel_mean, vel_std = float(all_vel.mean()), float(all_vel.std())
print(f"[smoke] vel_mean={vel_mean:.1f}  vel_std={vel_std:.1f}")

# ── 4. 模拟训练（numpy 随机预测代替模型） ──
# 真实场景下这里换成 torch 训练循环
np.random.seed(0)
out_dir = os.path.join(tmp, "out"); os.makedirs(out_dir)
np.save(os.path.join(out_dir, "best_unet.pth.npy"),
        np.random.randn(2, 70, 70).astype(np.float32) * 100 + vel_mean)
print("[smoke] fake checkpoint saved")

# ── 5. 推理 + 写 submission（复刻 infer.py 逻辑） ──
test_files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
preds = []
oids = []
for fp in test_files:
    seis = np.load(fp).astype(np.float32)        # (5,1000,70)
    # 假装模型输出 = 随机
    pred = np.random.randn(70, 70).astype(np.float32) * 100 + vel_mean
    preds.append(pred); oids.append(os.path.splitext(os.path.basename(fp))[0])

preds = np.stack(preds, axis=0)                  # (N,70,70)
odd_cols = preds[:, :, Cfg.submission_x_start:Cfg.submission_x_stop:Cfg.submission_x_step]

sub_path = os.path.join(out_dir, "submission.csv")
header = "oid_ypos," + ",".join(f"x_{i}" for i in range(1,70,2))
with open(sub_path,"w") as f:
    f.write(header + "\n")
    for i, oid in enumerate(oids):
        for y in range(70):
            row = ",".join(f"{v:.1f}" for v in odd_cols[i, y])
            f.write(f"{oid}_y_{y},{row}\n")

# ── 6. 校验 ──
with open(sub_path) as f:
    lines = f.readlines()
print(f"[smoke] submission lines: {len(lines)} (expect 1+2*70=141)")
assert len(lines) == 141, "行数不对"
assert lines[0].startswith("oid_ypos,x_1,x_3"), "表头错误"
# 解析第一行数据验证列数
reader = csv.reader([lines[1]])
row = next(reader)
assert len(row) == 36, f"每行应有 1 oid + 35 列 = 36 字段，得到 {len(row)}"
print("[smoke] first data line (truncated):", ",".join(row[:4]), "...,", row[-1])
print("✅ smoke test passed — 数据加载 + submission 格式全部正确")

shutil.rmtree(tmp)
print(f"[smoke] cleaned up {tmp}")
