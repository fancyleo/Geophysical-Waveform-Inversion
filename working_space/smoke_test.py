"""Validate data pairing and submission formatting without requiring PyTorch."""

import os, sys, shutil, tempfile, csv
import numpy as np
from config import Cfg

# 1. Create synthetic data.
tmp = tempfile.mkdtemp(prefix="wi_smoke_")
families = {
    "FlatVel_A":   ("data",  "model",  "data{}.npy",  "model{}.npy"),
    "CurveVel_B":  ("data",  "model",  "data{}.npy",  "model{}.npy"),
    "FlatFault_A": (None,    None,     "seis_5_1_{}.npy","vel_5_1_{}.npy"),
}
N = 16  # Keep each file small so the test uses minimal disk space.

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

# Create synthetic test data.
test_dir = os.path.join(tmp, "test"); os.makedirs(test_dir)
for i in range(2):
    np.save(os.path.join(test_dir, f"oid{i:03d}.npy"),
            np.random.randn(5, 1000, 70).astype(np.float32))
print(f"[smoke] fake data at: {tmp}")

# 2. Reuse the file-pairing logic in a dependency-free test.
sys.path.insert(0, os.path.dirname(__file__))

def find_pairs(root, families):
    """Find matching seismic and velocity files for the selected families."""
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
print(f"[smoke] paired files: {len(pairs)}  (expected 6)")
assert len(pairs) == 6, "Unexpected number of file pairs"

# 3. Compute velocity mean and standard deviation.
all_vel = []
for _, mf in pairs:
    all_vel.append(np.load(mf).ravel())
all_vel = np.concatenate(all_vel)
vel_mean, vel_std = float(all_vel.mean()), float(all_vel.std())
print(f"[smoke] vel_mean={vel_mean:.1f}  vel_std={vel_std:.1f}")

# 4. Simulate training with random NumPy predictions.
np.random.seed(0)
out_dir = os.path.join(tmp, "out"); os.makedirs(out_dir)
np.save(os.path.join(out_dir, "best_unet.pth.npy"),
        np.random.randn(2, 70, 70).astype(np.float32) * 100 + vel_mean)
print("[smoke] fake checkpoint saved")

# 5. Simulate inference and write a submission.
test_files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
preds = []
oids = []
for fp in test_files:
    seis = np.load(fp).astype(np.float32)        # (5,1000,70)
    # Use random values as a stand-in for model output.
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

# 6. Validate the generated submission.
with open(sub_path) as f:
    lines = f.readlines()
print(f"[smoke] submission lines: {len(lines)} (expected 1+2*70=141)")
assert len(lines) == 141, "Unexpected number of submission rows"
assert lines[0].startswith("oid_ypos,x_1,x_3"), "Invalid submission header"
# Parse the first data row and validate its field count.
reader = csv.reader([lines[1]])
row = next(reader)
assert len(row) == 36, f"Expected 36 fields per row, got {len(row)}"
print("[smoke] first data line (truncated):", ",".join(row[:4]), "...,", row[-1])
print("Smoke test passed: data loading and submission format are valid")

shutil.rmtree(tmp)
print(f"[smoke] cleaned up {tmp}")
