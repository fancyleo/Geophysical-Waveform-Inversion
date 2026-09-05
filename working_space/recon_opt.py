"""Reconstruction-error optimization (Ruby, 14th-place solution).

Iteratively refine the input seismic so that the forward-modeled prediction
matches the observation:

    x_{k+1} = x_k - lam * (F(M(x_k)) - x_0)

  M   - our NN inverse model  (seismic -> velocity), working_space.model.UNet
  F   - forward simulator     (velocity -> seismic), working_space.forward_model
  x_0 - original observed seismic (raw physical traces, before sign·log1p)

This is pure inference-time refinement: no retraining, and it is orthogonal to
ensembling (the refined input can then be fed to any model/ensemble).

Physics intuition: M approximates the true inverse map locally, so
F(M(x+dx)) - F(M(x)) ~= dx; pushing x toward the observation along the
reconstruction residual reduces inversion error without needing gradients of F.

Usage:
    python working_space/recon_opt.py --ckpt <path> --family FlatVel_A --n 100 \
        --lam 0.8 --n_iter 3
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg, resolve_device
from model import UNet
from forward_model import vel_to_seis


def abs_log1p(x):
    """Legacy preprocessing (pre-sign·log1p checkpoints)."""
    return np.log1p(np.abs(x))


def sign_log1p(x):
    """Production preprocessing (matches data.py / infer.py)."""
    return np.sign(x) * np.log1p(np.abs(x))


FEATURES = {"sign": sign_log1p, "abs": abs_log1p}


@torch.no_grad()
def _predict_norm(models, feat, device):
    """Ensemble mean normalized velocity (70,70) from a preprocessed feature."""
    x = torch.from_numpy(feat[None]).float().to(device, non_blocking=True)
    preds = torch.stack([m(x) for m in models])   # (n_models, 1, 70, 70)
    return preds.mean(0)[0].cpu().numpy()


@torch.no_grad()
def baseline_velocity(models, x_raw, vel_mean, vel_std, device,
                      feature_fn=sign_log1p):
    """Ensemble mean prediction: raw seismic (5,1000,70) -> velocity (70,70) m/s."""
    feat = feature_fn(x_raw).astype(np.float32)
    return _predict_norm(models, feat, device) * vel_std + vel_mean


@torch.no_grad()
def recon_opt(models, x_raw, vel_mean, vel_std, device,
              lam=0.8, n_iter=3, forward=vel_to_seis, trace=False,
              feature_fn=sign_log1p, early_stop=True, grow_tol=1.05):
    """Single-sample reconstruction-error optimization (ensemble M).

    Args:
        models: list of UNet in eval mode; prediction is the ensemble mean.
            feature_fn must match their training preprocessing (old checkpoints:
            abs_log1p; new: sign_log1p).
        x_raw: (5, 1000, 70) observed seismic (raw).
        lam:   update step. Use a SMALL value (0.1-0.3) when residuals are large
            relative to the signal; our 1/47-data models diverge at lam>=0.6.
        n_iter: max refinement steps.
        forward: callable vel(70,70) -> seis(5,1000,70).
        trace: collect per-iteration residual RMS.
        feature_fn: preprocessing used to feed the models.
        early_stop: stop if residual RMS grows > grow_tol (divergence guard).
        grow_tol: multiplicative growth tolerance for early stop.

    Returns:
        (v_opt, x_opt, history) with v_opt (70,70) m/s after refinement and
        x_opt the refined seismic (5,1000,70).
    """
    x = np.asarray(x_raw, dtype=np.float32).copy()
    history = []
    prev_rms = None
    for k in range(n_iter):
        v = baseline_velocity(models, x, vel_mean, vel_std, device,
                              feature_fn=feature_fn)               # (70,70) m/s
        recons = forward(v)                                          # (5,1000,70)
        resid = recons - x_raw                                       # fixed obs target
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if early_stop and prev_rms is not None and rms > prev_rms * grow_tol:
            break   # diverging: keep current x and stop
        x = x - float(lam) * resid
        prev_rms = rms
        if trace:
            history.append({"iter": k + 1, "resid_rms": rms,
                            "recons_std": float(recons.std())})
    v_opt = baseline_velocity(models, x, vel_mean, vel_std, device,
                              feature_fn=feature_fn)
    return v_opt, x, history


# ---------------------------------------------------------------------------
# Validation entry point (hold-out file from training data, has true velocity)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    from data import find_pairs

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",
                        default=str(Cfg.project_root / "output/model_260829_2211/best_unet.pth"),
                        help="single checkpoint (used when --ckpts not given)")
    parser.add_argument("--ckpts",
                        default=",".join([
                            str(Cfg.project_root / "output/model_260828_1042/best_unet.pth"),
                            str(Cfg.project_root / "output/model_260824_0501kaggle/best_unet.pth"),
                            str(Cfg.project_root / "output/model_260829_2211/best_unet.pth"),
                        ]),
                        help="comma-separated checkpoints -> ensemble-mean M")
    parser.add_argument("--family", default="FlatVel_A")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.3,
                        help="update step; use SMALL (0.1-0.3) when residuals diverge")
    parser.add_argument("--n_iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_early_stop", action="store_true",
                        help="disable divergence guard")
    parser.add_argument("--feature", choices=["sign", "abs"], default="sign",
                        help="preprocessing that matches the checkpoints (sign=new, abs=legacy)")
    args = parser.parse_args()
    feature_fn = FEATURES[args.feature]
    ckpt_list = [p.strip() for p in (args.ckpts or args.ckpt).split(",") if p.strip()]

    device = resolve_device()
    vel_mean, vel_std = Cfg.vel_mean, Cfg.vel_std
    print(f"device: {device}")
    print(f"family: {args.family} | n={args.n} | lam={args.lam} | n_iter={args.n_iter} "
          f"| feature={args.feature} | early_stop={not args.no_early_stop}")
    print(f"checkpoints ({len(ckpt_list)}):")
    for ck in ckpt_list:
        print(f"  - {ck}")

    models = []
    for ck in ckpt_list:
        m = UNet(in_ch=Cfg.n_src, base=Cfg.model_base_channels).to(device)
        state = torch.load(ck, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    pairs = find_pairs(Cfg.train_data_dir, [args.family])
    if not pairs:
        raise SystemExit(f"no data for {args.family}")
    seis_path, vel_path = pairs[0]          # file 0 as hold-out (no leakage)
    seis_arr = np.load(seis_path, mmap_mode="r")
    vel_arr = np.load(vel_path, mmap_mode="r")
    n = min(args.n, int(seis_arr.shape[0]))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(int(seis_arr.shape[0]), size=n, replace=False)

    mae_base, mae_opt, resid_trend = [], [], []
    t0 = time.perf_counter()
    for j, i in enumerate(idx, 1):
        x_raw = np.array(seis_arr[i], dtype=np.float32)            # (5,1000,70)
        v_true = np.array(vel_arr[i], dtype=np.float32).squeeze()  # (70,70)
        v0 = baseline_velocity(models, x_raw, vel_mean, vel_std, device,
                               feature_fn=feature_fn)
        v1, _, hist = recon_opt(models, x_raw, vel_mean, vel_std, device,
                                lam=args.lam, n_iter=args.n_iter, trace=True,
                                feature_fn=feature_fn,
                                early_stop=not args.no_early_stop)
        mae_base.append(float(np.abs(v0 - v_true).mean()))
        mae_opt.append(float(np.abs(v1 - v_true).mean()))
        resid_trend.append([h["resid_rms"] for h in hist])
        if j % 20 == 0 or j == n:
            el = (time.perf_counter() - t0) / j
            print(f"  [{j}/{n}] base={np.mean(mae_base):6.1f}  "
                  f"opt(lam={args.lam},it={args.n_iter})={np.mean(mae_opt):6.1f}  "
                  f"delta={np.mean(mae_opt)-np.mean(mae_base):+6.1f}  {el:5.1f}s/smp", flush=True)

    print(f"\n===== {args.family} n={n} =====")
    print(f"baseline MAE  : {np.mean(mae_base):.1f} m/s")
    print(f"recon-opt MAE : {np.mean(mae_opt):.1f} m/s  (lam={args.lam}, n_iter={args.n_iter})")
    print(f"delta         : {np.mean(mae_opt)-np.mean(mae_base):+.1f} m/s")
    if args.n_iter > 1 and resid_trend:
        # early_stop 会让各样本 history 长度不一 -> pad with NaN, then nanmean
        max_len = max(len(t) for t in resid_trend)
        padded = np.full((len(resid_trend), max_len), np.nan)
        for r_i, t in enumerate(resid_trend):
            padded[r_i, :len(t)] = t
        trend = np.nanmean(padded, axis=0)
        print("residual RMS per iteration:", " ".join(f"{t:.4f}" for t in trend))
