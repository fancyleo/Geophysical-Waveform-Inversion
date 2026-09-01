"""Forward modeling: velocity map -> seismic traces (2D acoustic FDTD).

Ported from jaewook704's Kaggle notebook "waveform-inversion-vel-to-seis"
(https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis),
which follows the OpenFWI / KAUST SeismicInversion FD lab
(arxiv 2111.02926, csim.kaust.edu.sa/files/SeismicInversion/Chapter.FD/lab.FD2.8).

Physics: 2nd-order-in-time, 24th-order-in-space acoustic wave equation with a
2nd-order ABC (absorbing boundary condition) layer. Ricker source wavelet at
15 Hz. Geometry matches the competition data:

  - grid: 70 x 70 cells, dx = 10 m  -> 700 m x 700 m model
  - time: dt = 1e-3 s, nt = 1000 samples
  - sources: 5 at x cells [1, 18, 35, 53, 70] (= x 10/180/350/530/700 m), z = 10 m
  - receivers: 70 at x = (1..70)*10 m, z = 10 m

Input  : vel (70, 70) velocity map (m/s)
Output : seis (5, nt, 70) float32 — one seismic gather per source.

Usage:
    from forward_model import vel_to_seis
    seis = vel_to_seis(vel)          # (5, 1000, 70)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Wavelet / geometry / boundary helpers (kept close to the original)
# ---------------------------------------------------------------------------

def ricker(f, dt, nt=None):
    """Ricker wavelet; 1-based indexing (w[0] unused). Returns (w, tw)."""
    nw = int(2.2 / f / dt)
    nw = 2 * (nw // 2) + 1
    nc = nw // 2 + 1  # 1-based center index

    k = np.arange(1, nw + 1)          # 1-based
    alpha = (nc - k) * f * dt * np.pi
    beta = alpha ** 2
    w0 = (1.0 - 2.0 * beta) * np.exp(-beta)

    if nt is not None:
        if nt < len(w0):
            raise ValueError("nt is smaller than condition!")
        w = np.zeros(nt + 1)
        w[1:len(w0) + 1] = w0
    else:
        w = np.zeros(len(w0) + 1)
        w[1:] = w0
    tw = np.arange(1, len(w)) * dt
    return w, tw


def padvel(v0, nbc):
    """Pad velocity with edge values, plus one extra row/col (1-based grid)."""
    v_padded = np.pad(v0, ((nbc, nbc), (nbc, nbc)), mode="edge")
    nz, nx = v_padded.shape
    v = np.zeros((nz + 1, nx + 1))
    v[1:, 1:] = v_padded
    return v


def expand_source(s0, nt):
    """Expand source series into a 1-based length-(nt+1) array."""
    s0 = np.asarray(s0).flatten()
    s = np.zeros(nt + 1)
    s[1:len(s0) + 1] = s0
    return s


def adjust_sr(coord, dx, nbc):
    """Physical source/receiver coords -> padded grid cell indices (1-based)."""
    isx = int(round(coord["sx"] / dx)) + 1 + nbc
    isz = int(round(coord["sz"] / dx)) + 1 + nbc
    igx = (np.round(np.array(coord["gx"]) / dx) + 1 + nbc).astype(int)
    igz = (np.round(np.array(coord["gz"]) / dx) + 1 + nbc).astype(int)
    if abs(coord["sz"]) < 0.5:
        isz += 1
    igz = igz + (np.abs(np.array(coord["gz"])) < 0.5).astype(int)
    return isx, isz, igx, igz


def AbcCoef2D(vel, nbc, dx):
    """2nd-order ABC damping coefficients for the padded grid."""
    nzbc, nxbc = vel.shape[1] - 1, vel.shape[0] - 1
    velmin = np.min(vel[1:, 1:])
    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc

    a = (nbc - 1) * dx
    kappa = 3.0 * velmin * np.log(1e7) / (2.0 * a)

    damp1d = kappa * (((np.arange(1, nbc + 1) - 1) * dx / a) ** 2)
    damp = np.zeros((nzbc + 1, nxbc + 1))

    for iz in range(1, nzbc + 1):
        damp[iz, 1:nbc + 1] = damp1d[::-1]
        damp[iz, nx + nbc + 1: nx + 2 * nbc + 1] = damp1d
    for ix in range(nbc + 1, nbc + nx + 1):
        damp[1:nbc + 1, ix] = damp1d[::-1]
        damp[nz + nbc + 1: nz + 2 * nbc + 1, ix] = damp1d
    return damp


# ---------------------------------------------------------------------------
# Main FDTD loop
# ---------------------------------------------------------------------------

def a2d_mod_abc24(v, nbc, dx, nt, dt, s, coord, isFS=False):
    """2D acoustic FDTD (24th-order space, 2nd-order time, 2nd-order ABC).

    Returns seis (nt+1, ng); time axis 1-based, row 0 is zero padding.
    """
    ng = len(coord["gx"])
    seis = np.zeros((nt + 1, ng))

    c1 = -2.5
    c2 = 4.0 / 3.0
    c3 = -1.0 / 12.0

    v = padvel(v, nbc)
    abc = AbcCoef2D(v, nbc, dx)

    alpha = (v * dt / dx) ** 2
    kappa = abc * dt
    temp1 = 2 + 2 * c1 * alpha - kappa
    temp2 = 1 - kappa
    beta_dt = (v * dt) ** 2
    s = expand_source(s, nt)
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)

    p0 = np.zeros_like(v)
    p1 = np.zeros_like(v)

    for it in range(1, nt + 1):
        p = (temp1 * p1 - temp2 * p0 +
             alpha * (
                 c2 * (np.roll(p1, 1, axis=1) + np.roll(p1, -1, axis=1) +
                       np.roll(p1, 1, axis=0) + np.roll(p1, -1, axis=0)) +
                 c3 * (np.roll(p1, 2, axis=1) + np.roll(p1, -2, axis=1) +
                       np.roll(p1, 2, axis=0) + np.roll(p1, -2, axis=0))
             ))

        # Source injection
        p[isz, isx] += beta_dt[isz, isx] * s[it]

        # Free surface (not used by the competition data)
        if isFS:
            p[nbc, :] = 0.0
            p[nbc - 1: nbc + 1, :] = -p[nbc + 1: nbc + 3, :]

        # Vectorized receiver recording (original had a slow per-receiver loop)
        seis[it, :] = p[igz, igx]

        p0, p1 = p1, p

    return seis


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_IDX = [1, 18, 35, 53, 70]   # x grid cells (1-based in original)


def vel_to_seis(vel, nt=1000, source_idx=None, dx=10.0, dt=1e-3, freq=15,
                nbc=120, isFS=False):
    """Forward-model a velocity map into 5-source seismic gathers.

    Args:
        vel: (70, 70) velocity map (m/s).
        nt: number of time samples (competition data uses 1000).
        source_idx: optional 5-array of source x positions in grid cells
            (0-based cells). Default matches competition geometry
            ([1,18,35,53,70] in original 1-based indexing).
        dx / dt / freq / nbc / isFS: physical / boundary parameters.

    Returns:
        seis: (5, nt, 70) float32.
    """
    vel = np.asarray(vel, dtype=np.float64).squeeze()
    if vel.shape != (70, 70):
        raise ValueError(f"vel must be (70, 70), got {vel.shape}")

    if source_idx is None:
        source_idx = DEFAULT_SOURCE_IDX
    if len(source_idx) != 5:
        raise ValueError("expected 5 sources")

    s, _ = ricker(freq, dt)                  # short wavelet; expanded in FDTD loop

    coord = {
        "sz": 1 * dx,
        "gx": np.arange(1, 71) * dx,         # receivers at x = 1..70 cells
        "gz": np.ones(70) * dx,
    }

    gathers = []
    for sx in source_idx:
        coord["sx"] = sx * dx
        seis = a2d_mod_abc24(vel, nbc, dx, nt, dt, s, coord, isFS)
        gathers.append(seis[1:])             # drop 1-based row 0 -> (nt, 70)

    return np.stack(gathers, axis=0).astype(np.float32)   # (5, nt, 70)


# ---------------------------------------------------------------------------
# Smoke / validation entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import find_pairs
    from config import Cfg

    pairs = find_pairs(Cfg.train_data_dir, ["FlatVel_A"])
    if not pairs:
        raise SystemExit("no FlatVel_A data found")
    seis_path, vel_path = pairs[0]   # find_pairs returns (seismic, velocity)
    vel = np.load(vel_path, mmap_mode="r")[0].squeeze().astype(np.float64)
    real_seis = np.load(seis_path, mmap_mode="r")[0].astype(np.float32)

    print(f"vel shape: {vel.shape} | real seis shape: {real_seis.shape}")
    t0 = __import__("time").perf_counter()
    pred = vel_to_seis(vel)
    el = __import__("time").perf_counter() - t0
    print(f"forward seis shape: {pred.shape} | time: {el:.1f}s")

    # Compare magnitude / correlation with the real gather (per source).
    for s_i in range(5):
        a = pred[s_i]
        b = real_seis[s_i]
        corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        print(f"src{s_i}: pred std={a.std():.4f} real std={b.std():.4f} "
              f"corr={corr:.3f}")
