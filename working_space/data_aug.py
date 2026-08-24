"""Data augmentation for seismic waveform inversion.

Each augmentation is a pure function with the signature::

    fn(seismic: np.ndarray (n_src, n_steps, n_recv),
       velocity: np.ndarray (70, 70),
       rng: np.random.Generator, **params) -> (seismic, velocity)

and is registered under a short name so it can be enabled from ``config.py``
without touching this file's dispatch logic.

Augmentations operate on the *raw physical quantities*: the seismic trace
before ``abs``/``log1p`` and the velocity before mean/std normalization. This
keeps physics-based transforms (time shift, amplitude scaling, noise) valid.

How to add a new augmentation
-----------------------------
1. Write a function with the signature above.
2. Decorate it with ``@register_aug("short_name")``.
3. Enable it in ``Cfg.augmentations`` (see ``config.py``)::

       "short_name": {"prob": 0.5, "any_param": value}
"""

import numpy as np

_AUGMENTATIONS = {}


def register_aug(name):
    """Register an augmentation function under ``name``."""

    def decorator(fn):
        _AUGMENTATIONS[name] = fn
        return fn

    return decorator


def get_augmentation(name):
    """Return the registered augmentation function by name."""
    return _AUGMENTATIONS[name]


def augmentation_names():
    """List all registered augmentation names."""
    return list(_AUGMENTATIONS)


def apply_augmentation(seismic, velocity, config, rng, enabled=True):
    """Apply the configured augmentations in order.

    Args:
        seismic: raw seismic sample, shape (n_src, n_steps, n_recv).
        velocity: raw velocity model, shape (70, 70).
        config: dict mapping augmentation name -> params (must include "prob").
        rng: numpy Generator used by every transform for reproducibility.
        enabled: set False to skip all augmentation (used for validation).

    Returns:
        (seismic, velocity) after applying the enabled augmentations.
    """
    if not enabled or not config:
        return seismic, velocity
    for name, params in config.items():
        fn = _AUGMENTATIONS.get(name)
        if fn is None:
            continue
        seismic, velocity = fn(seismic, velocity, rng, **params)
    return seismic, velocity


# ---------------------------------------------------------------------------
# Tier 1 - geometry augmentations (low risk, highest value)
# ---------------------------------------------------------------------------

@register_aug("xflip")
def xflip(seismic, velocity, rng, prob=0.5, **kwargs):
    """Mirror the receiver axis and the matching velocity axis.

    Physical basis: a linear receiver array observed from the mirror position
    is a valid acquisition geometry, so flipping the waveform left-right and
    mirroring the velocity model yields a realistic new sample.

    NOTE: assumes ``velocity[:, i]`` varies along the receiver (horizontal)
    axis. If a visualization shows the horizontal axis is the first dimension,
    change the velocity flip to ``velocity[::-1, :]``.
    """
    if rng.random() < prob:
        seismic = seismic[..., ::-1]
        velocity = velocity[:, ::-1]
    return seismic, velocity


@register_aug("time_shift")
def time_shift(seismic, velocity, rng, prob=0.5, max_shift=100, **kwargs):
    """Shift the time axis (source excitation delay) with zero padding.

    Physical basis: changing when the source fires shifts the whole trace in
    time without changing the subsurface. Uses pad + slice instead of
    ``np.roll`` so the waveform does not wrap around (non-physical).
    """
    if rng.random() < prob:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        if shift != 0:
            n_steps = seismic.shape[1]
            padded = np.pad(seismic, ((0, 0), (max_shift, max_shift), (0, 0)))
            start = max_shift + shift
            seismic = padded[:, start:start + n_steps, :]
    return seismic, velocity


# ---------------------------------------------------------------------------
# Tier 2 - statistical augmentations (low risk)
# ---------------------------------------------------------------------------

@register_aug("noise")
def noise(seismic, velocity, rng, prob=0.5, sigma=0.01, **kwargs):
    """Add Gaussian observation noise.

    ``sigma`` is relative to the sample's own standard deviation so it stays
    robust to different trace amplitudes.
    """
    if rng.random() < prob:
        noise_std = sigma * float(seismic.std())
        seismic = seismic + rng.normal(0.0, noise_std, size=seismic.shape).astype(
            np.float32
        )
    return seismic, velocity


@register_aug("receiver_dropout")
def receiver_dropout(seismic, velocity, rng, prob=0.3, drop_ratio=0.15, **kwargs):
    """Zero out a random subset of receiver traces (dead-channel simulation)."""
    if rng.random() < prob:
        n_recv = seismic.shape[-1]
        n_drop = max(1, int(n_recv * drop_ratio))
        idx = rng.choice(n_recv, size=n_drop, replace=False)
        seismic = seismic.copy()
        seismic[..., idx] = 0.0
    return seismic, velocity


# ---------------------------------------------------------------------------
# Tier 3 - physics-approximation augmentations (medium risk, verify by A/B)
# ---------------------------------------------------------------------------

@register_aug("amplitude_scale")
def amplitude_scale(seismic, velocity, rng, prob=0.5, low=0.85, high=1.15, **kwargs):
    """Scale trace amplitudes (source-strength variation).

    Physical basis: for the linear wave equation, a stronger/weaker source
    scales the recorded amplitudes linearly without changing the velocity.
    Keep the range modest because the traces are not normalized here.
    """
    if rng.random() < prob:
        scale = rng.uniform(low, high)
        seismic = seismic * scale
    return seismic, velocity


@register_aug("source_dropout")
def source_dropout(seismic, velocity, rng, prob=0.3, **kwargs):
    """Zero out one random source channel (multi-source redundancy)."""
    if rng.random() < prob:
        src = int(rng.integers(0, seismic.shape[0]))
        seismic = seismic.copy()
        seismic[src] = 0.0
    return seismic, velocity
