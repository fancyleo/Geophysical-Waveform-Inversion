"""pretrain.py — basic data augmentation for the pretrain/training pipeline.

The original ``data_aug.py`` was rewritten as the ``pretrain`` module,
implementing basic data augmentation for seismic waveform inversion.

Design conventions
------------------
Every augmentation is a pure function with the signature::

    fn(seismic: np.ndarray (n_src, n_steps, n_recv),
       velocity: np.ndarray (70, 70),
       rng: np.random.Generator, **params) -> (seismic, velocity)

Functions are registered with ``@register_aug("short_name")`` and can be
toggled by name via ``Cfg.augmentations`` in ``config.py``, without touching
this module's dispatch logic.

Augmentations operate on the *raw physical quantities* — the seismic trace
before ``abs``/``log1p`` and the velocity before mean/std normalization — so
physics-based transforms (time shift, amplitude scaling, noise) stay valid.

How to add a new augmentation
-----------------------------
1. Write a function with the signature above in this file.
2. Decorate it with ``@register_aug("short_name")``.
3. Enable it in ``Cfg.augmentations`` in ``config.py``::

       "short_name": {"prob": 0.5, "any_param": value}
"""

import numpy as np

_AUGMENTATIONS = {}


def register_aug(name):
    """Register ``name`` as a data augmentation function."""

    def decorator(fn):
        _AUGMENTATIONS[name] = fn
        return fn

    return decorator


def get_augmentation(name):
    """Return the registered augmentation function by name."""
    return _AUGMENTATIONS[name]


def augmentation_names():
    """List the names of all registered augmentations."""
    return list(_AUGMENTATIONS)


def apply_augmentation(seismic, velocity, config, rng, enabled=True):
    """Apply the configured augmentations in order.

    Args:
        seismic: raw seismic sample, shape (n_src, n_steps, n_recv).
        velocity: raw velocity model, shape (70, 70).
        config: dict mapping augmentation name -> params (must include "prob").
        rng: numpy Generator shared by every transform for reproducibility.
        enabled: set False to skip all augmentations (used for validation).

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


def augment_sample(seismic, velocity, config, seed=0):
    """Apply the configured pipeline with a fixed seed; returns (raw, augmented).

    Useful for testing/visualization: given one raw sample, get the raw and
    augmented copies for comparison.
    """
    rng = np.random.default_rng(seed)
    augmented_seismic, augmented_velocity = apply_augmentation(
        seismic, velocity, config, rng, enabled=True
    )
    # Flips/shifts can produce views with negative strides; restore C-contiguous buffers for display and torch usage.
    augmented_seismic = np.ascontiguousarray(augmented_seismic)
    augmented_velocity = np.ascontiguousarray(augmented_velocity)
    return (seismic, velocity), (augmented_seismic, augmented_velocity)


# ---------------------------------------------------------------------------
# Tier 1 - geometry augmentations (low risk, highest value)
# ---------------------------------------------------------------------------

@register_aug("xflip")
def xflip(seismic, velocity, rng, prob=0.5, **kwargs):
    """Mirror the receiver axis and the matching velocity axis.

    Physical basis: observing with a linear receiver array from the mirrored
    position is a valid acquisition geometry, so flipping the waveform
    left-right and mirroring the velocity model yields a realistic new sample.

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
    time without changing the subsurface. Uses a preallocated zero buffer and
    in-place slice copy (instead of ``np.pad`` + slice) so the waveform does not
    wrap around AND the returned array is exactly (n_steps,) sized. This avoids
    a large per-sample temporary, which matters because with num_workers=0 every
    temporary is allocated in the rank process and feeds host-RSS growth.
    """
    if rng.random() < prob:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        if shift != 0:
            n_steps = seismic.shape[1]
            shifted = np.zeros_like(seismic)
            if shift > 0:
                shifted[:, :n_steps - shift, :] = seismic[:, shift:, :]
            else:
                shifted[:, -shift:, :] = seismic[:, :n_steps + shift, :]
            seismic = shifted
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
