"""Legacy compatibility shim — superseded by :mod:`pretrain`.

The original ``data_aug.py`` was rewritten as ``pretrain.py`` (basic data
augmentation for the pretrain/training pipeline). This file is kept only so
legacy imports (e.g. ``from data_aug import apply_augmentation``) keep working;
new code should import from ``pretrain`` instead.
"""

from pretrain import *  # noqa: F401,F403
from pretrain import (  # noqa: F401
    apply_augmentation,
    augment_sample,
    augmentation_names,
    get_augmentation,
    register_aug,
)
