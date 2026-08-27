"""Legacy compatibility shim — superseded by :mod:`pretrain`.

原 ``data_aug.py`` 已重写为 ``pretrain.py``（预训练/训练阶段的基础数据增强）。
本文件仅保留以便旧引用（如 ``from data_aug import apply_augmentation``）仍可
工作；新代码请从 ``pretrain`` 导入。
"""

from pretrain import *  # noqa: F401,F403
from pretrain import (  # noqa: F401
    apply_augmentation,
    augment_sample,
    augmentation_names,
    get_augmentation,
    register_aug,
)
