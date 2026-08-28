"""数据增强探索的配置注册表与胜出写回工具。

- ``AUG_CONFIGS``：全部候选增强配置（aug_explore.ipynb 引用，单一数据源）
- ``write_winner_aug()``：把胜出增强写入 config.py 的覆盖文件
  ``output/aug_explore/winner_aug.json``，并更新 ``Cfg.augmentations``（内存），
  使正式训练（train.py / preflight.ipynb）直接使用该增强。
- ``clear_winner_aug()``：删除覆盖文件，恢复 config.py 默认增强。
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 候选增强配置（dict 格式与 Cfg.augmentations 一致）
# ---------------------------------------------------------------------------
AUG_NONE = {}
AUG_XFLIP = {"xflip": {"prob": 0.5}}
AUG_SHIFT = {"time_shift": {"prob": 0.5, "max_shift": 100}}
AUG_XFLIP_SHIFT = {                                   # 生产默认
    "xflip": {"prob": 0.5},
    "time_shift": {"prob": 0.5, "max_shift": 100},
}
AUG_NOISE = {"noise": {"prob": 0.5, "sigma": 0.01}}
AUG_AMP = {"amplitude_scale": {"prob": 0.5, "low": 0.85, "high": 1.15}}
AUG_ALL = {                                            # 全部启用、低强度
    "xflip": {"prob": 0.5},
    "time_shift": {"prob": 0.5, "max_shift": 100},
    "noise": {"prob": 0.3, "sigma": 0.005},
    "amplitude_scale": {"prob": 0.3, "low": 0.9, "high": 1.1},
    "receiver_dropout": {"prob": 0.2, "drop_ratio": 0.1},
}

# (名称, 增强 dict) —— 顺序即探索执行顺序
AUG_CONFIGS = [
    ("none(对照)", AUG_NONE),
    ("xflip", AUG_XFLIP),
    ("time_shift", AUG_SHIFT),
    ("xflip+shift(生产)", AUG_XFLIP_SHIFT),
    ("noise", AUG_NOISE),
    ("amplitude_scale", AUG_AMP),
    ("all_light", AUG_ALL),
]

_CONFIG_OVERRIDE = PROJECT_ROOT / "output" / "aug_explore" / "winner_aug.json"


def winner_aug_path():
    """返回 Cfg.augmentations 的覆盖文件路径。"""
    return _CONFIG_OVERRIDE


def write_winner_aug(aug_dict, name="custom", comment=""):
    """把胜出增强写回覆盖文件，并更新 Cfg.augmentations（内存）。

    Args:
        aug_dict: 增强配置 dict，如 {"noise": {"prob": 0.5, "sigma": 0.01}}
        name: 配置名称（写入 JSON 便于追溯）
        comment: 备注，如 best val MAE
    """
    from config import Cfg

    path = _CONFIG_OVERRIDE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "comment": comment, "augmentations": aug_dict}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    Cfg.augmentations = aug_dict
    print(f"[aug_config] 已写回胜出增强 [{name}] -> {path}")
    print(f"[aug_config] Cfg.augmentations 现为：{json.dumps(aug_dict, ensure_ascii=False)}")
    return path


def clear_winner_aug():
    """删除覆盖文件，恢复 config.py 默认增强。"""
    if _CONFIG_OVERRIDE.exists():
        _CONFIG_OVERRIDE.unlink()
        print(f"[aug_config] 已删除覆盖文件 {_CONFIG_OVERRIDE}，恢复默认增强。")
    else:
        print("[aug_config] 无覆盖文件。")


if __name__ == "__main__":
    # 简单自检：打印注册表与覆盖文件路径
    print("AUG_CONFIGS:")
    for name, aug in AUG_CONFIGS:
        print(f"  {name:16s} {json.dumps(aug, ensure_ascii=False)}")
    print("winner_aug_path:", winner_aug_path())
