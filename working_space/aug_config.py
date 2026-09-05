"""Configuration registry and winner write-back helpers for augmentation search.

- ``AUG_CONFIGS``: all candidate augmentation configs (single source of truth
  referenced by aug_explore.ipynb)
- ``write_winner_aug()``: writes the winning augmentation to config.py's
  override file ``output/aug_explore/winner_aug.json`` and updates
  ``Cfg.augmentations`` (in memory) so formal training (train.py /
  preflight.ipynb) uses it directly.
- ``clear_winner_aug()``: deletes the override file, restoring config.py's
  default augmentations.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Candidate augmentation configs (dict format matches Cfg.augmentations)
# ---------------------------------------------------------------------------
AUG_NONE = {}
AUG_XFLIP = {"xflip": {"prob": 0.5}}
AUG_SHIFT = {"time_shift": {"prob": 0.5, "max_shift": 100}}
AUG_XFLIP_SHIFT = {                                   # production default
    "xflip": {"prob": 0.5},
    "time_shift": {"prob": 0.5, "max_shift": 100},
}
AUG_NOISE = {"noise": {"prob": 0.5, "sigma": 0.01}}
AUG_AMP = {"amplitude_scale": {"prob": 0.5, "low": 0.85, "high": 1.15}}
AUG_ALL = {                                            # all enabled, low intensity
    "xflip": {"prob": 0.5},
    "time_shift": {"prob": 0.5, "max_shift": 100},
    "noise": {"prob": 0.3, "sigma": 0.005},
    "amplitude_scale": {"prob": 0.3, "low": 0.9, "high": 1.1},
    "receiver_dropout": {"prob": 0.2, "drop_ratio": 0.1},
}

# (name, aug dict) -- order is the exploration order
AUG_CONFIGS = [
    ("none (control)", AUG_NONE),
    ("xflip", AUG_XFLIP),
    ("time_shift", AUG_SHIFT),
    ("xflip+shift (production)", AUG_XFLIP_SHIFT),
    ("noise", AUG_NOISE),
    ("amplitude_scale", AUG_AMP),
    ("all_light", AUG_ALL),
]

_CONFIG_OVERRIDE = PROJECT_ROOT / "output" / "aug_explore" / "winner_aug.json"


def winner_aug_path():
    """Return the override-file path for Cfg.augmentations."""
    return _CONFIG_OVERRIDE


def write_winner_aug(aug_dict, name="custom", comment=""):
    """Write the winning augmentation to the override file and update
    Cfg.augmentations (in memory).

    Args:
        aug_dict: augmentation config dict, e.g. {"noise": {"prob": 0.5, "sigma": 0.01}}
        name: config name (stored in JSON for traceability)
        comment: free-form note, e.g. best val MAE
    """
    from config import Cfg

    path = _CONFIG_OVERRIDE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "comment": comment, "augmentations": aug_dict}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    Cfg.augmentations = aug_dict
    print(f"[aug_config] wrote winning augmentation [{name}] -> {path}")
    print(f"[aug_config] Cfg.augmentations is now: {json.dumps(aug_dict, ensure_ascii=False)}")
    return path


def clear_winner_aug():
    """Delete the override file and restore config.py's default augmentations."""
    if _CONFIG_OVERRIDE.exists():
        _CONFIG_OVERRIDE.unlink()
        print(f"[aug_config] deleted override file {_CONFIG_OVERRIDE}; default augmentations restored.")
    else:
        print("[aug_config] no override file present.")


if __name__ == "__main__":
    # Simple self-check: print the registry and the override-file path
    print("AUG_CONFIGS:")
    for name, aug in AUG_CONFIGS:
        print(f"  {name:16s} {json.dumps(aug, ensure_ascii=False)}")
    print("winner_aug_path:", winner_aug_path())
