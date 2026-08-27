"""pretrain.py — 预训练/训练阶段的基础数据增强。

将原 ``data_aug.py`` 重写为 ``pretrain`` 模块，实现地震波形反演的基础数据增强。

设计约定
--------
每个增强是一个纯函数，签名为::

    fn(seismic: np.ndarray (n_src, n_steps, n_recv),
       velocity: np.ndarray (70, 70),
       rng: np.random.Generator, **params) -> (seismic, velocity)

函数通过 ``@register_aug("short_name")`` 注册，可在 ``config.py`` 的
``Cfg.augmentations`` 中按名字开关，无需改动本模块的调度逻辑。

增强作用于 *原始物理量*（seismic 在做 ``abs``/``log1p`` 之前、velocity 在做
均值/标准差归一化之前），以保证时间平移、幅值缩放、加噪等物理变换的有效性。

新增一个增强的步骤
------------------
1. 在本文件写一个签名为上面的函数。
2. 用 ``@register_aug("short_name")`` 装饰。
3. 在 ``config.py`` 的 ``Cfg.augmentations`` 中启用::

       "short_name": {"prob": 0.5, "any_param": value}
"""

import numpy as np

_AUGMENTATIONS = {}


def register_aug(name):
    """把 ``name`` 注册为一个数据增强函数。"""

    def decorator(fn):
        _AUGMENTATIONS[name] = fn
        return fn

    return decorator


def get_augmentation(name):
    """按名字返回已注册的增强函数。"""
    return _AUGMENTATIONS[name]


def augmentation_names():
    """返回所有已注册增强的名字列表。"""
    return list(_AUGMENTATIONS)


def apply_augmentation(seismic, velocity, config, rng, enabled=True):
    """按配置顺序依次应用增强。

    Args:
        seismic: 原始地震样本，shape (n_src, n_steps, n_recv)。
        velocity: 原始速度模型，shape (70, 70)。
        config: dict 映射 增强名 -> 参数（必须含 "prob"）。
        rng: numpy Generator，所有变换共用以保证可复现。
        enabled: 设为 False 跳过全部增强（用于验证集）。

    Returns:
        (seismic, velocity) 应用增强后的结果。
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
    """用固定种子应用配置的增强流程，返回 (raw, augmented)。

    便于测试/可视化：给定一个原始样本，得到增强前后的两份拷贝做对比。
    """
    rng = np.random.default_rng(seed)
    augmented_seismic, augmented_velocity = apply_augmentation(
        seismic, velocity, config, rng, enabled=True
    )
    # 翻转/平移可能产生负 stride 的 view，统一转成 C 连续以便显示与 torch 使用。
    augmented_seismic = np.ascontiguousarray(augmented_seismic)
    augmented_velocity = np.ascontiguousarray(augmented_velocity)
    return (seismic, velocity), (augmented_seismic, augmented_velocity)


# ---------------------------------------------------------------------------
# 第一梯队 - 几何类增强（低风险，收益最高）
# ---------------------------------------------------------------------------

@register_aug("xflip")
def xflip(seismic, velocity, rng, prob=0.5, **kwargs):
    """沿接收器轴镜像，速度模型同步镜像。

    物理依据：把线性接收阵列放到镜像位置观测是合法观测几何，因此左右翻转
    波形并镜像速度模型即可得到新的真实样本。

    NOTE: 假定 ``velocity[:, i]`` 沿接收器（水平）轴变化。若可视化发现水平轴
    是第一维，需把速度翻转改为 ``velocity[::-1, :]``。
    """
    if rng.random() < prob:
        seismic = seismic[..., ::-1]
        velocity = velocity[:, ::-1]
    return seismic, velocity


@register_aug("time_shift")
def time_shift(seismic, velocity, rng, prob=0.5, max_shift=100, **kwargs):
    """时间轴平移（震源激发延迟），零填充。

    物理依据：震源激发时刻改变会整体平移时程而不改变地下介质。用预分配的零
    缓冲 + 原位切片拷贝（而非 ``np.pad``+切片），既避免波形回绕，又让返回数组
    恰好为 (n_steps,) 大小，减少每样本临时内存（num_workers=0 时会累积到
    rank 进程的 host RSS）。
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
# 第二梯队 - 统计类增强（低风险）
# ---------------------------------------------------------------------------

@register_aug("noise")
def noise(seismic, velocity, rng, prob=0.5, sigma=0.01, **kwargs):
    """叠加高斯观测噪声。

    ``sigma`` 相对样本自身标准差取值，保证对不同幅值波形都稳健。
    """
    if rng.random() < prob:
        noise_std = sigma * float(seismic.std())
        seismic = seismic + rng.normal(0.0, noise_std, size=seismic.shape).astype(
            np.float32
        )
    return seismic, velocity


@register_aug("receiver_dropout")
def receiver_dropout(seismic, velocity, rng, prob=0.3, drop_ratio=0.15, **kwargs):
    """随机置零一部分接收道（模拟死道）。"""
    if rng.random() < prob:
        n_recv = seismic.shape[-1]
        n_drop = max(1, int(n_recv * drop_ratio))
        idx = rng.choice(n_recv, size=n_drop, replace=False)
        seismic = seismic.copy()
        seismic[..., idx] = 0.0
    return seismic, velocity


# ---------------------------------------------------------------------------
# 第三梯队 - 物理近似增强（中风险，建议 A/B 验证）
# ---------------------------------------------------------------------------

@register_aug("amplitude_scale")
def amplitude_scale(seismic, velocity, rng, prob=0.5, low=0.85, high=1.15, **kwargs):
    """缩放波形幅值（震源强度变化）。

    物理依据：对线性波动方程，更强/更弱的震源会线性缩放记录幅值而不改变速度。
    因为这里不做归一化，幅值变化范围保持温和。
    """
    if rng.random() < prob:
        scale = rng.uniform(low, high)
        seismic = seismic * scale
    return seismic, velocity


@register_aug("source_dropout")
def source_dropout(seismic, velocity, rng, prob=0.3, **kwargs):
    """置零一个随机震源通道（多源冗余）。"""
    if rng.random() < prob:
        src = int(rng.integers(0, seismic.shape[0]))
        seismic = seismic.copy()
        seismic[src] = 0.0
    return seismic, velocity
