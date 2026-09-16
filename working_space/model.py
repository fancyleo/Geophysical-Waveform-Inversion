"""Models mapping five-source seismic data to a 70 x 70 velocity map.

Two architectures are available (select with ``--model`` / ``Cfg.model_name``):

``unet``
    The project's original U-Net: symmetric 3x3 kernels, MaxPool downsampling on
    both axes, ConvTranspose upsampling, bilinear resize to 70x70. Historically
    built with ReLU; **LeakyReLU is now the default** (``Cfg.activation``), and
    ``--act relu`` still reproduces the original architecture exactly.

``seisunet``
    "SeisU-Net-OpenFWI" from ``working_space/seisunet.md``: an asymmetric
    time-compression head (``k x 1`` kernels, stride ``(2, 1)``) that shrinks the
    1000-sample time axis to ~63 while leaving the 70 receivers untouched,
    followed by a U-shaped body with **concat** skips and **resize-conv**
    upsampling (bilinear + 3x3 conv, avoids checkerboarding).

Both consume ``(B, n_src=5, n_steps=1000, n_recv=70)`` features produced by
``data.py`` (``sign * log1p`` of the raw traces) and return ``(B, 70, 70)``
velocities in *normalized* target space, ``(v - mean) / std``, as expected by
``training.py`` / ``train.py``.

Deviation from the seisunet.md spec: the write-up assumes per-sample input
standardisation and a MinMax ``[-1, 1]`` target with a ``Tanh`` head. This
project feeds a fixed ``sign*log1p`` transform and z-scored targets, so
``out_activation`` defaults to ``"none"``; the ``Tanh`` variant remains available
for experiments that also change the target scaling.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Cfg

MODEL_NAMES = ("unet", "seisunet")
ACT_NAMES = ("relu", "leaky_relu")
LEAKY_SLOPE = 0.2


def make_activation(name="leaky_relu"):
    """Return a fresh activation module for ``name`` (see ``ACT_NAMES``)."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
    raise ValueError(f"Unknown activation '{name}'; expected one of {ACT_NAMES}")


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two ``conv - norm - activation`` blocks (3x3 by default)."""

    def __init__(self, in_ch, out_ch, act="leaky_relu",
                 kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            make_activation(act),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            make_activation(act),
        )

    def forward(self, x):
        return self.net(x)


class _ConvNormAct(nn.Module):
    """Single ``conv - norm - activation`` block with an explicit kernel."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 1), stride=(1, 1),
                 padding=(1, 0), act="leaky_relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            make_activation(act),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Original U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """Encode (B, 5, 1000, 70) seismic input into a (B, 70, 70) velocity map.

    ``dropout``: if > 0, apply ``Dropout2d`` on the bottleneck and each decoder
    block output (regularization against overfitting). Default 0 = unchanged.
    ``act``: hidden activation; ``"leaky_relu"`` (default) or ``"relu"`` to
    reproduce the pre-2026-09 architecture that the existing checkpoints used.
    """

    def __init__(self, in_ch=Cfg.n_src, base=Cfg.model_base_channels,
                 dropout=0.0, act=Cfg.activation, out_activation="none"):
        super().__init__()
        self.dropout = float(dropout)
        self.act = act
        self.out_activation = out_activation
        # Encoder: progressively reduce time and receiver dimensions.
        self.enc1 = DoubleConv(in_ch, base, act)      # 1000x70 -> 1000x70
        self.pool1 = nn.MaxPool2d(2, 2)               # -> 500x35
        self.enc2 = DoubleConv(base, base * 2, act)   # 500x35
        self.pool2 = nn.MaxPool2d(2, 2)               # -> 250x17 (pad to 250x18)
        self.enc3 = DoubleConv(base * 2, base * 4, act)
        self.pool3 = nn.MaxPool2d(2, 2)               # -> 125x9
        self.enc4 = DoubleConv(base * 4, base * 8, act)
        self.pool4 = nn.MaxPool2d(2, 2)               # -> 62x4 (pad to 62x5)
        self.enc5 = DoubleConv(base * 8, base * 16, act)

        # Project the bottleneck back to the first decoder resolution.
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base * 16, base * 8, kernel_size=(9, 14), stride=(2, 2)),
            nn.BatchNorm2d(base * 8),
            make_activation(act),
        )

        # Decoder blocks with skip connections.
        self.dec1 = DoubleConv(base * 16, base * 8, act)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=4, stride=2, padding=1)
        self.dec2 = DoubleConv(base * 8, base * 4, act)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1)
        self.dec3 = DoubleConv(base * 4, base * 2, act)
        self.up4 = nn.ConvTranspose2d(base * 2, base, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec4 = DoubleConv(base * 2, base, act)
        self.dec5 = DoubleConv(base, base, act)

        self.head = nn.Conv2d(base, 1, 1)
        self.out_act = nn.Tanh() if out_activation == "tanh" else None

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e2p = nn.functional.pad(e2, (0, 1))  # pad receiver dim 17 -> 18
        e3 = self.enc3(self.pool2(e2p))
        e4 = self.enc4(self.pool3(e3))
        e4p = nn.functional.pad(e4, (0, 1))  # pad receiver dim 4 -> 5
        e5 = self.enc5(self.pool4(e4p))
        if self.dropout > 0:
            e5 = nn.functional.dropout2d(e5, self.dropout, training=self.training)

        u = self.up(e5)
        # Align decoder feature maps with their skip connections.
        u = nn.functional.interpolate(u, size=(125, 9), mode="nearest")
        d1 = self.dec1(torch.cat([u, e4], dim=1))
        if self.dropout > 0:
            d1 = nn.functional.dropout2d(d1, self.dropout, training=self.training)
        d2 = self.up2(d1)
        d2 = nn.functional.interpolate(d2, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))
        if self.dropout > 0:
            d2 = nn.functional.dropout2d(d2, self.dropout, training=self.training)
        d3 = self.up3(d2)
        d3 = nn.functional.interpolate(d3, size=e2p.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2p], dim=1))
        if self.dropout > 0:
            d3 = nn.functional.dropout2d(d3, self.dropout, training=self.training)
        d4 = self.up4(d3)
        d4 = nn.functional.interpolate(d4, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e1], dim=1))
        if self.dropout > 0:
            d4 = nn.functional.dropout2d(d4, self.dropout, training=self.training)
        d5 = nn.functional.interpolate(d4, size=(Cfg.img_size, Cfg.img_size),
                                       mode="bilinear", align_corners=False)
        d5 = self.dec5(d5)
        if self.dropout > 0:
            d5 = nn.functional.dropout2d(d5, self.dropout, training=self.training)
        out = self.head(d5).squeeze(1)  # (B, 70, 70)
        return self.out_act(out) if self.out_act is not None else out


# ---------------------------------------------------------------------------
# SeisU-Net-OpenFWI
# ---------------------------------------------------------------------------
class SeisUEncoder(nn.Module):
    """Time-compression head built from asymmetric ``k x 1`` kernels.

    5 x 1000 x 70 --(7x1, s2)--> 32 x 500 x 70
                  --(3x1, s2)--> 64 x 250 x 70
                  --(3x1)     --> 64 x 250 x 70
                  --(3x1, s2)--> 128 x 125 x 70
                  --(3x1)     --> 128 x 125 x 70
                  --(3x1, s2)--> 128 x 63  x 70

    Only the time axis is touched, so the 70 receivers stay at full resolution and
    early downsampling cannot smear reflection events. The head ends at T ~= 63,
    i.e. time and space are comparable before the U-body -- the design rationale
    of seisunet.md.
    """

    def __init__(self, in_ch=Cfg.n_src, base=Cfg.model_base_channels,
                 act="leaky_relu"):
        super().__init__()
        self.head = nn.Sequential(
            _ConvNormAct(in_ch, base, (7, 1), (2, 1), (3, 0), act),
            _ConvNormAct(base, base * 2, (3, 1), (2, 1), (1, 0), act),
            _ConvNormAct(base * 2, base * 2, (3, 1), (1, 1), (1, 0), act),
            _ConvNormAct(base * 2, base * 4, (3, 1), (2, 1), (1, 0), act),
            _ConvNormAct(base * 4, base * 4, (3, 1), (1, 1), (1, 0), act),
            _ConvNormAct(base * 4, base * 4, (3, 1), (2, 1), (1, 0), act),
        )

    def forward(self, x):
        return self.head(x)


class SeisDown(nn.Module):
    """``2 x (3x3 conv - BN - act)`` followed by 2x2 max-pooling."""

    def __init__(self, in_ch, out_ch, act="leaky_relu"):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, act)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(x))


class SeisUp(nn.Module):
    """Resize-conv upsampling + concat skip + 3x3 conv (seisunet.md)."""

    def __init__(self, in_ch, out_ch, skip_ch, act="leaky_relu"):
        super().__init__()
        self.reduce = _ConvNormAct(in_ch, out_ch, (3, 3), (1, 1), (1, 1), act)
        self.fuse = _ConvNormAct(out_ch + skip_ch, out_ch, (3, 3), (1, 1), (1, 1), act)

    def forward(self, x, skip):
        # Bilinear resize (not transposed conv) to avoid checkerboard artefacts.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class SeisUNet(nn.Module):
    """Asymmetric-kernel U-Net: (B, 5, 1000, 70) -> (B, 70, 70).

    Channel plan for ``base=32`` (matches the tables in seisunet.md):

        head  32 -> 64 -> 128                         (128 x 63 x 70)
        down  128->256 -> 512 -> 512 -> 512           (512 x 4 x 5)
        up    512->256 (+512) -> 128 (+512) -> 64 (+256) -> 32 (+128)
        out   3x3 conv -> 1 channel, resized to 70x70
    """

    def __init__(self, in_ch=Cfg.n_src, base=Cfg.model_base_channels,
                 dropout=0.0, act="leaky_relu", out_activation="none"):
        super().__init__()
        self.dropout = float(dropout)
        self.act = act
        self.out_activation = out_activation

        self.enc_head = SeisUEncoder(in_ch, base, act)

        # U-shaped body (channel peak 16*base = 512 at base=32).
        self.down1 = SeisDown(base * 4, base * 8, act)
        self.down2 = SeisDown(base * 8, base * 16, act)
        self.down3 = SeisDown(base * 16, base * 16, act)
        self.down4 = SeisDown(base * 16, base * 16, act)

        self.up4 = SeisUp(base * 16, base * 8, base * 16, act)
        self.up3 = SeisUp(base * 8, base * 4, base * 16, act)
        self.up2 = SeisUp(base * 4, base * 2, base * 8, act)
        self.up1 = SeisUp(base * 2, base, base * 4, act)

        self.out = nn.Conv2d(base, 1, 3, padding=1)
        self.out_act = nn.Tanh() if out_activation == "tanh" else None

    def forward(self, x):
        # Head ends at T=63 -> pad to 64 so the four 2x2 pools halve cleanly
        # (64 -> 32 -> 16 -> 8 -> 4).
        e = F.pad(self.enc_head(x), (0, 0, 0, 1))

        d1 = self.down1(e)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        if self.dropout > 0:
            d4 = F.dropout2d(d4, self.dropout, training=self.training)

        u = self.up4(d4, d3)
        if self.dropout > 0:
            u = F.dropout2d(u, self.dropout, training=self.training)
        u = self.up3(u, d2)
        if self.dropout > 0:
            u = F.dropout2d(u, self.dropout, training=self.training)
        u = self.up2(u, d1)
        if self.dropout > 0:
            u = F.dropout2d(u, self.dropout, training=self.training)
        u = self.up1(u, e)

        u = F.interpolate(u, size=(Cfg.img_size, Cfg.img_size),
                          mode="bilinear", align_corners=False)
        out = self.out(u).squeeze(1)  # (B, 70, 70)
        return self.out_act(out) if self.out_act is not None else out


# ---------------------------------------------------------------------------
# Factory / metadata helpers
# ---------------------------------------------------------------------------
def build_model(name=None, in_ch=Cfg.n_src, base=None, dropout=0.0, act=None,
                out_activation="none"):
    """Instantiate a model by name using the shared configuration defaults."""
    name = name or Cfg.model_name
    base = Cfg.model_base_channels if base is None else base
    act = act or Cfg.activation
    if name == "unet":
        return UNet(in_ch=in_ch, base=base, dropout=dropout, act=act,
                    out_activation=out_activation)
    if name == "seisunet":
        return SeisUNet(in_ch=in_ch, base=base, dropout=dropout, act=act,
                        out_activation=out_activation)
    raise ValueError(f"Unknown model '{name}'; expected one of {MODEL_NAMES}")


def resolve_model_spec(ckpt_path):
    """Infer ``(model_name, act, out_activation)`` from a checkpoint's run dir.

    ``train.py`` records ``model_type`` / ``act`` / ``out_activation`` in the run
    metadata. Runs saved before those fields existed used the original ReLU
    U-Net, so they fall back to ``("unet", "relu", "none")`` -- which is what
    keeps the pre-2026-09 checkpoints loadable with the architecture they were
    trained with.
    """
    run_dir = Path(ckpt_path).resolve().parent
    for meta_name in ("results.json", "config.json"):
        meta_path = run_dir / meta_name
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        argv = meta.get("arguments", {}) if isinstance(meta, dict) else {}
        model_name = meta.get("model_type") or argv.get("model")
        act = meta.get("act") or argv.get("act")
        out_activation = meta.get("out_activation") or argv.get("out_activation")
        if model_name or act or out_activation:
            return (model_name or "unet", act or "relu", out_activation or "none")
    return ("unet", "relu", "none")
