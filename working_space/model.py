"""Models mapping five-source seismic data to a 70 x 70 velocity map.

Three architectures are available (select with ``--model`` / ``Cfg.model_name``):

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

``caformer``
    Port of @brendanartley's community solution: an **ImageNet-22k pretrained**
    ``caformer_b36`` timm backbone over the full-resolution input, plus a MONAI
    decoder using PixelShuffle upsampling, SCSE attention and intermediate convs.
    123.67 M parameters. This is the only model here that is *pretrained*, which
    is a different lever from the from-scratch width experiments: those showed
    capacity is not this dataset's binding constraint, but transfer learning is
    an axis they never tested. It is also the most expensive option (~48 min per
    epoch at batch 24 on one RTX 4090, versus 12.5 min for ``seisunet``).

All three consume ``(B, n_src=5, n_steps=1000, n_recv=70)`` features produced by
``data.py`` (``sign * log1p`` of the raw traces) and return ``(B, 70, 70)``
velocities in *normalized* target space, ``(v - mean) / std``, as expected by
``training.py`` / ``train.py``.

Deviation from the seisunet.md spec: the write-up assumes per-sample input
standardisation and a MinMax ``[-1, 1]`` target with a ``Tanh`` head. This
project feeds a fixed ``sign*log1p`` transform and z-scored targets, so
``out_activation`` defaults to ``"none"``; the ``Tanh`` variant remains available
for experiments that also change the target scaling. ``CAFormer`` follows the
same rule -- see its docstring for the full list of deliberate deviations.

``timm`` / ``monai`` are optional and only imported when ``caformer`` is built, so
``unet`` and ``seisunet`` keep working without them.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Cfg

MODEL_NAMES = ("unet", "seisunet", "caformer")
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
# CAFormer (timm encoder + PixelShuffle/SCSE decoder)
# ---------------------------------------------------------------------------
# Port of @brendanartley's "CAFormer Full Resolution Improved" notebook
# (community_solution/@brendanartley-caformer-full-resolution-improved.ipynb).
# ``timm`` and ``monai`` are imported lazily inside CAFormer so that ``unet`` and
# ``seisunet`` keep working in environments where those packages, or the
# ImageNet weights, are unavailable.
#
# Measured for the default backbone and this project's (B, 5, 1000, 70) input:
#   backbone   72x72x128 -> 36x36x256 -> 18x18x512 -> 9x9x768
#   decoder    768@9 -> 128@18 -> 64@36 -> 32@72      (four 2x pixel-shuffles)
#   head       3x3 conv -> 1 channel, centre-cropped 72x72 -> 70x70
#   parameters 123.67 M (93.32 M of which is the pretrained backbone)
CAFORMER_DEFAULT_BACKBONE = "caformer_b36.sail_in22k_ft_in1k"


class CAConvBnAct2d(nn.Module):
    """``conv(bias=False)`` -> optional norm -> activation (original ConvBnAct2d).

    ``norm_layer=None`` (the original's ``nn.Identity``) means no normalisation at
    all in the decoder, which is how the notebook trains it.
    """

    def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1,
                 norm_layer=None, act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.norm = norm_layer(out_ch) if norm_layer is not None else nn.Identity()
        self.act = make_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CASCSEModule2d(nn.Module):
    """Spatial + channel squeeze-and-excitation: ``x*cSE(x) + x*sSE(x)``.

    Keeps the original's ``in_channels`` argument name: ``CAAttention2d`` forwards
    its keyword arguments straight through.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)   # guard: original divides to 0 below 16
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1),
            nn.Tanh(),
            nn.Conv2d(hidden, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class CAAttention2d(nn.Module):
    """``None`` -> identity; ``"scse"`` -> :class:`CASCSEModule2d`."""

    def __init__(self, name=None, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity()
        elif name == "scse":
            self.attention = CASCSEModule2d(**params)
        else:
            raise ValueError(f"Unknown attention '{name}'; expected None or 'scse'")

    def forward(self, x):
        return self.attention(x)


class CADecoderBlock2d(nn.Module):
    """upsample -> [intermediate conv on the skip] -> concat -> attn1 -> 2x conv -> attn2."""

    def __init__(self, in_ch, skip_ch, out_ch, act="relu", attention_type=None,
                 intermediate_conv=False, upsample_mode="deconv", scale_factor=2):
        super().__init__()
        # Lazily-imported MONAI upsamplers.
        from monai.networks.blocks import SubpixelUpsample, UpSample

        if upsample_mode == "pixelshuffle":
            self.upsample = SubpixelUpsample(spatial_dims=2, in_channels=in_ch,
                                             scale_factor=scale_factor)
        else:
            self.upsample = UpSample(spatial_dims=2, in_channels=in_ch,
                                     out_channels=in_ch, scale_factor=scale_factor,
                                     mode=upsample_mode)

        if intermediate_conv:
            k = 3
            target = skip_ch if skip_ch != 0 else in_ch
            self.intermediate_conv = nn.Sequential(
                CAConvBnAct2d(target, target, k, k // 2, act=act),
                CAConvBnAct2d(target, target, k, k // 2, act=act),
            )
        else:
            self.intermediate_conv = None

        self.attention1 = CAAttention2d(name=attention_type,
                                        in_channels=in_ch + skip_ch)
        self.conv1 = CAConvBnAct2d(in_ch + skip_ch, out_ch, kernel_size=3,
                                   padding=1, act=act)
        self.conv2 = CAConvBnAct2d(out_ch, out_ch, kernel_size=3, padding=1, act=act)
        self.attention2 = CAAttention2d(name=attention_type, in_channels=out_ch)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if self.intermediate_conv is not None:
            if skip is not None:
                skip = self.intermediate_conv(skip)
            else:
                x = self.intermediate_conv(x)
        if skip is not None:
            x = self.attention1(torch.cat([x, skip], dim=1))
        x = self.conv2(self.conv1(x))
        return self.attention2(x)


class CAUnetDecoder2d(nn.Module):
    """MONAI-style U-Net decoder (source: https://arxiv.org/abs/1505.04597).

    ``encoder_channels`` must be ordered **deepest first** (the order the reversed
    backbone feature list arrives in). A 4-level encoder automatically drops the
    first entry of ``decoder_channels``, which is what makes the (256,128,64,32)
    default collapse to (128,64,32) for caformer_b36.
    """

    def __init__(self, encoder_channels, skip_channels=None,
                 decoder_channels=(256, 128, 64, 32), scale_factors=(2, 2, 2, 2),
                 act="relu", attention_type="scse", intermediate_conv=True,
                 upsample_mode="pixelshuffle"):
        super().__init__()
        if len(encoder_channels) == 4:
            decoder_channels = tuple(decoder_channels[1:])
        self.decoder_channels = decoder_channels
        if skip_channels is None:
            skip_channels = list(encoder_channels[1:]) + [0]

        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList([
            CADecoderBlock2d(ic, sc, dc, act=act, attention_type=attention_type,
                             intermediate_conv=intermediate_conv,
                             upsample_mode=upsample_mode,
                             scale_factor=scale_factors[i])
            for i, (ic, sc, dc) in enumerate(
                zip(in_channels, skip_channels, decoder_channels))
        ])

    def forward(self, feats):
        res = [feats[0]]
        feats = feats[1:]
        for i, block in enumerate(self.blocks):
            skip = feats[i] if i < len(feats) else None
            res.append(block(res[-1], skip=skip))
        return res


class CASegmentationHead2d(nn.Module):
    """3x3 conv to ``out_channels`` followed by a non-trainable upsample."""

    def __init__(self, in_channels, out_channels, scale_factor=1,
                 kernel_size=3, mode="nontrainable"):
        super().__init__()
        from monai.networks.blocks import UpSample

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.upsample = UpSample(spatial_dims=2, in_channels=out_channels,
                                 out_channels=out_channels,
                                 scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.upsample(self.conv(x))


class CAFormer(nn.Module):
    """Pretrained timm backbone + PixelShuffle/SCSE decoder: (B,5,1000,70) -> (B,70,70).

    **Deviations from the original notebook** (all deliberate, for integration):

    1. ``forward`` returns values in this project's **normalized** target space
       ``(v - mean) / std``. The original denormalises inside ``forward``
       (``x_seg * 1500 + 3000``) because it trains on a ``MinMax`` target; here
       denormalisation stays where the rest of the project does it
       (``training.py`` / ``infer.py`` / ``eval_holdout.py``).
    2. ``forward`` is **deterministic**. The original averages a flip view when
       ``not self.training``; here that exact operation is available at inference
       time through ``tta.py --tta src_recv`` (its ``dims=(-3,-1)`` mirror the
       original ``torch.flip(dims=[-3,-1])``), so training and evaluation stay
       comparable across architectures.
    3. ``act`` selects the decoder activation and defaults to
       ``Cfg.activation`` (LeakyReLU). Pass ``"relu"`` to reproduce the original
       exactly.
    4. ``base`` is accepted but **unused**: the width is fixed by the pretrained
       backbone, not by ``--base_channels``. It is kept only so ``build_model``
       and ``resolve_model_spec`` have a uniform signature.
    5. ``dropout > 0`` is routed to the backbone's stochastic-depth rate
       (``drop_path_rate``), the natural regulariser for a pretrained hybrid
       transformer. The original uses ``drop_path_rate=0.0``.

    Only :data:`CAFORMER_DEFAULT_BACKBONE` is wired for checkpoint reload: the
    backbone name is not recorded in ``results.json`` and there is no CLI flag for
    it, so a different backbone cannot be rebuilt automatically.
    """

    def __init__(self, in_ch=Cfg.n_src, base=Cfg.model_base_channels, dropout=0.0,
                 act=None, out_activation="none",
                 backbone=CAFORMER_DEFAULT_BACKBONE, pretrained=True,
                 drop_path_rate=None, decoder_channels=(256, 128, 64, 32),
                 scale_factors=(2, 2, 2, 2), attention_type="scse",
                 intermediate_conv=True, upsample_mode="pixelshuffle",
                 img_size=None):
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # keep unet/seisunet usable without timm
            raise ImportError(
                "CAFormer needs `timm` (and `monai` for its decoder). Install with "
                "`pip install timm monai`, then retry."
            ) from exc

        act = act or Cfg.activation
        self.act = act
        self.out_activation = out_activation
        self.img_size = int(img_size or Cfg.img_size)
        self.backbone_name = backbone

        if drop_path_rate is None:
            drop_path_rate = float(dropout)   # --dropout doubles as stochastic depth

        try:
            self.backbone = timm.create_model(
                backbone, in_chans=in_ch, pretrained=pretrained,
                features_only=True, drop_path_rate=float(drop_path_rate),
            )
        except Exception as exc:   # offline / no ImageNet weights available
            raise RuntimeError(
                f"Could not build timm backbone '{backbone}' with "
                f"pretrained={pretrained}. If this is a network error, set an "
                f"accessible mirror, e.g. HF_ENDPOINT=https://hf-mirror.com."
            ) from exc

        if not hasattr(self.backbone, "stem") or not hasattr(self.backbone, "stages_0"):
            raise ValueError(
                f"Backbone '{backbone}' has no .stem/.stages_0 to adapt; this port "
                f"targets the caformer_* family."
            )

        # Deepest stage first, matching the reversed feature list fed to the decoder.
        encoder_channels = [f["num_chs"] for f in self.backbone.feature_info][::-1]
        self.decoder = CAUnetDecoder2d(
            encoder_channels=encoder_channels, decoder_channels=decoder_channels,
            scale_factors=scale_factors, act=act, attention_type=attention_type,
            intermediate_conv=intermediate_conv, upsample_mode=upsample_mode,
        )
        self.seg_head = CASegmentationHead2d(
            in_channels=self.decoder.decoder_channels[-1], out_channels=1,
            scale_factor=1,
        )
        self.out_act = nn.Tanh() if out_activation == "tanh" else None
        self._adapt_stem()

    def _adapt_stem(self):
        """Squeeze the TIME axis in the stem (4x) and again with the stage-0 pool.

        Keeps the 70 receivers at full resolution while the 1000-sample time axis
        shrinks to 72, so the decoder never has to up-sample a degenerate map.
        The 78-pixel reflection pad is the original's constant and is tuned for
        ``n_steps=1000`` (1000 + 2*78 = 1156 -> 72 after both 4x reductions).
        """
        stem_conv = self.backbone.stem.conv
        stem_conv.stride = (4, 1)
        stem_conv.padding = (0, 4)
        self.backbone.stages_0.downsample = nn.AvgPool2d(kernel_size=(4, 1),
                                                        stride=(4, 1))
        self.backbone.stem = nn.Sequential(
            nn.ReflectionPad2d((0, 0, 78, 78)), self.backbone.stem
        )

    def _centre_crop(self, x):
        """Centre-crop the decoded map to ``img_size`` (72x72 -> 70x70).

        The original hardcodes ``[..., 1:-1, 1:-1]``; deriving the offsets from the
        real tensor keeps it correct if the map is ever an odd number of pixels
        larger and turns a silent shape bug into an explicit error.
        """
        h, w = x.shape[-2:]
        size = self.img_size
        if (h, w) == (size, size):
            return x
        if h < size or w < size:
            raise ValueError(
                f"CAFormer produced a {h}x{w} map, smaller than the {size}x{size} "
                f"target: the stem padding is tuned for {Cfg.n_steps} time samples."
            )
        top, left = (h - size) // 2, (w - size) // 2
        return x[..., top:top + size, left:left + size]

    def forward(self, x):
        feats = self.backbone(x)            # shallowest -> deepest
        decoded = self.decoder(feats[::-1])  # deepest -> shallowest
        out = self.seg_head(decoded[-1])
        if self.out_act is not None:
            out = self.out_act(out)
        return self._centre_crop(out).squeeze(1)   # (B, 70, 70)


# ---------------------------------------------------------------------------
# Factory / metadata helpers
# ---------------------------------------------------------------------------
def build_model(name=None, in_ch=Cfg.n_src, base=None, dropout=0.0, act=None,
                out_activation="none", pretrained=None):
    """Instantiate a model by name using the shared configuration defaults.

    ``pretrained`` only affects ``caformer``: leave it ``None`` (= pretrained) when
    starting a training run, and pass ``False`` when rebuilding a model to load a
    checkpoint -- the checkpoint already carries every weight, so downloading
    ImageNet weights first is wasted work and fails outright on an offline
    machine.
    """
    name = name or Cfg.model_name
    base = Cfg.model_base_channels if base is None else base
    act = act or Cfg.activation
    if name == "unet":
        return UNet(in_ch=in_ch, base=base, dropout=dropout, act=act,
                    out_activation=out_activation)
    if name == "seisunet":
        return SeisUNet(in_ch=in_ch, base=base, dropout=dropout, act=act,
                        out_activation=out_activation)
    if name == "caformer":
        # `base` is unused here: the width comes from the pretrained backbone.
        extra = {} if pretrained is None else {"pretrained": pretrained}
        return CAFormer(in_ch=in_ch, base=base, dropout=dropout, act=act,
                        out_activation=out_activation, **extra)
    raise ValueError(f"Unknown model '{name}'; expected one of {MODEL_NAMES}")


def resolve_model_spec(ckpt_path):
    """Infer ``(model_name, act, out_activation, base)`` from a checkpoint's run dir.

    ``train.py`` records ``model_type`` / ``act`` / ``out_activation`` /
    ``model_base_channels`` in the run metadata. Runs saved before those fields
    existed used the original ReLU U-Net at the default width, so they fall back
    to ``("unet", "relu", "none", Cfg.model_base_channels)`` -- which is what
    keeps the pre-2026-09 checkpoints loadable with the architecture they were
    trained with. Returns a 4-tuple; callers that only need the first three can
    unpack ``spec[:3]``.
    """
    default_base = Cfg.model_base_channels
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
        base = meta.get("model_base_channels") or argv.get("base_channels")
        base = int(base) if base else default_base
        if model_name or act or out_activation:
            return (model_name or "unet", act or "relu",
                    out_activation or "none", base)
    return ("unet", "relu", "none", default_base)
