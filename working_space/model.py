"""U-Net model mapping five-source seismic data to a 70 x 70 velocity map."""

import torch
import torch.nn as nn

from config import Cfg


class DoubleConv(nn.Module):
    """Two convolution, batch-normalization, and ReLU blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Encode (B, 5, 1000, 70) seismic input into a (B, 70, 70) velocity map."""

    def __init__(self, in_ch=Cfg.n_src, base=Cfg.model_base_channels):
        super().__init__()
        # Encoder: progressively reduce time and receiver dimensions.
        self.enc1 = DoubleConv(in_ch, base)      # 1000x70 -> 1000x70
        self.pool1 = nn.MaxPool2d(2, 2)          # -> 500x35
        self.enc2 = DoubleConv(base, base * 2)   # 500x35
        self.pool2 = nn.MaxPool2d(2, 2)          # -> 250x17 (pad to 250x18)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2, 2)          # -> 125x9
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2, 2)          # -> 62x4 (pad to 62x5)
        self.enc5 = DoubleConv(base * 8, base * 16)

        # Project the bottleneck back to the first decoder resolution.
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base * 16, base * 8, kernel_size=(9, 14), stride=(2, 2)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks with skip connections.
        self.dec1 = DoubleConv(base * 16, base * 8)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=4, stride=2, padding=1)
        self.dec2 = DoubleConv(base * 8, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1)
        self.dec3 = DoubleConv(base * 4, base * 2)
        self.up4 = nn.ConvTranspose2d(base * 2, base, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec4 = DoubleConv(base * 2, base)
        self.dec5 = DoubleConv(base, base)

        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e2p = nn.functional.pad(e2, (0, 1))  # pad receiver dim 17 -> 18
        e3 = self.enc3(self.pool2(e2p))
        e4 = self.enc4(self.pool3(e3))
        e4p = nn.functional.pad(e4, (0, 1))  # pad receiver dim 4 -> 5
        e5 = self.enc5(self.pool4(e4p))

        u = self.up(e5)
        # Align decoder feature maps with their skip connections.
        u = nn.functional.interpolate(u, size=(125, 9), mode="nearest")
        d1 = self.dec1(torch.cat([u, e4], dim=1))
        d2 = self.up2(d1)
        d2 = nn.functional.interpolate(d2, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))
        d3 = self.up3(d2)
        d3 = nn.functional.interpolate(d3, size=e2p.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2p], dim=1))
        d4 = self.up4(d3)
        d4 = nn.functional.interpolate(d4, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e1], dim=1))
        d5 = nn.functional.interpolate(d4, size=(Cfg.img_size, Cfg.img_size),
                                       mode="bilinear", align_corners=False)
        d5 = self.dec5(d5)
        return self.head(d5).squeeze(1)  # (B, 70, 70)
