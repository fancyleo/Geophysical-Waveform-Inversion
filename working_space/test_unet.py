"""Minimal U-Net shape and gradient smoke test.

Run from the repository root with:
    python working_space/test_unet.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import UNet
from config import Cfg


def main():
    model = UNet(in_ch=Cfg.n_src, base=Cfg.test_base_channels)
    inputs = torch.randn(
        Cfg.test_batch_size, Cfg.n_src, Cfg.n_steps, Cfg.n_recv
    )
    outputs = model(inputs)

    expected_shape = (Cfg.test_batch_size, Cfg.img_size, Cfg.img_size)
    assert outputs.shape == expected_shape, (
        f"unexpected output shape: {tuple(outputs.shape)}"
    )

    loss = outputs.abs().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    print(f"forward output: {tuple(outputs.shape)}")
    print("backward pass: ok")


if __name__ == "__main__":
    main()