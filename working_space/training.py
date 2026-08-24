"""Training and validation loops plus model parallelization helpers."""

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def resolve_parallel_mode(mode, device, gpu_count):
    """Resolve the requested parallel mode against the available hardware."""
    if mode not in {"single", "data_parallel", "ddp"}:
        raise ValueError(f"Unsupported parallel mode: {mode}")
    if mode == "ddp":
        raise NotImplementedError(
            "DDP is reserved for the distributed training entry point. "
            "Use data_parallel for the current notebook workflow."
        )
    if mode == "data_parallel" and device.startswith("cuda") and gpu_count > 1:
        return "data_parallel"
    return "single"


def wrap_model_for_parallel(model, mode, device):
    """Wrap a model for the selected parallel backend."""
    resolved_mode = resolve_parallel_mode(mode, device, torch.cuda.device_count())
    if resolved_mode == "data_parallel":
        print(f"[info] using DataParallel on {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model), resolved_mode
    return model, resolved_mode


def unwrap_model(model):
    """Return the underlying model from a parallel wrapper."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one epoch; optimize when ``optimizer`` is given, otherwise validate."""
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total, count = 0.0, 0
    progress = tqdm(loader, desc="train" if optimizer else "valid", leave=False)
    for seismic, velocity in progress:
        seismic, velocity = seismic.to(device), velocity.to(device)
        velocity = velocity.squeeze(1) if velocity.dim() == 4 else velocity

        prediction = model(seismic)
        loss = criterion(prediction, velocity)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = seismic.size(0)
        total += loss.item() * batch_size
        count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

        # Release intermediates so DataParallel gather buffers do not accumulate.
        del seismic, velocity, prediction, loss
    return total / max(count, 1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch and return the sample-weighted mean loss."""
    return _run_epoch(model, loader, criterion, device, optimizer=optimizer)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate the model and return the sample-weighted mean validation loss."""
    return _run_epoch(model, loader, criterion, device, optimizer=None)
