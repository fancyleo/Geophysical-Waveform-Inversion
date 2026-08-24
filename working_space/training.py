"""Training and validation loops plus model parallelization helpers."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm


def is_distributed():
    """Return whether a distributed process group is active."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Return the current process rank (0 when not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    """Return the process-group size (1 when not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    """Return whether the current process should write artifacts."""
    return get_rank() == 0


def setup_ddp(local_rank, world_size):
    """Initialize the NCCL process group and pin this process to its GPU."""
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=local_rank,
        world_size=world_size,
    )
    return torch.device(f"cuda:{local_rank}")


def cleanup_ddp():
    """Destroy the distributed process group if it was initialized."""
    if is_distributed():
        dist.destroy_process_group()


def resolve_parallel_mode(mode, device, gpu_count, world_size=1):
    """Resolve the requested parallel mode against the available hardware.

    ``single`` is never changed. ``data_parallel`` and ``ddp`` both run through
    DistributedDataParallel when multiple processes are used; they degrade to
    ``single`` on a single device.
    """
    if mode == "single":
        return "single"
    if mode in {"data_parallel", "ddp"}:
        return "ddp" if (world_size > 1 and gpu_count > 0) else "single"
    raise ValueError(f"Unsupported parallel mode: {mode}")


def wrap_model_for_parallel(model, mode, device, world_size=1):
    """Wrap a model for the selected parallel backend."""
    resolved_mode = resolve_parallel_mode(
        mode, str(device), torch.cuda.device_count(), world_size
    )
    if resolved_mode == "ddp":
        if is_main_process():
            print(f"[info] using DistributedDataParallel, world_size={get_world_size()}")
        device_index = device.index if device.type == "cuda" else None
        kwargs = {"device_ids": [device_index]} if device_index is not None else {}
        return DistributedDataParallel(model, **kwargs), resolved_mode
    return model, resolved_mode


def unwrap_model(model):
    """Return the underlying model from a parallel wrapper."""
    if isinstance(model, (nn.DataParallel, DistributedDataParallel)):
        return model.module
    return model


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
