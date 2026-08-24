"""Data loading utilities: file pairing, dataset, and flat sample indices."""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Cfg


def find_fault_pairs(family_dir):
    """Find recursive same-directory pairs such as ``seis2.npy`` and ``vel2.npy``."""
    pairs = []
    pattern = os.path.join(family_dir, "**", "seis*.npy")
    for seismic_path in sorted(glob.glob(pattern, recursive=True)):
        filename = os.path.basename(seismic_path)
        velocity_path = os.path.join(
            os.path.dirname(seismic_path), "vel" + filename[4:]
        )
        if os.path.isfile(velocity_path):
            pairs.append((seismic_path, velocity_path))
    return pairs


def find_pairs(root, families=None):
    """Find paired seismic and velocity files for the selected families."""
    families = Cfg.families if families is None else families
    pairs = []
    for family in families:
        family_dir = os.path.join(root, family)
        if not os.path.isdir(family_dir):
            print(f"[warn] missing family dir: {family_dir}")
            continue

        # Velocity and style families store data and models in separate folders.
        data_dir = os.path.join(family_dir, "data")
        model_dir = os.path.join(family_dir, "model")
        if os.path.isdir(data_dir) and os.path.isdir(model_dir):
            for seismic_path in sorted(glob.glob(os.path.join(data_dir, "*.npy"))):
                base = os.path.basename(seismic_path)
                model_path = os.path.join(model_dir, base.replace("data", "model"))
                if os.path.exists(model_path):
                    pairs.append((seismic_path, model_path))
            continue

        # Fault families keep matching seis/vel files in the same directory tree.
        pairs.extend(find_fault_pairs(family_dir))

    print(f"[info] total paired files: {len(pairs)}")
    return pairs


def build_flat_indices(pairs):
    """Build flattened ``(file_index, sample_index)`` pairs for all files."""
    indices = []
    for file_index, (seismic_path, _) in enumerate(pairs):
        sample_count = np.load(seismic_path, mmap_mode="r").shape[0]
        indices.extend((file_index, sample_index) for sample_index in range(sample_count))
    return indices


class SeisVelDataset(Dataset):
    """Lazily load individual seismic and velocity samples from NumPy files."""

    def __init__(self, pairs, idx_list, vel_mean=Cfg.vel_mean, vel_std=Cfg.vel_std):
        self.pairs = pairs
        self.idx_list = idx_list
        self.vel_mean = vel_mean
        self.vel_std = vel_std
        # Cache opened memory-mapped arrays within each worker process.
        self._seis_cache = {}
        self._vel_cache = {}

    def __len__(self):
        return len(self.idx_list)

    def _open(self, file_index):
        """Open and cache the seismic and velocity arrays for one file pair."""
        if file_index not in self._seis_cache:
            seismic_path, velocity_path = self.pairs[file_index]
            self._seis_cache[file_index] = np.load(seismic_path, mmap_mode="r")
            self._vel_cache[file_index] = np.load(velocity_path, mmap_mode="r")
        return self._seis_cache[file_index], self._vel_cache[file_index]

    def __getitem__(self, index):
        """Load, preprocess, and return one seismic/velocity sample pair."""
        file_index, sample_index = self.idx_list[index]
        seismic_arr, velocity_arr = self._open(file_index)

        # Indexing a memmap yields a read-only view; copy into a writable buffer
        # so the in-place normalization below is valid.
        seismic = np.array(seismic_arr[sample_index], dtype=np.float32, copy=True)
        velocity = np.array(velocity_arr[sample_index], dtype=np.float32, copy=True)

        velocity -= self.vel_mean
        velocity /= self.vel_std

        # Compress the seismic dynamic range with in-place ops to limit temporaries.
        np.abs(seismic, out=seismic)
        np.log1p(seismic, out=seismic)

        seismic = seismic.reshape(Cfg.n_src, Cfg.n_steps, Cfg.n_recv)
        return torch.from_numpy(seismic), torch.from_numpy(velocity)
