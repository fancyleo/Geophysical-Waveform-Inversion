# Aliyun Kaggle Resume-Training Workflow

> **Goal**: After Kaggle's free GPU quota runs out, keep developing Kaggle competition models by using **CPU machine to download data → migrate to an ESSD cloud disk → train on the GPU machine**.
> **Supported platforms**: Aliyun (ECS + ESSD cloud disks + container services all work)

---

## 1. Overall Architecture

```
┌────────────────────────┐   detach    ┌────────────────────┐   attach    ┌──────────────────────────┐
│CPU Machine (ECS Shared)│ ──────────▶│Aliyun ESSD Disk    │ ──────────▶│GPU Machine (ECS gn7i)    │
│downloader container    │ write+detach│/dev/sdb → /data    │ mount+train │gpu-train container       │
│- kaggle API download   │             │├─ input (ro)       │             │- JupyterLab :8888        │
│- preprocess            │             │└─ working (rw)     │             │- /kaggle/input, /working │
└────────────────────────┘             └────────────────────┘             └──────────────────────────┘
```

**Core idea**: The CPU container does the "heavy lifting" (download/extract/preprocess), while the GPU container only does the "compute" (training). Shared storage uses an **Aliyun ESSD block device** (mounted to `/data` on the host, and to `/kaggle/input` (read-only) and `/kaggle/working` (read-write) inside containers). Once the CPU finishes downloading, it detaches the disk → the GPU machine attaches it, perfectly replicating the Kaggle Notebook directory conventions.

---

## 2. Aliyun Resource Preparation

| Resource | Recommended Spec | Purpose | Cost Reference |
|------|---------|------|---------|
| ECS (CPU downloader) | ecs.g7.large / shared s6 | Runs the downloader container | Pay-as-you-go, ~0.5 CNY/hour |
| ECS (GPU trainer) | ecs.gn7i-c8g1.2xlarge (shared A100/80G) or Lingjun cluster | Runs the gpu-train container | Spot instances can save up to 70% |
| ESSD disk | Capacity ≥ 200G (100G data + headroom), performance 2000 IOPS+ | Shared data (migrated between machines via detach/attach) | Billed by capacity/performance, cheaper than NAS |
| VPC (Virtual Private Cloud) | Default | Connects CPU/GPU | Free |
| Security group | Open port 8888 (Jupyter) | Remote access | Free |

> 💡 **Cost-saving tips**:
> - Downloading is a one-time task, so use **pay-as-you-go** on the CPU machine and delete it when done
> - Use **spot instances** on the GPU machine, priced as low as 10% of on-demand
> - Buy only one ESSD disk and **migrate it** between the CPU/GPU machines; no need to keep it attached to the GPU machine long-term (start/stop the GPU machine on demand)

---

## 3. Directory Structure

```
.
├── docker-compose.yml        # Orchestration: downloader + gpu-train
├── pyproject.toml            # Python dependencies (core 6 + torch)
├── setup.sh                  # One-click deployment script
├── validate.py               # Config validation script
├── .env.example               # Environment variable template (copy to .env yourself)
│
├── downloader/
│   └── Dockerfile            # CPU download image
├── gpu/
│   ├── Dockerfile            # GPU training image (Kaggle style)
│   ├── requirements-gpu.txt  # Locked GPU dependencies (torch reuses base image cu121)
│   └── entrypoint.sh         # Entrypoint: wait for data-ready marker, then start JupyterLab
│
├── scripts/
│   ├── download.sh           # Download script (resume + extract + completion marker)
│   └── preprocess.py         # Preprocessing script (manifest/parquet/split)
│
├── kaggle-api/
│   └── kaggle.json           # ⚠️ Put your Kaggle API Token here
│
└── host /data                # ⚠️ ESSD block device /dev/sdb mounted to /data
    └── kaggle/
        ├── input/            # Dataset (written by CPU, read-only for GPU)
        └── working/          # Preprocessing artifacts + training output (both directions)
        # Container paths: /kaggle/input (ro) and /kaggle/working (rw),
        # determined by ${DATA_MOUNT:-/data} in compose (default /data)
```

---

## 4. Quick Start

### Step 1: Prepare the environment

```bash
# 1. Clone/upload this directory to the CPU machine
cd /path/to/this-dir

# 2. Place your Kaggle API Token
#    Go to https://www.kaggle.com/settings -> Create New Token to download kaggle.json
cp ~/Downloads/kaggle.json ./kaggle-api/kaggle.json
chmod 600 ./kaggle-api/kaggle.json

# 3. Configure environment variables
cp .env.example .env
# Edit .env to fill in the competition name, ESSD mount point (DATA_MOUNT, default /data), etc.

# 4. Validate all configuration in one go
python validate.py
```

### Step 2: CPU machine — download + preprocess data to ESSD

```bash
# 0. First confirm the ESSD is mounted to /data
df -h | grep /data    # /dev/sdb → /data

# Method A: use docker-compose (recommended)
docker compose --profile downloader up --build

# Method B: run the container directly (equivalent to above)
docker compose run --rm downloader

# If preprocessing is needed (generate parquet / split)
docker compose run --rm downloader python /scripts/preprocess.py
```

When done, the `/data/kaggle/working/.download_complete` marker file will appear on the ESSD, indicating the data is ready.

### Step 3: Migrate the ESSD to the GPU machine and start training

```bash
# 1. Unmount and detach the ESSD on the CPU machine (console/CLI both work)
sudo umount /data
# Console: Disk Management → unmount /dev/sdb

# 2. Attach the ESSD to the GPU machine and mount it to /data
sudo mkfs.ext4 /dev/sdb            # Format only the first time (this wipes the disk — be careful!)
sudo mkdir -p /data && sudo mount /dev/sdb /data
df -h | grep /data                 # Confirm the mount succeeded

# 3. Start the GPU training container
docker compose --profile gpu up --build -d

# 4. View logs / get the Jupyter token
docker compose logs -f gpu-train
# Open http://<GPU-machine-public-IP>:8888 in a browser and paste the token
```

### Step 4: Access data in training code (fully Kaggle style)

```python
import pandas as pd
import torch

# Data paths are identical to the Kaggle Notebook
train = pd.read_csv("/kaggle/input/train.csv")
model = torch.load("/kaggle/input/pretrained.pth")

# Output to working; it lands back on the ESSD automatically
torch.save(model.state_dict(), "/kaggle/working/checkpoint.pth")
```

---

## 5. Key Design Notes

### 1. Why is input mounted read-only (`:ro`)
Prevents the training container from accidentally modifying the original dataset, ensuring data consistency; if changes are needed, re-preprocess on the CPU side.

### 2. Resume downloads
`download.sh` uses `kaggle ... --resume`, so a 100G download can be re-run after an interruption without re-downloading what's already fetched.

### 3. Completion marker
`working/.download_complete` is an empty file + timestamp; the GPU side can poll it to check whether the data is ready before starting training.

### 4. GPU image selection
`gcr.io/kaggle-gpu-images/python:latest` ships with CUDA + cuDNN + cu121 torch + JupyterLab, closely matching the Kaggle Notebook environment; if your Aliyun GPU machine driver is ≥ 525, you can use it directly.
- **Don't reinstall torch**: reuse the cu121 torch that ships with the base image (`requirements-gpu.txt` doesn't include torch) to avoid version drift from reinstalling.
- Training code is copied to `/kaggle/code` at build time (`PYTHONPATH=/kaggle/code`), and data paths hit the ESSD directly via the environment variable `WAVEFORM_DATA_ROOT=/kaggle/input`.
- Before starting, the container waits for the `.download_complete` marker written by the downloader (`entrypoint.sh`); it blocks until the data is ready, preventing the training script from running against empty data.

### 5. shm_size
`--shm_size=16g` resolves the common out-of-shared-memory error in PyTorch DataLoader multi-process mode.

---

## 6. FAQ / Pitfalls

| Problem | Cause | Solution |
|------|------|---------|
| Container can't read data after ESSD migration | Forgot to remount /data | Confirm `df -h` on the GPU machine after `mount /dev/sdb /data`; compose uses `${DATA_MOUNT:-/data}` |
| ESSD IO is slow | Single-disk performance is limited | Copy `input/` to the GPU machine's local NVMe before training, then write back checkpoints when done |
| GPU container `torch.cuda.is_available()=False` | NVIDIA Container Toolkit not installed | Install `nvidia-container-toolkit` on the GPU machine and use `--gpus all` |
| 100G download timeout | Kaggle API rate limiting | `download.sh` already uses `--resume`; just re-run to resume |
| Permission denied | UID mismatch | This workflow uniformly uses `chmod 777 /data/kaggle` + the kaggle user, so it usually needs no action |
| Repeated downloads | Container restart | Dual protection via the `.download_complete` marker + `--resume` |
| GPU spot instance reclaimed | Normal behavior | Save checkpoints to `/kaggle/working` (ESSD) periodically, then resume from the latest ckpt after restart |

---

## 7. Advanced: Migrating to Kubernetes / ACK (large-scale training)

If the data volume or training scale grows, you can deploy `gpu-train` to **Aliyun Container Service ACK**:
- Mount the ESSD disk as a PV/PVC (or upgrade to shared storage)
- Use `resources.limits.nvidia.com/gpu: 1` for GPU
- Manage training jobs with Job / TFJob

The service definitions in `docker-compose.yml` can be migrated directly to Kubernetes Deployments.

---

## 8. One-click cleanup

```bash
# Stop the containers
docker compose down

# Unmount the ESSD (on both CPU and GPU machines)
sudo umount /data

# Detach the ESSD / delete the disk (via console; be careful to avoid deleting data!)
```

---

**Summary**: The CPU moves the data, the ESSD stores it, and the GPU trains the model — the disk is hot-plugged between the two machines, all three are decoupled, and you start/stop on demand for the lowest cost. After Kaggle's free GPU is exhausted, this is the closest alternative to the native Kaggle experience.
