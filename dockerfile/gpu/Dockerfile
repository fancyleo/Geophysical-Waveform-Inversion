# ============================================================
# GPU training image (Kaggle style) — improved version
# Base: official Kaggle GPU image (Ubuntu + CUDA + cu121 torch + JupyterLab)
# Tailored for Aliyun GPU instances (NVIDIA Container Runtime)
#
# Key fixes vs the old version (following the downloader template's
# "minimal and reliable" principle):
#   1. No longer uninstall torch: the base image ships cu121 torch. The old
#      version uninstalled it without reinstalling (pyproject had no torch),
#      leaving the image with NO torch at all — training was impossible.
#   2. No longer use `uv sync` to build the project package: the old version
#      only COPY'd pyproject.toml (missing README and source), so uv sync would
#      fail; uv also creates a .venv by default while Jupyter uses the system
#      python, installing deps in the wrong place. Now deps are installed via
#      pip from requirements-gpu.txt (torch reused from the base image),
#      consistent with the downloader's pip approach.
#   3. Copy working_space/ training code into /kaggle/code: the old image had
#      no training code, and compose only mounted the downloader's scripts, so
#      the training scripts were not found.
#   4. Build context moved to the repo root (see docker-compose build.context),
#      combined with the repo-root .dockerignore excluding input/ and other
#      large directories, to avoid sending hundreds of GB into the build
#      context (which would stall or fail the build).
# ============================================================
FROM gcr.io/kaggle-gpu-images/python:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/kaggle/code

# 1) Create the kaggle user and the /kaggle directory structure (Kaggle style)
RUN useradd -m -d /kaggle -s /bin/bash kaggle \
    && usermod -aG sudo kaggle \
    && echo "kaggle ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /kaggle/input /kaggle/working /kaggle/code \
    && chown -R kaggle:kaggle /kaggle

# 2) Copy the training code (build context = repo root; .dockerignore already
#    excludes large directories and secrets)
COPY working_space/ /kaggle/code/

# 3) Install/lock remaining deps (torch/torchvision/torchaudio reused from the
#    base image's cu121 build; do not reinstall)
COPY dockerfile/gpu/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-gpu.txt && \
    rm -f /tmp/requirements-gpu.txt && \
    python -c "import torch, torchvision, torchaudio; print('torch', torch.__version__)"

# 4) Entrypoint script: wait for the ESSD data-ready marker, then start JupyterLab
COPY dockerfile/gpu/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# 5) Switch to the kaggle user and start JupyterLab (Kaggle style)
USER kaggle
WORKDIR /kaggle
EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--notebook-dir=/kaggle", "--ServerApp.allow_remote_access=True"]
