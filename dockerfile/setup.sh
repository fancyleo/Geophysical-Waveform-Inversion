#!/bin/bash
# ============================================================
# Aliyun Kaggle continue-training workflow - one-click deployment script
# Role: on the CPU machine, download the Kaggle dataset to the ESSD (/data)
#       so the GPU container can train on it.
# Usage: bash setup.sh
# ============================================================
set -e

echo "===== [1/5] Checking the Docker environment ====="
if ! command -v docker &> /dev/null; then
    echo "✗ Docker is not installed; install it first:"
    echo "  curl -fsSL https://get.docker.com | sh"
    exit 1
fi
echo "✓ Docker $(docker --version)"

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "✗ docker-compose is not installed; install it first"
    exit 1
fi
echo "✓ docker-compose available"

echo ""
echo "===== [2/5] Creating the directory structure (ESSD mount point /data) ====="
echo "  Please confirm the ESSD is mounted at /data: df -h | grep /data  (block device /dev/sdb)"
mkdir -p /data/kaggle/input
mkdir -p /data/kaggle/working
mkdir -p ./kaggle-api
chmod -R 777 /data/kaggle
echo "✓ /data/kaggle/{input,working} created"

echo ""
echo "===== [3/5] Configuration files ====="
echo "  ① docker-compose.yml        - orchestrates the CPU downloader + GPU trainer"
echo "  ② downloader/Dockerfile     - CPU download container"
echo "  ③ gpu/Dockerfile            - GPU training container (Kaggle style)"
echo "  ④ gpu/requirements-gpu.txt  - GPU dependency pins (torch reused from base image)"
echo "  ⑤ gpu/entrypoint.sh         - GPU entrypoint: waits for data-ready, then starts Jupyter"
echo "  ⑥ scripts/download.sh       - download script"
echo "  ⑦ scripts/preprocess.py     - preprocessing script (CPU side)"
echo "  ⑧ .dockerignore             - excludes input/ and other large dirs and secrets to keep the build context small"
echo ""

echo "===== [4/5] Before you start ====="
echo "  1) Put the Kaggle API token at: ./kaggle-api/kaggle.json"
echo "     Get it at: https://www.kaggle.com/settings -> Create New Token"
echo "  2) Edit the .env file and fill in the Aliyun settings (see .env.example)"
echo ""

echo "===== [5/5] Quick start commands ====="
echo "  # Step 1: download data on the CPU machine (ESSD mounted at /data)"
echo "  docker compose --profile downloader up --build"
echo ""
echo "  # Step 2: migrate the ESSD (umount+detach on the CPU machine -> attach on the GPU machine and mount /dev/sdb /data)"
echo ""
echo "  # Step 3: start training on the GPU machine"
echo "  docker compose --profile gpu up --build -d"
echo ""
echo "  # View logs"
echo "  docker compose logs -f gpu-train"
echo ""
echo "✓ Deployment script finished"
