#!/bin/bash
# ============================================================
# GPU training container entrypoint
# Role: wait for the "data ready" marker written by the CPU downloader, then
# start JupyterLab.
#
# Environment variables (overridable):
#   DATA_READY_WAIT    = 1 wait / 0 skip (default 1)
#   DATA_READY_MARKER  = marker file path (default /kaggle/working/.download_complete)
#   DATA_READY_TIMEOUT = max wait seconds (default 1800)
# ============================================================
set -e

MARKER="${DATA_READY_MARKER:-/kaggle/working/.download_complete}"
TIMEOUT="${DATA_READY_TIMEOUT:-1800}"

if [ "${DATA_READY_WAIT:-1}" = "1" ]; then
    echo "[entrypoint] Waiting for data-ready marker: ${MARKER} (up to ${TIMEOUT}s)..."
    waited=0
    until [ -f "${MARKER}" ] || [ "${waited}" -ge "${TIMEOUT}" ]; do
        sleep 5
        waited=$((waited + 5))
    done
    if [ -f "${MARKER}" ]; then
        echo "[entrypoint] ✓ Data ready: $(cat "${MARKER}")"
    else
        echo "[entrypoint] ⚠ Timed out waiting for the data marker; starting anyway (check ESSD mount / downloader)"
    fi
fi

exec "$@"
