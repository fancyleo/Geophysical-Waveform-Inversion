#!/bin/bash
# ============================================================
# Kaggle dataset download script (runs inside the CPU container)
# Features: resume, dedupe, completion marker, supports competition/dataset
# ============================================================
set -e

echo "===== Kaggle Downloader ====="
echo "Competition: ${KAGGLE_COMPETITION}"
echo "Download to: ${DOWNLOAD_DIR}"
echo "Working dir: ${WORKING_DIR}"
echo ""

mkdir -p "${DOWNLOAD_DIR}" "${WORKING_DIR}"

# ----------------------------------------------------------
# Decide the download type: competition or dataset
# Controlled by the KAGGLE_TYPE env var; defaults to competition
# ----------------------------------------------------------
KAGGLE_TYPE="${KAGGLE_TYPE:-competition}"

if [ "${KAGGLE_TYPE}" = "dataset" ]; then
    # Format: <owner>/<dataset-name>, e.g. "liulong/geophysical-dataset"
    echo ">>> Downloading dataset: ${KAGGLE_COMPETITION}"
    kaggle datasets download -d "${KAGGLE_COMPETITION}" \
        -p "${DOWNLOAD_DIR}" \
        --resume \
        --unzip
else
    echo ">>> Downloading competition: ${KAGGLE_COMPETITION}"
    kaggle competitions download -c "${KAGGLE_COMPETITION}" \
        -p "${DOWNLOAD_DIR}" \
        --resume
fi

# ----------------------------------------------------------
# Unzip competition zips (dataset already uses --unzip)
# ----------------------------------------------------------
if [ "${KAGGLE_TYPE}" != "dataset" ]; then
    echo ">>> Unzipping files..."
    cd "${DOWNLOAD_DIR}"
    for zipfile in *.zip; do
        [ -f "${zipfile}" ] || continue
        echo "  unzip: ${zipfile}"
        unzip -o -q "${zipfile}" -d .
        rm -f "${zipfile}"
    done
fi

# ----------------------------------------------------------
# Write the completion marker (lets the GPU container / scheduler know data is ready)
# ----------------------------------------------------------
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${WORKING_DIR}/.download_complete"
echo ""
echo "===== Download Complete ====="
ls -lh "${DOWNLOAD_DIR}"
echo "✓ Marker file: ${WORKING_DIR}/.download_complete"
