#!/usr/bin/env bash
# =============================================================================
# download_nuscenes.sh — Download nuScenes dataset to HPC storage
# =============================================================================
# Usage:
#   bash scripts/download_nuscenes.sh [mini|trainval]
#
# Arguments:
#   mini      Download the mini split (~4 GB) — recommended for development
#   trainval  Download the full trainval split (~60 GB) — required for training
#
# If no argument is provided, defaults to "mini".
#
# Environment:
#   HPC_DATA_ROOT  Override data download directory (default: ~/data)
#   NUSCENES_TOKEN nuScenes download token from https://nuscenes.org
#
# Requirements:
#   - Register at https://nuscenes.org to obtain an API download token
#   - Set NUSCENES_TOKEN environment variable before running
#   - Run on HPC storage node or submit-host (not compute node)
# =============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
SPLIT="${1:-mini}"
DATA_ROOT="${HPC_DATA_ROOT:-$HOME/data}"
TARGET_DIR="${DATA_ROOT}/nuscenes"

# ── Validate split argument ────────────────────────────────────────────────────
if [[ "$SPLIT" != "mini" && "$SPLIT" != "trainval" ]]; then
    echo "ERROR: Invalid split '${SPLIT}'. Choose 'mini' or 'trainval'." >&2
    exit 1
fi

# ── Check for download token ───────────────────────────────────────────────────
if [[ -z "${NUSCENES_TOKEN:-}" ]]; then
    echo "============================================================"
    echo " nuScenes Download Token Required"
    echo "============================================================"
    echo ""
    echo " 1. Register at: https://www.nuscenes.org/nuscenes#download"
    echo " 2. Accept the Terms of Use and download agreement"
    echo " 3. Copy your API token from the download page"
    echo " 4. Export the token before running this script:"
    echo ""
    echo "    export NUSCENES_TOKEN='your_token_here'"
    echo "    bash scripts/download_nuscenes.sh ${SPLIT}"
    echo ""
    echo " NOTE: Tokens are tied to your registered account email."
    echo "============================================================"
    exit 1
fi

# ── Dataset size warnings ──────────────────────────────────────────────────────
echo "============================================================"
echo " DriveSense-VLM — nuScenes Dataset Download"
echo "============================================================"
echo " Split:       ${SPLIT}"
echo " Target dir:  ${TARGET_DIR}"

if [[ "$SPLIT" == "mini" ]]; then
    echo " Approx size: ~4 GB (mini — 10 scenes)"
    echo " Recommended: Use for local dev / sanity checks"
    # nuScenes mini filenames (update if download URLs change)
    FILES=(
        "v1.0-mini.tgz"
    )
else
    echo " Approx size: ~60 GB (trainval — 700 scenes)"
    echo " WARNING: Ensure you have at least 80 GB free on ${DATA_ROOT}"
    echo " Recommended: Run on HPC scratch storage, not home directory"
    FILES=(
        "v1.0-trainval01_blobs.tgz"
        "v1.0-trainval02_blobs.tgz"
        "v1.0-trainval03_blobs.tgz"
        "v1.0-trainval04_blobs.tgz"
        "v1.0-trainval05_blobs.tgz"
        "v1.0-trainval06_blobs.tgz"
        "v1.0-trainval07_blobs.tgz"
        "v1.0-trainval08_blobs.tgz"
        "v1.0-trainval09_blobs.tgz"
        "v1.0-trainval10_blobs.tgz"
        "v1.0-trainval_meta.tgz"
    )
fi

echo "============================================================"
echo ""

# ── Check if already downloaded ───────────────────────────────────────────────
VERSION_DIR="${TARGET_DIR}/v1.0-${SPLIT}"
if [[ -d "${VERSION_DIR}" ]]; then
    echo "Dataset already exists at: ${VERSION_DIR}"
    echo "Skipping download. Delete the directory to re-download."
    exit 0
fi

# ── Create target directory ────────────────────────────────────────────────────
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

# ── Download files ─────────────────────────────────────────────────────────────
BASE_URL="https://s3.amazonaws.com/data.nuscenes.org/public/${SPLIT}"

echo "Starting download to: ${TARGET_DIR}"
echo ""

for FILE in "${FILES[@]}"; do
    OUTPUT_PATH="${TARGET_DIR}/${FILE}"

    if [[ -f "${OUTPUT_PATH}" ]]; then
        echo "[SKIP] Already downloaded: ${FILE}"
        continue
    fi

    echo "[DOWN] Downloading: ${FILE}"
    # Use wget with progress bar; fall back to curl if wget unavailable
    if command -v wget &>/dev/null; then
        wget --header="Authorization: token ${NUSCENES_TOKEN}" \
             --progress=bar:force \
             --output-document="${OUTPUT_PATH}" \
             "${BASE_URL}/${FILE}"
    else
        curl --header "Authorization: token ${NUSCENES_TOKEN}" \
             --progress-bar \
             --location \
             --output "${OUTPUT_PATH}" \
             "${BASE_URL}/${FILE}"
    fi

    echo "[EXTR] Extracting: ${FILE}"
    tar -xzf "${OUTPUT_PATH}" -C "${TARGET_DIR}"
    rm "${OUTPUT_PATH}"
    echo "[DONE] Extracted: ${FILE}"
    echo ""
done

# ── Verify extraction ──────────────────────────────────────────────────────────
echo "============================================================"
echo " Verifying extraction..."
echo "============================================================"

EXPECTED_DIRS=("maps" "samples" "sweeps" "v1.0-${SPLIT}")
MISSING=()

for DIR in "${EXPECTED_DIRS[@]}"; do
    if [[ -d "${TARGET_DIR}/${DIR}" ]]; then
        echo " [OK] ${DIR}/"
    else
        echo " [MISSING] ${DIR}/"
        MISSING+=("${DIR}")
    fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo ""
    echo "Download complete!"
    echo "Dataset path: ${TARGET_DIR}"
    echo ""
    echo "Update configs/data.yaml if your path differs from default:"
    echo "  paths.nuscenes_root: '${TARGET_DIR}'"
    echo ""
    echo "Or set the environment variable:"
    echo "  export HPC_DATA_ROOT='${DATA_ROOT}'"
else
    echo ""
    echo "ERROR: Missing directories: ${MISSING[*]}" >&2
    echo "Download may be incomplete. Try re-running the script." >&2
    exit 1
fi
