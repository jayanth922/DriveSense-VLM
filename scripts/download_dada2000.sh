#!/usr/bin/env bash
# =============================================================================
# download_dada2000.sh — Download and extract the DADA-2000 dataset
# =============================================================================
# Usage:
#   bash scripts/download_dada2000.sh
#
# Environment:
#   HPC_DATA_ROOT  Override data download directory (default: ~/data)
#
# DADA-2000 Availability:
#   DADA-2000 is a research dataset from the paper:
#   "DADA: Driver Attention in Driving Accident Scenarios" (TITS 2022)
#   Authors: Jianwu Fang, Dingxin Yan, Jiahuan Qiao, Jianru Xue
#
#   Access request:
#   1. Email the authors at: jianwu.fang@nwpu.edu.cn
#   2. Include your institution, research purpose, and PI contact
#   3. You will receive a download link (typically Google Drive or Baidu Pan)
#   4. Update the DADA_DOWNLOAD_URL variable below with the received link
#
#   Paper: https://ieeexplore.ieee.org/document/9156559
#   Project: https://github.com/JWFangit/LOTVS-DADA
# =============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_ROOT="${HPC_DATA_ROOT:-$HOME/data}"
TARGET_DIR="${DATA_ROOT}/dada2000"

# PLACEHOLDER: Replace with the actual download URL received from dataset authors
# Typical format: Google Drive or institutional server link
DADA_DOWNLOAD_URL="${DADA_DOWNLOAD_URL:-}"

# ── Check for download URL ─────────────────────────────────────────────────────
echo "============================================================"
echo " DriveSense-VLM — DADA-2000 Dataset Download"
echo "============================================================"
echo " Target dir: ${TARGET_DIR}"
echo ""

if [[ -z "${DADA_DOWNLOAD_URL:-}" ]]; then
    echo "============================================================"
    echo " DADA-2000 Download Instructions"
    echo "============================================================"
    echo ""
    echo " DADA-2000 requires a manual data access request:"
    echo ""
    echo " 1. Email the authors at: jianwu.fang@nwpu.edu.cn"
    echo "    Subject: DADA-2000 Dataset Access Request"
    echo "    Include: Institution, research purpose, PI contact"
    echo ""
    echo " 2. Review the dataset paper:"
    echo "    https://ieeexplore.ieee.org/document/9156559"
    echo ""
    echo " 3. GitHub project (may have updated access instructions):"
    echo "    https://github.com/JWFangit/LOTVS-DADA"
    echo ""
    echo " 4. Once you receive the download link, run:"
    echo "    export DADA_DOWNLOAD_URL='<your_download_link>'"
    echo "    bash scripts/download_dada2000.sh"
    echo ""
    echo "============================================================"
    exit 0
fi

# ── Check if already downloaded ───────────────────────────────────────────────
if [[ -d "${TARGET_DIR}/videos" ]]; then
    echo "Dataset already exists at: ${TARGET_DIR}"
    echo "Skipping download. Delete the directory to re-download."
    exit 0
fi

# ── Create target directory ────────────────────────────────────────────────────
mkdir -p "${TARGET_DIR}"

# ── Download archive ───────────────────────────────────────────────────────────
ARCHIVE_PATH="${TARGET_DIR}/dada2000.zip"
echo "[DOWN] Downloading DADA-2000 from provided URL..."

if command -v wget &>/dev/null; then
    wget --progress=bar:force \
         --output-document="${ARCHIVE_PATH}" \
         "${DADA_DOWNLOAD_URL}"
else
    curl --progress-bar \
         --location \
         --output "${ARCHIVE_PATH}" \
         "${DADA_DOWNLOAD_URL}"
fi

echo "[EXTR] Extracting archive..."
unzip -q "${ARCHIVE_PATH}" -d "${TARGET_DIR}"
rm "${ARCHIVE_PATH}"

# ── Verify file counts ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Verifying extraction..."
echo "============================================================"

VIDEO_COUNT=0
ANNOT_COUNT=0

if [[ -d "${TARGET_DIR}/videos" ]]; then
    VIDEO_COUNT=$(find "${TARGET_DIR}/videos" -name "*.mp4" -o -name "*.avi" 2>/dev/null | wc -l)
    echo " [OK]  videos/  — ${VIDEO_COUNT} video files"
else
    echo " [MISSING] videos/"
fi

if [[ -d "${TARGET_DIR}/annotations" ]]; then
    ANNOT_COUNT=$(find "${TARGET_DIR}/annotations" -name "*.json" 2>/dev/null | wc -l)
    echo " [OK]  annotations/ — ${ANNOT_COUNT} annotation files"
else
    echo " [MISSING] annotations/"
fi

# Expected: ~2000 videos and corresponding annotations
if [[ ${VIDEO_COUNT} -gt 0 && ${ANNOT_COUNT} -gt 0 ]]; then
    echo ""
    echo "Extraction complete!"
    echo "  Videos:      ${VIDEO_COUNT}"
    echo "  Annotations: ${ANNOT_COUNT}"
    echo "  Dataset path: ${TARGET_DIR}"
    echo ""
    echo "Update configs/data.yaml if your path differs:"
    echo "  paths.dada2000_root: '${TARGET_DIR}'"
else
    echo ""
    echo "WARNING: Low file counts detected. Archive may be incomplete." >&2
    echo "Expected ~2000 videos and annotations in DADA-2000." >&2
    exit 1
fi
