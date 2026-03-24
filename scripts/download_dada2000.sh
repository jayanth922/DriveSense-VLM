#!/usr/bin/env bash
# =============================================================================
# download_dada2000.sh — Download and extract the DADA-2000 dataset
# =============================================================================
# Usage:
#   bash scripts/download_dada2000.sh
#
# Environment variables:
#   HPC_DATA_ROOT       Override data download directory (default: ~/data)
#   DADA_DOWNLOAD_URL   Direct download URL (received from dataset authors)
#
# DADA-2000 Dataset:
#   "DADA: Driver Attention in Driving Accident Scenarios" (IEEE TITS 2022)
#   Authors: Jianwu Fang, Dingxin Yan, Jiahuan Qiao, Jianru Xue
#
# Access options (in order of preference):
#   1. HuggingFace Hub (community re-host — check for availability):
#      https://huggingface.co/datasets — search "DADA-2000"
#   2. GitHub project page (may have updated access links):
#      https://github.com/JWFangit/LOTVS-DADA
#   3. Baidu Netdisk (original distribution):
#      Check the GitHub repo README for the current Baidu Pan link.
#   4. Email request to dataset authors:
#      jianwu.fang@nwpu.edu.cn — include institution and research purpose.
#
# Expected directory layout after extraction:
#   <dada_root>/
#     DADA-2000/
#       001/           # accident category (001–054)
#         001/         # sequence ID
#           images/
#             001.png  # frame files (PNG, zero-padded)
#             002.png
#             ...
#         002/
#           ...
#       002/
#         ...
#     dada_text_annotations.xlsx   # optional text annotations (if provided)
#
# Paper: https://ieeexplore.ieee.org/document/9156559
# =============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_ROOT="${HPC_DATA_ROOT:-$HOME/data}"
TARGET_DIR="${DATA_ROOT}/dada2000"
DADA_DOWNLOAD_URL="${DADA_DOWNLOAD_URL:-}"

echo "============================================================"
echo " DriveSense-VLM — DADA-2000 Dataset Download"
echo "============================================================"
echo " Target dir: ${TARGET_DIR}"
echo ""

# ── No URL provided: print manual instructions ─────────────────────────────────
if [[ -z "${DADA_DOWNLOAD_URL:-}" ]]; then
    echo "DADA_DOWNLOAD_URL is not set. Printing access instructions."
    echo ""
    echo "  Option 1 — HuggingFace Hub"
    echo "    Search https://huggingface.co/datasets for 'DADA-2000'"
    echo "    and follow the dataset card instructions."
    echo ""
    echo "  Option 2 — GitHub project page"
    echo "    https://github.com/JWFangit/LOTVS-DADA"
    echo "    Check the README for the latest Baidu Netdisk or GDrive link."
    echo ""
    echo "  Option 3 — Email request"
    echo "    To: jianwu.fang@nwpu.edu.cn"
    echo "    Subject: DADA-2000 Dataset Access Request"
    echo "    Include: your institution, research purpose, and PI contact."
    echo ""
    echo "Once you have the download URL or archive, run:"
    echo "  export DADA_DOWNLOAD_URL='<url>'"
    echo "  bash scripts/download_dada2000.sh"
    echo ""
    echo "Or place the extracted archive at:"
    echo "  ${TARGET_DIR}/DADA-2000/"
    echo ""
    echo "Then update configs/data.yaml:"
    echo "  paths.dada2000_root: '${TARGET_DIR}'"
    exit 0
fi

# ── Already downloaded ─────────────────────────────────────────────────────────
if [[ -d "${TARGET_DIR}/DADA-2000" ]]; then
    echo "[OK] Dataset already exists at: ${TARGET_DIR}/DADA-2000"
    echo "     Delete the directory to re-download."
    exit 0
fi

mkdir -p "${TARGET_DIR}"

# ── Download archive ───────────────────────────────────────────────────────────
ARCHIVE_PATH="${TARGET_DIR}/dada2000_archive.zip"
echo "[1/3] Downloading DADA-2000 ..."

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

# ── Extract ────────────────────────────────────────────────────────────────────
echo "[2/3] Extracting archive ..."
unzip -q "${ARCHIVE_PATH}" -d "${TARGET_DIR}"
rm "${ARCHIVE_PATH}"

# ── Verify ────────────────────────────────────────────────────────────────────
echo "[3/3] Verifying extraction ..."

CATEGORY_COUNT=0
FRAME_COUNT=0

if [[ -d "${TARGET_DIR}/DADA-2000" ]]; then
    CATEGORY_COUNT=$(find "${TARGET_DIR}/DADA-2000" -mindepth 1 -maxdepth 1 -type d | wc -l)
    FRAME_COUNT=$(find "${TARGET_DIR}/DADA-2000" -name "*.png" | wc -l)
    echo " [OK]  DADA-2000/  — ${CATEGORY_COUNT} categories, ${FRAME_COUNT} PNG frames"
else
    echo " [MISSING] DADA-2000/ sub-directory not found."
    echo "  The archive may use a different top-level directory name."
    echo "  Rename or symlink it so that the structure is:"
    echo "    ${TARGET_DIR}/DADA-2000/<cat>/<seq>/images/*.png"
    exit 1
fi

if [[ -f "${TARGET_DIR}/dada_text_annotations.xlsx" ]]; then
    echo " [OK]  dada_text_annotations.xlsx found"
else
    echo " [INFO] dada_text_annotations.xlsx not found (optional)."
    echo "        The loader will infer metadata from file paths."
fi

echo ""
echo "Download complete!"
echo "  Categories : ${CATEGORY_COUNT}"
echo "  PNG frames : ${FRAME_COUNT}"
echo "  Path       : ${TARGET_DIR}"
echo ""
echo "Run the extraction pipeline:"
echo "  python scripts/run_dada_extraction.py --dada-root '${TARGET_DIR}'"
