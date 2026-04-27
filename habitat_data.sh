#!/bin/bash
# Download Gibson/Habitat dataset files
# Source: Gibson Database of 3D Spaces form
# https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform

set -e

DATA_DIR="${1:-/media/kojogyaase/disk_two/Research/recovery-from-failure/data/scene_datasets}"

echo "=== Gibson/Habitat Dataset Downloader ==="
echo "Download directory: $DATA_DIR"
echo ""

# ──────────────────────────────────────────────
# A. Gibson Env V1 data (CVPR 2018)
# ──────────────────────────────────────────────
GIBSON_V1_DIR="$DATA_DIR/gibson_v1"
mkdir -p "$GIBSON_V1_DIR"

echo "[A1] Gibson V1 Tiny (8.02 GB)..."
wget -c -P "$GIBSON_V1_DIR" https://storage.googleapis.com/gibson_scenes/gibson_tiny.tar.gz

echo "[A2] Gibson V1 Medium (20.8 GB)..."
wget -c -P "$GIBSON_V1_DIR" https://storage.googleapis.com/gibson_scenes/gibson_medium.tar.gz

echo "[A3] Gibson V1 Full (64.96 GB)..."
wget -c -P "$GIBSON_V1_DIR" https://storage.googleapis.com/gibson_scenes/gibson_full.tar.gz

echo "[A4] Gibson V1 Full+ (23.91 GB)..."
wget -c -P "$GIBSON_V1_DIR" https://storage.googleapis.com/gibson_scenes/gibson_fullplus.tar.gz

echo "[A5] Stanford 2D-3D-Semantics for Gibson (3.9 GB)..."
wget -c -P "$GIBSON_V1_DIR" https://storage.googleapis.com/gibson_scenes/2d3ds_for_gibson.tar.gz

# ──────────────────────────────────────────────
# B. iGibson / Gibson V2 (2020)
# ──────────────────────────────────────────────
IGIBSON_DIR="$DATA_DIR/igibson"
mkdir -p "$IGIBSON_DIR"

echo "[B1] iGibson dataset, 15 fully interactive scenes (no GDS required)..."
wget -c -P "$IGIBSON_DIR" https://storage.googleapis.com/gibson_scenes/ig_dataset.tar.gz

echo "[B2] 2D3DS for iGibson, 7 scenes (1.4 GB)..."
wget -c -P "$IGIBSON_DIR" https://storage.googleapis.com/gibsonchallenge/2d3ds_for_igibson.zip

echo "[B3] Gibson V2 4+ partition, 106 scenes (2.6 GB)..."
wget -c -P "$IGIBSON_DIR" https://storage.googleapis.com/gibson_scenes/gibson_v2_4+.tar.gz

echo "[B4] Gibson V2 all scenes, 572 scenes (108 GB)..."
wget -c -P "$IGIBSON_DIR" https://storage.googleapis.com/gibson_scenes/gibson_v2_all.tar.gz

echo "[B5] Interactive Gibson dataset, 10 scenes (780 MB)..."
wget -c -P "$IGIBSON_DIR" https://storage.googleapis.com/gibson_scenes/interactive_dataset.tar.gz

echo "[B6] Gibson sim2real challenge 2020 data..."
wget -c -P "$IGIBSON_DIR" https://storage.cloud.google.com/gibsonchallenge/gibson-challenge-data.tar.gz

# ──────────────────────────────────────────────
# C. Habitat-sim / challenge (ICCV 2019)
# ──────────────────────────────────────────────
HABITAT_DIR="$DATA_DIR/gibson_habitat"
mkdir -p "$HABITAT_DIR"

echo "[C1] Gibson Dataset trainval for Habitat (11 GB)..."
wget -c -P "$HABITAT_DIR" https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip

echo "[C2] Gibson Dataset for Habitat challenge, 4+ quality scenes (1.5 GB)..."
wget -c -P "$HABITAT_DIR" https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip

echo ""
echo "=== Downloads complete ==="
echo "Total files downloaded:"
echo "  A. Gibson V1:   tiny, medium, full, fullplus, 2d3ds"
echo "  B. iGibson/V2:  ig_dataset, 2d3ds_igibson, v2_4+, v2_all, interactive, challenge"
echo "  C. Habitat:     trainval, challenge"
echo ""
echo "NOTE: Total download size is very large (~230+ GB if all selected)."
echo "      Use wget -c to resume interrupted downloads."
echo "      Comment out sections you don't need before running."