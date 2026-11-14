#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR"

# DW4 / LJ13 / LJ55 test splits (DEM, github.com/jarridrb/DEM)
NPYS=(
    test_split_DW4.npy
    test_split_LJ13-1000.npy
    test_split_LJ55-1000-part1.npy
)

DEM_GITHUB="https://raw.githubusercontent.com/jarridrb/DEM/main/data"

for NPY in "${NPYS[@]}"; do
    curl -fL -o "$DEST_DIR/$NPY" "$DEM_GITHUB/$NPY"
done

# Alanine dipeptide: PITA's 300 K MD test and validation splits (10,000 frames
# each, nm), from PITA's public Mendeley dataset
# (https://data.mendeley.com/datasets/jnrdksfsyp),
# alanine/AL22_temp_300.00/{test,val}_split_AL22-10000.npy, checked against
# the dataset's SHA-256. (PITA's force field, data/aldp_ff14sb_obc1.npz, is tracked.)
MENDELEY="https://data.mendeley.com/public-files/datasets/jnrdksfsyp/files"
ALDP=(
    "test_split_AL22-10000_T300.npy cdf55685-40e5-4471-a276-200f80837ae8 137975efea8646f5516fff764354e81d01de47a36c65062f1f85967168190c4c"
    "val_split_AL22-10000_T300.npy 3c69d640-8a05-407d-b5ab-1724e169e3c4 83ea4a71dc93c42306ac5c741af93d8a16fe7d24907aa9191c6a3621a8dbb9b4"
)

for ENTRY in "${ALDP[@]}"; do
    read -r NAME ID SHA <<< "$ENTRY"
    curl -fL -o "$DEST_DIR/$NAME" "$MENDELEY/$ID/file_downloaded"
    echo "$SHA  $DEST_DIR/$NAME" | sha256sum -c -
done

echo "Files downloaded to $DEST_DIR"
