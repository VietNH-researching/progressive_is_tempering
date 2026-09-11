#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR"

NPYS=(
    test_split_DW4.npy
    test_split_LJ13-1000.npy
    test_split_LJ55-1000-part1.npy
)

DEM_GITHUB="https://raw.githubusercontent.com/jarridrb/DEM/main/data"

for NPY in "${NPYS[@]}"; do
    curl -fL -o "$DEST_DIR/$NPY" "$DEM_GITHUB/$NPY"
done

echo "Files downloaded to $DEST_DIR"
