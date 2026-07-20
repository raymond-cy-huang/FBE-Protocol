#!/bin/bash
set -e

if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

SAM_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything"

echo "Downloading sam_vit_b_01ec64.pth checkpoint..."
$CMD "${SAM_BASE_URL}/sam_vit_b_01ec64.pth" || { echo "Failed to download sam_vit_b_01ec64.pth"; exit 1; }

echo "Downloading sam_vit_l_0b3195.pth checkpoint..."
$CMD "${SAM_BASE_URL}/sam_vit_l_0b3195.pth" || { echo "Failed to download sam_vit_l_0b3195.pth"; exit 1; }

echo "Downloading sam_vit_h_4b8939.pth checkpoint..."
$CMD "${SAM_BASE_URL}/sam_vit_h_4b8939.pth" || { echo "Failed to download sam_vit_h_4b8939.pth"; exit 1; }

echo "All checkpoints are downloaded successfully."
