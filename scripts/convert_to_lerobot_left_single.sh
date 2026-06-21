#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="grasp the cup and place it on the black plate"
SOURCE_DIR="/mnt/workspace/xiajiawei/kai0/data/task_1_left_single_collect_001"
OUTPUT_DIR="/mnt/workspace/xiajiawei/kai0/data/task_1_left_single_collect_001_lerobot"
OVERWRITE=""
INCLUDE_THIRD_CAMERA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-name)
      TASK_NAME="$2"
      shift 2
      ;;
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="--overwrite"
      shift
      ;;
    --include-third-camera)
      INCLUDE_THIRD_CAMERA="--include-third-camera"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd /mnt/workspace/xiajiawei/kai0

export LD_LIBRARY_PATH="/mnt/workspace/xiajiawei/ffmpeg61_libs:${LD_LIBRARY_PATH:-}"

./.venv/bin/python train_deploy_alignment/data_augment/convert_h5_lerobot_left_single_official.py \
  --source-dir "$SOURCE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --task-name "$TASK_NAME" \
  $OVERWRITE \
  $INCLUDE_THIRD_CAMERA
