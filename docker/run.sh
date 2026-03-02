#!/usr/bin/env bash
set -e

IMAGE_NAME="tag_gar:cu124"
CONTAINER_NAME="tag_gar_dev"

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 改成你真实的大数据路径（WSL路径）
TAG_DATA_DIR="/home/yota/code/stgcn-clean/data/basketball"

mkdir -p "${PROJ_DIR}/outputs"

docker run --gpus all -it --rm \
  --name "${CONTAINER_NAME}" \
  -e TZ=Asia/Tokyo \
  -e SGA_ROOT=/datasets/basketball \
  -v "${PROJ_DIR}:/workspace:rw" \
  -v "${TAG_DATA_DIR}:/datasets/basketball:ro" \
  -v "${PROJ_DIR}/outputs:/workspace/outputs:rw" \
  "${IMAGE_NAME}" \
  bash
