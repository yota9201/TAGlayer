#!/usr/bin/env bash
set -euo pipefail

MODE="dry-run"
if [[ "${1:-}" == "--apply" ]]; then
  MODE="apply"
elif [[ "${1:-}" == "--dry-run" || "${1:-}" == "" ]]; then
  MODE="dry-run"
else
  echo "Usage: $0 [--dry-run|--apply]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Project root: $ROOT_DIR"
echo "Mode: $MODE"

mapfile -d '' TARGETS < <(find . \
  \( -type d -name "__pycache__" \
  -o -type d -name "build" \
  -o -type d -name "dist" \
  -o -type d -name "*.egg-info" \
  -o -type f -name "*.pyc" \
  -o -type f -name "*.pyo" \
  -o -type f -name "*.pyd" \
  -o -type f -name "*:Zone.Identifier" \) \
  -print0)

COUNT="${#TARGETS[@]}"

if [[ "$COUNT" -eq 0 ]]; then
  echo "No cleanup targets found."
  exit 0
fi

echo "Found $COUNT cleanup targets."
echo "Top 20 targets:"
for t in "${TARGETS[@]:0:20}"; do
  echo "  $t"
done

if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry run complete. Re-run with --apply to delete these targets."
  exit 0
fi

for t in "${TARGETS[@]}"; do
  rm -rf -- "$t"
done

echo "Cleanup complete. Removed $COUNT targets."
