#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export KAGGLE_CONFIG_DIR="$PROJECT_DIR/.local/kaggle"
mkdir -p "$KAGGLE_CONFIG_DIR"
chmod 700 "$KAGGLE_CONFIG_DIR"
exec "$PROJECT_DIR/.venv/bin/kaggle" "$@"
