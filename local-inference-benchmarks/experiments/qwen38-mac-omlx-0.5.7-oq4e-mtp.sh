#!/bin/zsh
set -euo pipefail

exec /Users/michael/projects/macmini/qwen38-omlx-server/.venv/bin/omlx serve \
  --model-dir /Users/michael/projects/macmini/models/qwen38-omlx/fcmeyer \
  --host 0.0.0.0 \
  --port 2345 \
  --log-level info \
  --max-concurrent-requests 1 \
  --memory-guard balanced \
  --paged-ssd-cache-dir /Users/michael/projects/macmini/qwen38-omlx-server/state/cache \
  --paged-ssd-cache-max-size 20GB \
  --hot-cache-max-size 8GB \
  --no-hf-cache \
  --base-path /Users/michael/projects/macmini/qwen38-omlx-server/state \
  --api-key none
