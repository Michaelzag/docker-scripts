#!/bin/zsh
set -euo pipefail

SERVER=/Users/michael/projects/macmini/nemotron-3.5-lightning-server/runtime/llama-adb55e5/build/bin/llama-server
MODEL=/Users/michael/projects/macmini/models/nemotron-3.5-lightning-gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf
NEMOTRON_BATCH=${NEMOTRON_BATCH:-2048}
NEMOTRON_UBATCH=${NEMOTRON_UBATCH:-512}

exec "$SERVER" \
  --model "$MODEL" \
  --gpu-layers 99 \
  --ctx-size 524288 \
  --parallel 4 \
  --batch-size "$NEMOTRON_BATCH" \
  --ubatch-size "$NEMOTRON_UBATCH" \
  --alias mac/nvidia/nemotron-3.5-lightning-30b-a3b \
  --host 0.0.0.0 \
  --port 2345 \
  --jinja \
  --flash-attn on \
  --temp 1.0 \
  --top-p 0.95 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --metrics \
  --no-webui
