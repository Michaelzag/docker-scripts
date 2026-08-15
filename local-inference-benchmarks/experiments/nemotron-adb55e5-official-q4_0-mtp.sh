#!/bin/zsh
set -euo pipefail

SERVER=/Users/michael/projects/macmini/nemotron-3.5-lightning-server/runtime/llama-adb55e5/build/bin/llama-server
MODEL_DIR=/Users/michael/projects/macmini/models/nemotron-3.5-lightning-gguf/official-q4_0
MODEL="$MODEL_DIR/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
DRAFT="$MODEL_DIR/mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
NEMOTRON_BATCH=${NEMOTRON_BATCH:-4096}
NEMOTRON_UBATCH=${NEMOTRON_UBATCH:-1024}
NEMOTRON_DRAFT_MAX=${NEMOTRON_DRAFT_MAX:-3}

spec_args=()
if (( NEMOTRON_DRAFT_MAX > 0 )); then
  spec_args=(
    --spec-draft-model "$DRAFT"
    --spec-type draft-mtp
    --spec-draft-n-max "$NEMOTRON_DRAFT_MAX"
  )
fi

exec "$SERVER" \
  --model "$MODEL" \
  "${spec_args[@]}" \
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
