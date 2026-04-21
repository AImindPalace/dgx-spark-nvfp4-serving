#!/bin/bash
# Serve Jarvis_27B_trading + DFlash speculative decoding (z-lab/Qwen3.5-27B-DFlash drafter).
# Mirrors serve_jarvis_27b_trading.sh (BF16, fine-tune target) but swaps qwen3_next_mtp for dflash.
# Port 8001 so it runs alongside the MTP server on 8000 during A/B testing.
#
# Prereq: drafter downloaded to $DRAFTER_PATH. If missing, run:
#   hf download z-lab/Qwen3.5-27B-DFlash --local-dir /home/brandonv/models/Qwen3.5-27B-DFlash

set -euo pipefail

TARGET_PATH="${TARGET_PATH:-/home/brandonv/models/Jarvis_27B_trading}"
DRAFTER_PATH="${DRAFTER_PATH:-/home/brandonv/models/Qwen3.5-27B-DFlash}"
PORT="${PORT:-8001}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"

if [[ ! -d "$DRAFTER_PATH" ]]; then
    echo "ERROR: drafter not found at $DRAFTER_PATH" >&2
    echo "Run: hf download z-lab/Qwen3.5-27B-DFlash --local-dir $DRAFTER_PATH" >&2
    exit 1
fi

export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas

exec ~/models/vllm-native/bin/python3 -u -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_PATH" \
    --dtype bfloat16 \
    --trust-remote-code \
    --port "$PORT" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.75 \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --language-model-only \
    --attention-backend flash_attn \
    --max-num-batched-tokens 32768 \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFTER_PATH\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}" \
    --host 0.0.0.0
