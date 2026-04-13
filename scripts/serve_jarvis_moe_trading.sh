#!/bin/bash
# Serve Jarvis MoE trading (Qwen3.5-35B-A3B + Cycle 2 DoRA merged, BF16) via vLLM.
# Includes qwen3 reasoning parser so <think>...</think> splits into reasoning_content.

export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas

exec ~/models/vllm-native/bin/python3 -u -m vllm.entrypoints.openai.api_server \
    --model /home/brandonv/models/Jarvis_MoE_trading \
    --dtype bfloat16 \
    --trust-remote-code \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --host 0.0.0.0
