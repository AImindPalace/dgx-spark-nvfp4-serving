#!/bin/bash
# Serve Jarvis_27B_trading — BF16 Qwen3.5-27B + Cycle 2 DoRA (merged, MTP weights injected)
# VLM wrapper config + --language-model-only (Qwen3_5 class routes through VLM path)
# qwen3_next_mtp speculative decoding for ~1.5x speedup
# qwen3 reasoning parser for <think>/answer split

export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas

exec ~/models/vllm-native/bin/python3 -u -m vllm.entrypoints.openai.api_server \
    --model /home/brandonv/models/Jarvis_27B_trading \
    --dtype bfloat16 \
    --trust-remote-code \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.70 \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --language-model-only \
    --speculative-config "{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":1}" \
    --host 0.0.0.0
