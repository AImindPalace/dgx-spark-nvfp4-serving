#!/bin/bash
# Serve Jarvis_MoE_trading (Qwen3.5-35B-A3B + Cycle 2 DoRA merged, Q4_K_M GGUF) via llama-server
# Target: ~50-58 tok/s per NVIDIA/community benchmarks (vs ~30 tok/s on vLLM BF16)

export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-13.2/bin:$PATH

exec ~/models/llama.cpp/build/bin/llama-server \
    --model /home/brandonv/models/Jarvis_MoE_trading.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 30001 \
    --n-gpu-layers 99 \
    --ctx-size 16384 \
    --threads 8 \
    --jinja
