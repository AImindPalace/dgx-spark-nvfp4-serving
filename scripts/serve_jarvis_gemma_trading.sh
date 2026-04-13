#!/bin/bash
# Serve Jarvis_Gemma_trading (Gemma 4 26B-A4B + Cycle 2 DoRA merged, F16 GGUF) via llama-server
# Uses NVIDIAs DGX Spark llama.cpp recipe with SM_121 CUDA build.

export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-13.2/bin:$PATH

exec ~/models/llama.cpp/build/bin/llama-server \
    --model /home/brandonv/models/Jarvis_Gemma_trading.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 30000 \
    --n-gpu-layers 99 \
    --ctx-size 16384 \
    --threads 8 \
    --jinja
