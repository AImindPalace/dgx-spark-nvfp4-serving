# Jarvis_2 NVFP4 Quantization Runbook

**Target**: NVFP4-quantize `Jarvis_27B_trading` (Qwen3.5-27B + Cycle 2 DoRA merged, 54 GB BF16) into `Jarvis_2` (~19 GB NVFP4) for fast vLLM serving on DGX Spark with MTP.

**Why RunPod**: Spark's 121 GB unified memory OOMs during modelopt NVFP4 quantization (54 GB model + quant activations + optimizer state). Needs dedicated HBM.

**Cost/time**: ~$20-40, ~1-2 hours on H100/H200.

## Prereqs
- RunPod account with GPU access
- HF_TOKEN with write access to `bverbeck/*` (for uploading quantized model)
- Jarvis_27B_trading exists (merged earlier, on Spark and derivable from `Qwen/Qwen3.5-27B` + `bverbeck/trading-dora-adapter-v2`)

## Step 1: Launch pod

- Template: **Runpod Pytorch 2.4.0 (devel, has nvcc)**
- GPU: **1x H100 (80 GB) or H200 NVL (143 GB)** — H200 gives more headroom
- Volume: **200 GB minimum** (52 GB base + 54 GB merged + 19 GB output + calibration)

## Step 2: Install deps (exact versions — these match our merge pipeline)

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4

# Torch 2.7 + CUDA 12.6 — matches Spark's stack
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# Flash-attn + fla — required for Qwen3.5 fast paths during calibration
export TORCH_CUDA_ARCH_LIST='9.0'
pip install 'flash-attn==2.8.3' --no-build-isolation --no-deps --force-reinstall --no-cache-dir
pip install 'causal-conv1d>=1.4.0' --no-build-isolation --no-deps --force-reinstall --no-cache-dir
pip install -U --no-use-pep517 'git+https://github.com/fla-org/flash-linear-attention' --no-deps

# Merge + quantize deps
pip install transformers==5.5.3 peft huggingface_hub datasets sentencepiece protobuf accelerate

# modelopt for NVFP4 (NVIDIA's quantization toolkit)
pip install 'nvidia-modelopt[hf]' --extra-index-url https://pypi.nvidia.com

# Remove segfault source
pip uninstall -y hf-xet 2>/dev/null

# Verify
python3 -c "
import torch; print(f'torch {torch.__version__}')
import transformers; print(f'transformers {transformers.__version__}')
import modelopt.torch.quantization as mtq; print('modelopt OK')
print('READY')
"
```

## Step 3: Merge Cycle 2 adapter into base on RunPod

Upload `scripts/merge_adapter_qwen27_v2.py` + `scripts/post_merge_minimal.py` + `scripts/quantize_nvfp4_jarvis2.py` + a tarball of `training_data/` (for stratified calibration sampling):

```bash
# From local machine:
tar czf /tmp/training_data.tar.gz training_data/
scp -P <POD_PORT> /tmp/training_data.tar.gz \
    scripts/merge_adapter_qwen27_v2.py \
    scripts/post_merge_minimal.py \
    scripts/quantize_nvfp4_jarvis2.py \
    root@<POD_IP>:/workspace/
```

On the pod:
```bash
cd /workspace
tar xzf training_data.tar.gz

# Rewrite paths in merge script from ~/models/ to /workspace/
sed -i 's|/home/brandonv/models|/workspace|g' merge_adapter_qwen27_v2.py

export HF_TOKEN=<your_token>

# Download base + adapter (uses HF_TOKEN env)
python3 -c "
from huggingface_hub import snapshot_download
import os
print('Downloading Qwen3.5-27B (52 GB)...')
snapshot_download('Qwen/Qwen3.5-27B', local_dir='/workspace/Qwen3.5-27B',
    token=os.environ['HF_TOKEN'],
    allow_patterns=['*.safetensors','*.json','*.jinja','*.txt','*.model'])
print('Downloading trading-dora-adapter-v2 (1.9 GB)...')
snapshot_download('bverbeck/trading-dora-adapter-v2', local_dir='/workspace/trading-dora-adapter-v2',
    token=os.environ['HF_TOKEN'])
"

# CPU merge base + adapter → /workspace/Jarvis_27B_trading_raw/
python3 merge_adapter_qwen27_v2.py

# Minimal post-merge: inject MTP weights from base, strip mrope_* from config.
# (Lighter than post_merge_qwen27_v2.py — doesn't need a Cycle 1 REF dir on the pod.)
python3 post_merge_minimal.py /workspace/Jarvis_27B_trading_raw /workspace/Qwen3.5-27B
```

Output: `/workspace/Jarvis_27B_trading_raw/` (54 GB merged, MTP weights injected back in via post_merge_minimal).

## Step 4: NVFP4 quantize — MAX QUALITY config (4x H200 recommended)

Upload `scripts/quantize_nvfp4_jarvis2.py` and ensure `training_data/` is mirrored to `/workspace/training_data/` (needed for stratified calibration sampling across the 65-book corpus).

Script defaults (cost-no-object quality config):
- **512 samples × 4096 seq_len** = ~2M tokens calibrated (matches reasoning+answer inference distribution)
- **Stratified sampling** across 65 books × 4 templates (no domain dominates)
- **AWQ-lite preprocessing** → NVFP4 quant (if modelopt has `NVFP4_AWQ_LITE_CFG`; falls back to NVFP4_DEFAULT_CFG with `--no-awq`)
- **Layer exclusions**: first 2 + last 2 + MTP head kept in higher precision (most quant-sensitive)
- **Multi-GPU ready**: `device_map='auto'` splits across visible GPUs (4× H200 recommended for parallel calibration)

```bash
# training_data/ was already uploaded + extracted in Step 3 (tarball)
python3 quantize_nvfp4_jarvis2.py \
    --input /workspace/Jarvis_27B_trading_raw \
    --output /workspace/Jarvis_2 \
    --training-data-dir /workspace/training_data
# Script uses 512 × 4096 by default with AWQ-lite + layer exclusions.
# Explicit override: --num-calib-samples 256 --seq-len 2048 --no-awq
```

Expected timing on 4× H200 NVL (143 GB each):
- Calibration (512 × 4096 forward passes, tensor-parallel): ~20-30 min
- Weight quantization (sequential per-layer): ~30-45 min
- Save: ~3 min
- **Total: ~1-1.5 hours compute**

Output: `/workspace/Jarvis_2/` (~19 GB NVFP4, MTP+edge layers kept BF16).

## Step 5: Upload to HF

```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.create_repo('bverbeck/jarvis-2-nvfp4', private=True, exist_ok=True)
api.upload_folder(
    folder_path='/workspace/Jarvis_2',
    repo_id='bverbeck/jarvis-2-nvfp4',
    commit_message='Jarvis_2 NVFP4 — Qwen3.5-27B Cycle 2 DoRA merged, MTP preserved'
)
"
```

Upload ~19 GB → HF: ~20-40 min on typical pod uplink.

## Step 6: Back on Spark — pull + serve

```bash
# On Spark
cd ~/models
hf_TOKEN=... python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('bverbeck/jarvis-2-nvfp4', local_dir='/home/brandonv/models/Jarvis_2')
"

# Reuse the existing serve_jarvis_mtp.sh pattern — just change --model path to Jarvis_2
# Expected: serves at port 8000, ~17-20 tok/s with qwen3_next_mtp speculative decoding
```

## Step 7: Shootout Jarvis_2 vs existing

```bash
# From the trader repo:
python -m benchmarks benchmark \
    --label jarvis2-nvfp4-mtp \
    --phase shootout \
    --optimization 'NVFP4+MTP Cycle 2' \
    --base-url http://localhost:8000 \
    --model /home/brandonv/models/Jarvis_2

python scripts/shootout_harvest.py \
    --label jarvis2-nvfp4-mtp-fair4k \
    --max-tokens 4096 --temperature 0.3 \
    --base-url http://localhost:8000 \
    --model /home/brandonv/models/Jarvis_2

python scripts/shootout_score.py \
    --reference benchmarks/results/2026-04-03_deepseek-reasoner \
    --candidates \
        benchmarks/results/2026-04-12_jarvis1-nvfp4-mtp-fair4k \
        benchmarks/results/2026-04-12_gemma4-q4km-llamacpp-fair4k \
        benchmarks/results/2026-04-12_qwenmoe-q5imatrix-llamacpp-fair4k \
        benchmarks/results/<new-jarvis2-dir> \
    --output benchmarks/results/shootout_scores_jarvis2.json
```

**Expected result**: Jarvis_2 ≥ 51/60 (matching or beating Jarvis_1 since same architecture + better Cycle 2 training) at ~17 tok/s.

## Step 8 (after Jarvis_2 validates): Eagle-3 head training

Separate RunPod job on H200. Use Jarvis_2 as target. Training corpus: `training_runs/train.json` (53k trading prompts). Budget ~$200-500, ~5-10 hours. See `project_eagle3_speculation.md` memory for approach.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| modelopt OOM during calibration | Calibration batch too big | Drop `--num-calib-samples` from 256 → 128 or `--seq-len` from 256 → 128 |
| `mtp.*` keys missing after quant | modelopt stripped them | Our script's `*mtp*: {"enable": False}` exclusion should prevent this — verify config applied |
| **Output size is ~54 GB not ~19 GB** | **modelopt 0.42+ breaks `model.save_pretrained()` for quantized weights** | **Must use `mte.export_hf_checkpoint(model, export_dir=path)` — already patched in `quantize_nvfp4_jarvis2.py`. If you hand-port the script, don't skip this.** |
| vLLM fails to load NVFP4 on Spark | Config not VLM wrapper format | Copy Jarvis_1's config.json structure (VLM wrapper with text_config), rename weight keys if needed (see `scripts/post_merge_qwen27_v2.py`) |
| Speed below 17 tok/s | MTP not activating | Verify serve flag `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}'` and MTP weights in merged model (grep `mtp.` in index.json) |
| `transformers version 5.5.3 is not tested with nvidia-modelopt` warning | modelopt only tests against older transformers | Ignore — works fine in practice for Qwen3.5 quant path. Don't downgrade transformers; Qwen3.5 needs 5.5+. |
| flash-attn compile fails on pod during install | Network / CUDA version mismatch | Skip it. modelopt calibration falls back to SDPA, ~no perf impact for a one-shot quant job. |
| `fla: "The fast path is not available"` warning during calibration | flash-linear-attention not installed | Safe to ignore for quant-only workflow. Calibration forward passes use torch fallback, ~2x slower but total adds only a few minutes on an H200. |

## Files involved

- `scripts/quantize_nvfp4_jarvis2.py` — the quant script, paths already set for Jarvis_2
- `scripts/merge_adapter_qwen27_v2.py` — merge (reuse with /workspace/ paths)
- `scripts/post_merge_qwen27_v2.py` — post-merge (reuse)
- `training_runs/train.json` — 53k trading prompts (upload subset for calibration)
