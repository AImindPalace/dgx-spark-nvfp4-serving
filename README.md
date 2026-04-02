# Serving a Fine-Tuned Qwen3.5-27B (Dense, NVFP4) on DGX Spark

Notes from getting a DoRA fine-tuned Qwen3.5-27B dense model, quantized to NVFP4 via `nvidia-modelopt`, serving on a single DGX Spark. This took significant trial and error across multiple frameworks and configurations. Sharing what worked and what didn't so others don't have to repeat it.

Tested April 2026 on DGX Spark OS 7.4.0, driver 580.126.09, CUDA 13.0.

## Why This Is Hard

Two things make this specific combination harder than most Spark inference setups:

### DGX Spark Unified Memory (UMA)
The Spark's GB10 shares 128 GB between CPU and GPU. Most inference frameworks call `torch.cuda.mem_get_info()` to determine available GPU memory — on UMA, this returns the **entire system memory pool** (~121 GB usable), leading frameworks to over-allocate and freeze the system. The Spark doesn't throw a CUDA OOM; it enters an unrecoverable `nvidia-modeset` D-state that requires a physical reboot.

### Qwen3.5-27B Dense Architecture
Qwen3.5 is not a standard transformer. It has:
- **64 layers**: 48 GDN linear attention + 16 full attention (alternating 3:1 pattern)
- **`model_type: "qwen3_5"`** — a VLM wrapper architecture (`Qwen3_5ForConditionalGeneration`), even for text-only use
- **MTP (Multi-Token Prediction)** head for speculative decoding
- **SharedFusedMoE** for MLPs, even in the dense model

This causes specific issues:
- The Marlin NVFP4 backend can't handle GDN layers (tile-size constraint: `size_n=96` is not divisible by 64)
- TRT-LLM doesn't support `input_scale` tensors in GDN linear attention layers yet
- The VLM wrapper config format is **required** by vLLM's model registry, even for text-only serving
- SM121 (GB10) is missing the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction needed for native NVFP4 float-to-E2M1 conversion. The [eugr prebuilt wheels](https://github.com/eugr/spark-vllm-docker) include a software E2M1 patch (originally from [Avarok](https://github.com/Avarok-Cybersecurity/dgx-vllm)) that works around this.

### Dense vs MoE Performance
Most Spark NVFP4 benchmarks are for MoE models (Qwen3.5-35B-A3B, etc.) which only activate a fraction of parameters per token, getting 30-60+ tok/s. A **dense** 27B model activates all parameters every token and is bandwidth-bound on the Spark's 273 GB/s. Expect ~10 tok/s, not 30+.

## The Setup That Works

- **Framework**: vLLM 0.18.2+ via [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) prebuilt aarch64 wheels (native, no Docker)
- **Backend**: FLASHINFER_CUTLASS (auto-selected by vLLM, SM121 E2M1 patch built into eugr wheels)
- **PyTorch**: Nightly from `https://download.pytorch.org/whl/nightly/cu130`
- **Transformers**: 5.x+ (required for `qwen3_5` model type recognition)
- **Performance**: ~11 tok/s decode at batch 1, ~55 GB memory used, ~66 GB free

## Full Path: Fine-Tuned Model → Serving on Spark

### Step 1: Quantize with ModelOpt (on RunPod or similar)

This happens on a GPU with enough VRAM to hold the full BF16 model (51 GB). We used a RunPod B200.

```python
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

# After merging your DoRA/LoRA adapter:
# 1. Fix architectures (PEFT merge strips this)
model.config.architectures = ["Qwen3_5ForConditionalGeneration"]

# 2. Quantize (exclude MTP head)
config = mtq.NVFP4_DEFAULT_CFG.copy()
config["quant_cfg"]["*mtp*"] = {"enable": False}
mtq.quantize(model, config, forward_loop=calibration_fn)

# 3. Export — use export_hf_checkpoint, NOT save_pretrained
#    save_pretrained creates a 53 GB fake-quantized BF16 file (wrong)
#    export_hf_checkpoint creates a 19 GB packed NVFP4 file (correct)
export_hf_checkpoint(model, output_dir="./Jarvis_1")
```

Output: ~19 GB checkpoint with `hf_quant_config.json` (`quant_algo: "NVFP4"`).

### Step 2: Fix Config Files (Critical)

ModelOpt's export creates config files that don't work as-is with vLLM on the Spark. **These fixes are the difference between a 19 GB model and a 112 GB dequantized model that freezes your Spark.**

Run both scripts on your exported checkpoint:

```bash
python config-fixes/fix_config.py ./your_model/
python config-fixes/fix_index.py ./your_model/
```

See [Config Fixes](#config-fixes) below for details on what these fix and why.

### Step 3: Set Up the Spark Environment

```bash
# System packages (one-time)
sudo apt install -y cuda-toolkit-13-2 cuda-compat-13-2 python3-dev libcudnn9-cuda-13 libnccl2
sudo apt install -y earlyoom dropbear  # safety nets — see "Protecting Your Spark" below

# Python environment
python3 -m venv ~/models/vllm-native
source ~/models/vllm-native/bin/activate

# PyTorch with CUDA 13 (aarch64)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# eugr prebuilt vLLM wheels — check https://github.com/eugr/spark-vllm-docker/releases for latest
# Download the 4 wheel files, then:
pip install ./vllm-*.whl ./flashinfer_python-*.whl ./flashinfer_cubin-*.whl ./flashinfer_jit_cache-*.whl

# Upgrade transformers (eugr bundles 4.57.x, we need 5.x for qwen3_5 model type)
pip install 'transformers>=5.0'
```

**Why `cuda-compat-13-2`?** The eugr wheels are compiled for CUDA 13.2 but the Spark driver (580.x) natively supports CUDA 13.0. The compat package bridges this gap — same thing Docker's NVIDIA container toolkit does automatically.

**Why `transformers>=5.0`?** The `qwen3_5` and `qwen3_5_text` model types were added in transformers 5.x. Without it, vLLM falls back to `qwen3_next` which causes FP32 dequantization.

### Step 4: Serve

```bash
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas

exec ~/models/vllm-native/bin/python3 -u -m vllm.entrypoints.openai.api_server \
  --model ~/models/YOUR_MODEL \
  --quantization modelopt \
  --enforce-eager \
  --gpu-memory-utilization 0.40 \
  --max-model-len 4096 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --language-model-only \
  --host 0.0.0.0 --port 8000
```

**Key flags:**
- `--language-model-only`: Required. ModelOpt exports weights with `model.language_model.` prefix. This flag strips the prefix and skips vision encoder loading.
- `--gpu-memory-utilization 0.40`: Conservative for UMA. 0.40 × 121 GB = ~48 GB budget. With the 19 GB model, this leaves room for KV cache and system overhead.
- `--enforce-eager`: Disables CUDAGraph compilation. Safe default for initial setup.
- `--kv-cache-dtype fp8`: Reduces KV cache memory footprint.

**Do NOT set `VLLM_NVFP4_GEMM_BACKEND`.** Let vLLM auto-select FLASHINFER_CUTLASS. The eugr wheels include the SM121 software E2M1 patch. Forcing Marlin causes `size_n=96 not divisible by tile_n_size=64` on Qwen3.5's GDN linear attention layers.

### Step 5: Verify

```bash
# Health check
curl http://localhost:8000/health

# Test inference
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"~/models/YOUR_MODEL","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

You should see ~55 GB memory usage and the model responding. If memory spikes to 100+ GB, the config fixes weren't applied correctly.

## Config Fixes

### 1. `config.json` — VLM Wrapper Format

ModelOpt's `export_hf_checkpoint()` may produce a flattened config, or you may have manually changed `model_type` to `"qwen3_next"` based on other guides. Both cause vLLM to dequantize NVFP4 weights to FP32.

**The config.json must use the original Qwen3.5 VLM wrapper structure:**

```json
{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text",
    "...all model parameters..."
  },
  "vision_config": {
    "model_type": "qwen3_5",
    "...standard Qwen3.5 vision config..."
  },
  "quantization_config": {
    "...your existing quantization_config unchanged..."
  }
}
```

**Why each `model_type` fails:**
| model_type | What happens | Why |
|---|---|---|
| `"qwen3_next"` | 112 GB memory, system freeze | Weight loader doesn't recognize NVFP4 format, unpacks to FP32 |
| `"qwen3_5_text"` (flat) | `TypeError: Expected Qwen3_5Config, got Qwen3_5TextConfig` | vLLM routes through VLM path, needs wrapper config |
| **`"qwen3_5"` (wrapper)** | **19 GB, works** | **vLLM correctly identifies NVFP4, keeps weights packed** |

The `vision_config` section is required even for text-only serving. vLLM's model registry maps `Qwen3_5ForConditionalGeneration` through the VLM code path, which reads `vision_config.spatial_merge_size` during initialization. The `--language-model-only` flag then tells it to skip loading vision weights.

Run `python config-fixes/fix_config.py ./your_model/` to apply this fix automatically. See [`config-fixes/config.json`](config-fixes/config.json) for a complete working example.

### 2. `model.safetensors.index.json` — Shard References

ModelOpt's export may create an index file referencing shard filenames (`model-00001-of-00002.safetensors`, etc.) while the actual output is a single `model.safetensors` file. This causes `RuntimeError: Cannot find any model weights`.

Run `python config-fixes/fix_index.py ./your_model/` to fix this automatically.

## Protecting Your Spark

The Spark can enter an unrecoverable D-state if a process allocates too much memory on UMA. Install these before experimenting:

**earlyoom** — kills processes before the system freezes:
```bash
sudo apt install -y earlyoom
# /etc/default/earlyoom:
EARLYOOM_ARGS="-m 3 -s 10 --avoid '(^|/)(ssh|sshd|systemd|dropbear)$' --prefer '(^|/)(vllm|python3.*vllm)$' -r 1"
sudo systemctl restart earlyoom
```

**Dropbear SSH** — lightweight SSH server (500 KB) that stays responsive when OpenSSH can't:
```bash
sudo apt install -y dropbear
# /etc/default/dropbear:
DROPBEAR_PORT=2222
sudo systemctl restart dropbear
```

If your main SSH becomes unresponsive during model loading, connect via `ssh -p 2222 user@spark` and kill the process. This saved us from several physical reboots.

**Always flush page cache before loading a model:**
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## Things That Didn't Work

Documenting these with specific error messages so they're searchable.

| Approach | Error / Symptom | Root Cause |
|----------|----------------|------------|
| `model_type: "qwen3_next"` in config.json | 121 GB memory usage, system freeze | Weight loader dequantizes NVFP4 → FP32 |
| `model_type: "qwen3_5_text"` (flat config, no wrapper) | `TypeError: Expected Qwen3_5Config, got Qwen3_5TextConfig` | vLLM routes Qwen3_5ForCausalLM through VLM path, needs wrapper |
| `VLLM_NVFP4_GEMM_BACKEND=marlin` | `RuntimeError: size_n = 96 is not divisible by tile_n_size = 64` | Marlin tile constraint doesn't fit Qwen3.5 GDN linear attention layer dimensions |
| pip install vLLM 0.18.1 (aarch64 wheel) | `ImportError: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib` | Wheel compiled against CUDA 12 + torch 2.10.0, Spark needs CUDA 13 |
| TRT-LLM 1.3.0rc10 | `NotImplementedError: unsupported suffix 'input_scale' for model.layers.0.linear_attn` | TRT-LLM doesn't handle NVFP4 activation scales in Qwen3.5 GDN linear attention |
| TRT-LLM spark-single-gpu-dev tag | `ValueError: model type 'qwen3_5' not recognized` | Older transformers bundled in container doesn't know Qwen3.5 |
| Docker with `gpu_memory_utilization > 0.40` | System freeze (nvidia-modeset D-state) | UMA: `torch.cuda.mem_get_info()` reports full system RAM as GPU memory |
| `--language-model-only` with stale index.json | `RuntimeError: Cannot find any model weights` | Index referenced nonexistent shard filenames from modelopt export |

## Notes

- This was tested with a DoRA fine-tuned Qwen3.5-27B, but the config fixes likely apply to any Qwen3.5 model quantized through `modelopt export_hf_checkpoint()`. Stock NVFP4 models from NVIDIA or the community (e.g., [osoleve/Qwen3.5-27B-Text-NVFP4-MTP](https://huggingface.co/osoleve/Qwen3.5-27B-Text-NVFP4-MTP)) already have the correct config format and shouldn't need these fixes.
- Dense 27B performance (~11 tok/s) is much lower than MoE benchmarks you'll see online (30-60+ tok/s). This is expected — dense models are bandwidth-bound on the Spark.
- These notes may become outdated as vLLM, TRT-LLM, and the eugr wheels evolve. Check the dates and test on your setup.
- We are not affiliated with any of the projects mentioned. Just a user sharing what worked.

## Acknowledgments

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — prebuilt aarch64 vLLM wheels with SM121 patches (updated daily)
- [Avarok-Cybersecurity/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm) — original SM121 NVFP4 software E2M1 patch
- [osoleve/Qwen3.5-27B-Text-NVFP4-MTP](https://huggingface.co/osoleve/Qwen3.5-27B-Text-NVFP4-MTP) — reference working config that helped us identify the VLM wrapper requirement
- [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) — official guides
- The DGX Spark community on NVIDIA Developer Forums
