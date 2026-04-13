# Serving a Fine-Tuned Qwen3.5-27B (Dense, NVFP4) on DGX Spark

Notes from getting a DoRA fine-tuned Qwen3.5-27B dense model, quantized to NVFP4 via `nvidia-modelopt`, serving on a single DGX Spark. This took significant trial and error across multiple frameworks and configurations. Sharing what worked and what didn't so others don't have to repeat it.

Tested April 2026 on DGX Spark OS 7.4.0, driver 580.126.09, CUDA 13.0.

## 2026-04-12 Update — Multi-model serving + quality shootout

What was originally about Jarvis_1 (single-model NVFP4 serving) has grown into a full comparison across 3 serving stacks and 3 model architectures on the same Spark, with a blind-scored quality shootout. See:

- **[docs/2026-04-12-shootout-results.md](docs/2026-04-12-shootout-results.md)** — head-to-head quality + speed comparison across 21 prompts, blind-scored by Claude Opus against a DeepSeek-Reasoner reference. Punchline: Jarvis_1 (Cycle 1 NVFP4, 27B dense) wins quality at **51.0/60** despite being the slowest (17 tok/s); llama.cpp-served Qwen MoE Q5+imatrix scored 46.1 at 56 tok/s, Q4_K_M MoE 44.9 at 72 tok/s, Gemma 4 Q4_K_M 42.0 at 62 tok/s. Quality gap is structural (MoE active-param capacity), not quantization.
- **[docs/jarvis-serving-runbook.md](docs/jarvis-serving-runbook.md)** — current deployment runbook. Covers vLLM for NVFP4 27B, llama.cpp for MoE and Gemma 4 (GGUF Q4_K_M / Q5_K_M+imatrix) — the NVIDIA-official DGX Spark llama.cpp path that produces 60-74 tok/s on MoE + Gemma.
- **[docs/jarvis2-nvfp4-runbook.md](docs/jarvis2-nvfp4-runbook.md)** — RunPod pipeline for quantizing a Cycle 2 27B merge to NVFP4 (Jarvis_2). Documents the `mte.export_hf_checkpoint()` vs `model.save_pretrained()` gotcha (modelopt 0.42 silently drops quant state), AWQ-lite preprocessing, layer-exclusion policy, stratified calibration, 512×4096 sample/seq_len tuning.
- **[scripts/](scripts/)** — merge scripts (27B dense, Qwen MoE, Gemma 4), post-merge tools (MTP injection, key rename, config flattening, `post_merge_minimal.py` for fresh pods), quantization entrypoints, llama-server and vLLM serve scripts, `shootout_harvest.py` + `shootout_score.py` for blind evaluation.
- **[benchmarks/](benchmarks/)** — 21 trading/decision prompts, full streaming client, stats aggregation, and all 2026-04-12 results (completions, scoring JSON). Reproducible against your own fine-tunes by pointing the scripts at your endpoints.

What DIDN'T change: everything below is still the canonical Jarvis_1 writeup from the first Spark NVFP4 deploy in April. The new work extends rather than replaces it.

## 2026-04-12 key learnings (added to the list of gotchas at the bottom)

1. **modelopt 0.42's `model.save_pretrained()` silently drops the quant state.** The saved model is BF16 (54 GB), not NVFP4 (~19 GB). Use `mte.export_hf_checkpoint(model, export_dir=path)` instead. We ate one 45-min pod run on this. Memory of the fix existed for Jarvis_1 (April 1) but wasn't consulted when writing the Jarvis_2 script.
2. **Qwen3.5-35B-A3B MoE has no shipped MTP weights.** The config declares `mtp_num_hidden_layers: 1` but the HF release doesn't ship them. vLLM + `qwen3_next_mtp` is not an option for MoE without training a drafter head.
3. **Qwen3.5-35B-A3B MoE needs `mrope_section` + `mrope_interleaved` in config for llama.cpp GGUF quantization** (else `key not found: qwen35moe.rope.dimension_sections`), but vLLM rejects those same fields with an M-RoPE assertion. If you flip between stacks, strip them for vLLM / restore for llama.cpp.
4. **Gemma 4 (`gemma4` model_type, transformers 5.5+)** isn't in vLLM as of 0.18.2rc1. llama.cpp is the only production option on Spark right now. Llama.cpp's ARM64 CUDA 13 build (NVIDIA-official DGX Spark guide) gets ~67 tok/s on Gemma 4 26B-A4B at Q4_K_M.
5. **CPU merge is fine on Spark (contradicts earlier "RunPod for all heavy lifting" memory).** 35B MoE DoRA merge runs in ~90 seconds with `device_map='cpu'`, bypasses the CUDA VM reservation storm that kills `device_map='auto'` on unified memory. Use CPU path on Spark for merges; keep RunPod for quantization (modelopt does OOM the Spark).
6. **Training loss is not a transfer-quality predictor on this task set.** Gemma's adapter hit loss 0.261 (lowest of three models) and scored **last** in the shootout (42.0/60). Jarvis_1's adapter hit loss 0.967 (highest) and won **first** (51.0). Loss is fit-to-training-distribution; shootout measured transfer to novel trading prompts. Pick target model by architecture + quant + data diversity, not by final_loss.

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
- **Performance (baseline)**: ~11.2 tok/s decode at batch 1, ~55 GB memory used, ~66 GB free
- **Performance (MTP enabled)**: ~17.1 tok/s decode at batch 1 — **52% speedup**, no quality degradation (see [MTP](#mtp-multi-token-prediction-speculative-decoding))
- **Thinking mode**: Qwen3.5's native thinking mode works via `chat_template_kwargs` (see [Thinking Mode](#thinking-mode-qwen35s-native-chain-of-thought))

## Verified Benchmarks

Independently measured on a single DGX Spark, April 2026.

### Throughput

| Configuration | tok/s | Memory | Notes |
|--------------|-------|--------|-------|
| **NVFP4 baseline** (`--enforce-eager`) | 11.2 | ~55 GB | Stable across generation lengths |
| **NVFP4 + MTP** (speculative decoding) | **17.1** | ~59 GB | **52% faster**, no quality loss |

MTP throughput is consistent across 20 long-form completions (1,000–7,000 tokens each). The +4 GB memory overhead comes from the BF16 MTP draft head (~811 MB) plus additional KV cache for speculative tokens.

### Quality (Fine-Tuned Model)

We benchmarked our DoRA fine-tuned model (trained on 64 trading/finance/decision-science books) against DeepSeek Reasoner (`deepseek-reasoner`) using a 20-prompt evaluation suite covering market analysis, strategy, risk management, decision science, statistical reasoning, behavioral finance, and general reasoning. Each completion was scored 1-5 on five criteria: framework application, concrete parameters, conflict acknowledgment, reasoning depth, and proportional confidence.

| Model | Overall (all 20) | Excl. failures | Zero-answer failures | Perfect 5.0 scores |
|-------|-----------------|----------------|---------------------|-------------------|
| **Jarvis (NVFP4 + MTP)** | **4.71** | **4.91** | 1 | **11** |
| **Jarvis (NVFP4 baseline)** | 4.54 | 4.73 | 1 | 6 |
| DeepSeek Reasoner (API) | 3.95 | 4.38 | 4 | 3 |

The fine-tuned 27B model outperformed DeepSeek Reasoner on 11 of 15 comparable prompts. The largest gap was in framework application (+0.73) — the fine-tuned model consistently names and correctly applies specific analytical frameworks from its training corpus, while DeepSeek knows the concepts but applies them more generically. MTP did not degrade quality on any dimension.

*These scores reflect domain-specific evaluation criteria. Results will vary by use case. DeepSeek Reasoner is a strong general-purpose model being compared on domain-specific prompts that favor fine-tuned knowledge.*

### Community Context

As of April 2026, no other verified NVFP4 tok/s measurements for Qwen3.5-27B **dense** on a single DGX Spark exist in public forums or repositories. The closest data points:

| Source | Format | tok/s | Type |
|--------|--------|-------|------|
| [NVIDIA Forum (cho)](https://forums.developer.nvidia.com/t/how-fast-can-qwen3-5-27b-be-after-converting-to-nvfp4/362776) | NVFP4 (theoretical) | ~20.2 | Math only |
| [NVIDIA Forum (joshua.dale.warner)](https://forums.developer.nvidia.com/t/how-fast-can-qwen3-5-27b-be-after-converting-to-nvfp4/362776) | INT4 | ~12 | Measured |
| [NVIDIA Forum (josephbreda)](https://forums.developer.nvidia.com/t/how-fast-can-qwen3-5-27b-be-after-converting-to-nvfp4/362776) | NVFP4 | "no faster than FP8" | Vague, no number |
| [NVIDIA Forum](https://forums.developer.nvidia.com/t/run-qwen3-5-27b-with-spark-vllm-docker/362563) | BF16 | ~4 | Measured |
| **This repo (baseline)** | **NVFP4 (fine-tuned)** | **11.2** | **Measured** |
| **This repo (MTP)** | **NVFP4 + MTP** | **17.1** | **Measured** |

Our MTP result of 17.1 tok/s represents ~84% of the theoretical bandwidth peak (~20.2 tok/s), a significant improvement over the baseline's ~56%. This appears to be the first verified MTP-enabled dense Qwen3.5-27B measurement on a DGX Spark.

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
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --host 0.0.0.0 --port 8000
```

**Key flags:**
- `--language-model-only`: Required. ModelOpt exports weights with `model.language_model.` prefix. This flag strips the prefix and skips vision encoder loading.
- `--gpu-memory-utilization 0.40`: Conservative for UMA. 0.40 × 121 GB = ~48 GB budget. With the 19 GB model + MTP head, this leaves room for KV cache and system overhead.
- `--enforce-eager`: Disables CUDAGraph compilation. Safe default for initial setup.
- `--kv-cache-dtype fp8`: Reduces KV cache memory footprint.
- `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`: Enables MTP speculative decoding using the model's native BF16 MTP head. See [MTP section](#mtp-multi-token-prediction-speculative-decoding) for details.
- `--max-model-len 8192`: Increased from 4096 to support longer completions when thinking mode is active.

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

## MTP (Multi-Token Prediction / Speculative Decoding)

**MTP is working and recommended.** It provides a 52% throughput improvement with no measurable quality degradation.

Qwen3.5-27B includes a native MTP head (15 tensors, BF16) that enables speculative decoding — the draft head predicts the next token while the main model verifies, allowing multiple tokens per forward pass.

### How to Enable

Add this flag to your serve command:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

That's it. The MTP head must be present in your checkpoint (it is if you used `export_hf_checkpoint()` with the default config — only if you explicitly excluded MTP tensors during quantization would they be missing).

### Verified Results

| Metric | Baseline | MTP Enabled | Delta |
|--------|----------|-------------|-------|
| Throughput | 11.2 tok/s | **17.1 tok/s** | **+52%** |
| Memory | ~55 GB | ~59 GB | +4 GB |
| TTFT | ~175 ms | ~175 ms | No change |

Measured across 20 long-form completions (1,000–7,000 tokens each) on a single DGX Spark. Throughput is consistent — no outlier prompts showed degradation.

### Why It Works (Mixed Precision)

The main model runs NVFP4 while the MTP draft head runs BF16. This mixed-precision speculative decoding works on the eugr FLASHINFER_CUTLASS backend because the draft head is a small, separate forward pass that doesn't interact with the NVFP4 GEMM kernels. The draft head is ~811 MB — negligible overhead on the Spark's 121 GB memory pool.

### Gotcha: MTP Head Must Be BF16

During quantization with `modelopt`, exclude the MTP head from NVFP4 quantization:

```python
config = mtq.NVFP4_DEFAULT_CFG.copy()
config["quant_cfg"]["*mtp*"] = {"enable": False}
```

If you quantize the MTP head to NVFP4, speculative decoding accuracy drops and throughput gains disappear (the verifier rejects most draft tokens). The head must remain BF16.

## Thinking Mode (Qwen3.5's Native Chain-of-Thought)

Qwen3.5 supports a native thinking mode where the model produces a chain-of-thought reasoning trace in a `<think>...</think>` block, separate from the final answer. When properly configured, vLLM routes the thinking content to a `reasoning_content` field in the streaming response, keeping the completion clean.

### How to Enable

**Client side** — pass `enable_thinking` in the request:

```python
payload = {
    "model": "/path/to/your/model",
    "messages": [{"role": "user", "content": "Your prompt here"}],
    "max_tokens": 4096,
    "temperature": 0,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": True},
}
```

**Server side** — the serve command needs `--enable-reasoning-content` (vLLM 0.18.2+):

```bash
exec ~/models/vllm-native/bin/python3 -u -m vllm.entrypoints.openai.api_server \
  ... \
  --enable-reasoning-content
```

### Streaming Response Format

When thinking mode is active, the SSE stream contains two types of delta content:

```python
# In each SSE chunk:
delta = choices[0]["delta"]

# Thinking content (chain-of-thought reasoning)
reasoning_text = delta.get("reasoning_content", "")

# Answer content (the actual completion)
answer_text = delta.get("content", "")
```

The reasoning tokens are generated first, followed by the answer tokens. TTFT for the *answer* content will be longer because the model thinks first — typically 5-30 seconds of reasoning before the first answer token appears.

### Why This Matters

Without thinking mode properly configured, Qwen3.5 still thinks — but it emits the reasoning as free-form text in the regular completion stream. You get output like:

```
Here's a thinking process that leads to the suggested response:

1. **Deconstruct the Request:** ...
2. **Analyze:** ...
...
</think>

[actual answer buried at the end]
```

This wastes most of your `max_tokens` budget on scratchpad content. With `enable_thinking`, the reasoning is cleanly separated and doesn't count against `max_tokens` for the answer.

### Gotcha: `reasoning_content` vs `reasoning`

Some vLLM versions use `reasoning_content` in the delta, others use `reasoning`. Check your version. The field name in the streaming delta must match what your client code reads, or the thinking content silently goes to the wrong place (empty reasoning field, scratchpad leaks into completion).

## Notes

- This was tested with a DoRA fine-tuned Qwen3.5-27B, but the config fixes likely apply to any Qwen3.5 model quantized through `modelopt export_hf_checkpoint()`. Stock NVFP4 models from NVIDIA or the community (e.g., [osoleve/Qwen3.5-27B-Text-NVFP4-MTP](https://huggingface.co/osoleve/Qwen3.5-27B-Text-NVFP4-MTP)) already have the correct config format and shouldn't need these fixes.
- Dense 27B baseline performance (~11 tok/s) is much lower than MoE benchmarks you'll see online (30-60+ tok/s). This is expected — dense models are bandwidth-bound on the Spark. MTP closes the gap significantly (~17 tok/s).
- The fine-tuning was done with DoRA on 64 trading/finance/decision-science books. The quality benchmarks reflect domain-specific evaluation — your results will depend on your fine-tuning data and evaluation criteria.
- These notes may become outdated as vLLM, TRT-LLM, and the eugr wheels evolve. Check the dates and test on your setup.
- We are not affiliated with any of the projects mentioned. Just a user sharing what worked.

## Acknowledgments

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — prebuilt aarch64 vLLM wheels with SM121 patches (updated daily)
- [Avarok-Cybersecurity/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm) — original SM121 NVFP4 software E2M1 patch
- [osoleve/Qwen3.5-27B-Text-NVFP4-MTP](https://huggingface.co/osoleve/Qwen3.5-27B-Text-NVFP4-MTP) — reference working config that helped us identify the VLM wrapper requirement
- [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) — official guides
- The DGX Spark community on NVIDIA Developer Forums
