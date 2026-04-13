# Jarvis Serving Runbook (Living Doc)

> This is an iterative document. Update as we learn and improve.

## Serving Models at a Glance

| Model | Base | Format | Size | Stack | Port | Speed | Notes |
|-------|------|--------|------|-------|------|-------|-------|
| **Jarvis_Gemma_trading** | Gemma 4 26B-A4B + Cycle 2 DoRA | Q4_K_M GGUF | 16 GB | **llama.cpp** | 30000 | **67 tok/s** | Fastest — NVIDIA-official DGX Spark stack |
| **Jarvis_MoE_trading (llama.cpp)** | Qwen3.5-35B-A3B MoE + Cycle 2 DoRA | Q4_K_M GGUF | 20 GB | **llama.cpp** | 30001 | **74 tok/s** | **2.5x over vLLM BF16** — current recommended MoE serve |
| Jarvis_27B_trading | Qwen3.5-27B dense + Cycle 2 DoRA (v2) | BF16 | 54 GB | vLLM | 8000 | ~6.8 tok/s (MTP) | Memory-bandwidth-bound; keep on vLLM for MTP |
| Jarvis_1 | Qwen3.5-27B dense + Cycle 1 DoRA | NVFP4 | 19 GB | vLLM | 8000 | ~17 tok/s (MTP) | Legacy NVFP4 baseline |
| Jarvis_MoE_trading (vLLM) | same model, BF16 serve | BF16 | 65 GB | vLLM | 8000 | ~30 tok/s | Legacy — superseded by llama.cpp variant |
| Jarvis_MoE | Qwen3.5-35B-A3B MoE base (pre-merge) | BF16 | 65 GB | vLLM | 8000 | ~30 tok/s | Rollback baseline |

**Port allocations**: llama.cpp serves use 30000/30001; vLLM uses 8000. All can coexist on Spark with total memory allowing. The llama.cpp stack for MoE+Gemma uses ~45 GB combined, leaving headroom for the vLLM 27B if you want all three live simultaneously.

**Why llama.cpp won on Spark for MoE + Gemma**:
- Official NVIDIA/Google optimization for GB10 SM_121 + ARM64 CUDA 13
- Q4_K_M quant halves memory bandwidth vs BF16 → 2-3x throughput on Spark's 273 GB/s LPDDR5x
- Mature MoE expert kernels (FlashInfer CUTLASS MoE via llama.cpp was faster per community benches)
- Hot-swap LoRA via GGUF means future adapter retrains skip the full merge step (not yet used — current deploy is merged, but it's a future option)
- vLLM still wins for 27B dense MTP (llama.cpp has no `qwen3_next_mtp` spec decode path yet)

## Current State (2026-04-12 PM): llama.cpp primary, vLLM for 27B MTP only

### llama.cpp serves (primary)
- **Gemma 4 trading**: port 30000, Q4_K_M, 67 tok/s
- **Qwen MoE trading**: port 30001, Q4_K_M, 74 tok/s
- Both expose `/v1/chat/completions` (OpenAI-compatible) and auto-split `<think>...</think>` into `reasoning_content` + `content` natively (no `--reasoning-parser` flag needed — llama.cpp does it by default when `--jinja` is set and the chat template uses `<think>` tags)

### vLLM serve (secondary, 27B dense MTP only)
- **27B trading w/ MTP**: port 8000, BF16, ~6.8 tok/s, `Qwen3NextMTP` arch + `qwen3_next_mtp` spec decode + `--reasoning-parser qwen3`
- Uses VLM wrapper config + `--language-model-only` (see 3b below)

### Legacy / rollback
- **Jarvis_1** (NVFP4 27B): still runnable via `~/models/serve_jarvis_mtp.sh`, legacy quant-quality reference
- **Jarvis_MoE** (base BF16 MoE): `~/models/serve_jarvis_moe.sh`, use to compare pre-finetune baseline

## Legacy Reference: Jarvis_1 (~17 tok/s with MTP)
- **Model**: Jarvis_1 (Qwen3.5-27B, NVFP4, 19 GB)
- **Framework**: vLLM 0.18.2rc1 (eugr prebuilt wheels, native — no Docker)
- **Backend**: FLASHINFER_CUTLASS (SM121 patched)
- **Attention**: FlashInfer (required for SM121 — FlashAttention v2 PTX not compiled for GB10)
- **MTP**: `qwen3_next_mtp` with `num_speculative_tokens=1` (~1.5x speedup)
- **Memory**: ~18.55 GiB model + drafter, ~24 GiB KV cache available
- **Endpoint**: `http://localhost:8000/v1/chat/completions` (when running)

## How to Start

```bash
# Flush cache (recommended before first start after reboot)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Start serving
nohup ~/models/serve_jarvis.sh > ~/models/vllm_serve.log 2>&1 &

# Check health
curl http://localhost:8000/health

# Test inference
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/home/brandonv/models/Jarvis_1","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## How to Stop

```bash
pkill -f 'vllm.entrypoints'
```

## The Stack (what's installed and why)

| Component | Location | Why |
|-----------|----------|-----|
| vllm-native venv | `~/models/vllm-native/` | Isolated from UT daemon .venv |
| eugr vLLM wheel | 0.18.2rc1+cu132 | SM121 NVFP4 patches built in |
| eugr FlashInfer | 0.6.7 | SM121 E2M1 software conversion |
| torch nightly | 2.12.0.dev+cu130 | CUDA 13 aarch64 support |
| transformers | 5.4.0 | Required for `qwen3_5` model type |
| cuda-compat-13-2 | system package | Forward compat for cu132 wheels |
| earlyoom | system service | Kills runaway processes before zombie |
| Dropbear SSH | port 2222 | Emergency access during memory pressure |

## Config Files (CRITICAL — don't change without understanding)

### `~/models/Jarvis_1/config.json`
- **Must be VLM wrapper format**: `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`
- **Must have nested `text_config`** with `model_type: "qwen3_5_text"`
- **Must have `vision_config`** (even though we're text-only — vLLM routes through VLM code path)
- **Why**: Any other format causes vLLM to dequantize NVFP4 → FP32 (121 GB → zombie)

### `~/models/Jarvis_1/model.safetensors.index.json`
- All weight_map entries must point to `model.safetensors` (single file)
- **Why**: modelopt export created an index referencing nonexistent shard names

### `~/models/serve_jarvis.sh` (baseline, no MTP)
- `--language-model-only`: Required because weight keys have `model.language_model.` prefix
- `--gpu-memory-utilization 0.40`: 0.40 × 121 GB is safe. Higher = more KV cache but riskier
- `--enforce-eager`: Required — CUDA graphs + MTP corrupts output on SM121 (uncalibrated FP8 KV scales)
- No `VLLM_NVFP4_GEMM_BACKEND` override: Let vLLM auto-select FLASHINFER_CUTLASS

### `~/models/serve_jarvis_mtp.sh` (recommended, ~17 tok/s)
- All baseline flags plus:
- `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}'`: MTP speculative decoding
- Uses `qwen3_next_mtp` (not `mtp`) — handles BF16 MTP drafter natively, no eagle.py patch needed
- `--kv-cache-dtype fp8`: Required for FlashInfer on SM121 (FlashAttention v2 crashes)
- **Do NOT remove `--enforce-eager`** — CUDA graphs destroy quality with MTP on this setup

## Things That Went Wrong (so we don't repeat them)

| What | Symptom | Root Cause |
|------|---------|------------|
| `model_type: "qwen3_next"` | 121 GB memory, zombie | Weight loader dequants NVFP4 → FP32 |
| `VLLM_NVFP4_GEMM_BACKEND=marlin` | `size_n=96 not divisible by 64` | Marlin tile constraint on Qwen3.5 linear attn |
| Wrong index.json (shard names) | "Cannot find any model weights" | modelopt export created stale index |
| `model_type: "qwen3_5_text"` (flat) | vLLM expects VLM config | vLLM routes Qwen3_5ForCausalLM through VLM path |
| pip vLLM 0.18.1 | ABI mismatch | Built against CUDA 12 + torch 2.10, Spark has CUDA 13 |
| TRT-LLM 1.3.0rc10 | `input_scale` not supported | Linear attention NVFP4 tensors not handled yet |
| MTP + no `--enforce-eager` | All quality scores 1.0 (garbage) | Uncalibrated FP8 KV scales (1.0) + CUDA graph replay corrupts spec decode verification |
| MTP + no `--kv-cache-dtype fp8` | CUDA PTX unsupported toolchain crash | FlashAttention v2 PTX not compiled for SM121; only FlashInfer works on Spark |
| `VLLM_ATTENTION_BACKEND=FLASHINFER` (no fp8 kv) | Still selects FlashAttention v2 | Backend env var ignored; fp8 kv dtype is what constrains candidates to FlashInfer |
| `mtp` method + NVFP4 | Requires eagle.py patch | Generic MTP loader passes parent quant_config to drafter, shape mismatch on BF16 weights |

## Improvement Ideas (not yet tested)

### Performance
- [x] **MTP speculative decoding**: `qwen3_next_mtp` with `num_speculative_tokens=1` — **17.0 tok/s** (1.5x baseline)
- [x] **Remove `--enforce-eager`**: Tested — CUDA graphs add 0.4 tok/s alone (noise), destroy quality with MTP. **Keep enforce-eager.**
- [ ] **Increase `gpu-memory-utilization`**: 0.50 or 0.55 for more KV cache (test carefully)
- [ ] **Increase `max-model-len`**: Beyond 4096 for longer context
- [ ] **Benchmark different batch sizes**: `--max_batch_size` for UT daemon patterns
- [ ] **FP8 KV cache calibration**: Embed proper k_scale/v_scale in checkpoint — could unlock CUDA graphs + MTP
- [ ] **Eagle-3 draft head**: Train dedicated multi-token draft head for 25-40 tok/s (Jarvis_2)

### Stability
- [ ] **Systemd service**: Once stable, create `jarvis.service` for auto-start
- [ ] **Health monitoring**: Telegram alert if vLLM health check fails
- [ ] **Log rotation**: vllm_serve.log will grow without rotation

### Quality
- [ ] **Compare NVFP4 vs BF16 outputs**: Spot-check fine-tuned knowledge retention
- [ ] **Domain-specific eval**: Trading prompts from training data
- [ ] **Jarvis_2 QAT**: Quantization-aware fine-tuning for better NVFP4 quality

### Integration
- [ ] **Wire into UT daemon**: Update `ut_config.yaml` to use localhost:8000 as inference provider
- [ ] **Fallback chain**: If Jarvis is down, fall back to DeepSeek/Claude

## Updating the Stack

### eugr wheels (when new vLLM versions drop)
```bash
cd ~/models/vllm-native/wheels
# Check latest: https://github.com/eugr/spark-vllm-docker/releases
curl -sL -O <new_wheel_url>
~/models/vllm-native/bin/pip install ./new_wheel.whl
```

### transformers (if model format changes)
```bash
~/models/vllm-native/bin/pip install --upgrade transformers
```

## Emergency Recovery

If the Spark becomes unresponsive during model loading:
1. Try Dropbear: `ssh -p 2222 brandonv@10.0.0.85`
2. Kill the process: `pkill -f vllm` or `docker kill <container>`
3. If Dropbear also fails: physical reboot (hold power 10s)
4. After reboot: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` before retrying

---

# Adapter Deployment Pipeline (Reusable)

This is the playbook for taking a DoRA adapter on HF Hub → serving it locally on Spark. Proven for the 35B-A3B MoE on 2026-04-12 (~20 min total). Applies unchanged to 27B Cycle 2 once the adapter lands on HF.

## 1. Pull adapter + base to Spark
```bash
cd ~/models
git clone https://huggingface.co/bverbeck/<adapter-repo>
# For 35B MoE: trading-dora-adapter-qwen-moe
# For 27B Cycle 2: trading-dora-adapter-v2
```

Base model should already be on Spark (`Qwen3.5-27B/` or `Qwen3.5-35B-A3B-base/`).

## 2. CPU merge (Spark, ~2-5 min)
```bash
~/models/inference-venv/bin/python3 ~/models/merge_adapter_qwen_moe.py
# For 27B dense: adapt merge_adapter.py (earlier pattern, same CPU flow)
```

**Critical**: `device_map='cpu'` is non-negotiable.

- GPU device_map on Spark OOMs — CUDA VM reservation storm on unified memory (verified: 160 GB VMA for 65 GB model, kernel OOM at 97% load)
- CPU path: ~92s for 35B, ~85s for 27B. Peak RAM usage ~75 GB.
- Stop vLLM before merging (frees RAM). Can disable earlyoom for safety margin, but not strictly needed on CPU path.

Script template pattern (`~/models/merge_adapter_qwen_moe.py`):
```python
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map='cpu',
    low_cpu_mem_usage=True, trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUTPUT, safe_serialization=True, max_shard_size='4GB')
```

## 3. Post-merge — differs by model class

### 3a. MoE (Qwen3_5MoeForCausalLM) — key rename
When loading the MoE VLM wrapper base via `AutoModelForCausalLM`, transformers auto-extracts the text-only class but saves weights with `model.language_model.*` keys. vLLM's text-only MoE class (`Qwen3_5MoeForCausalLM`) expects `model.*`.

Run `scripts/post_merge_qwen_moe.py`:
- Rename `model.language_model.*` → `model.*` across all shards
- Rebuild `model.safetensors.index.json` with new keys
- Copy `config.json` (flat text-only) + chat template from known-good Jarvis_MoE (strips `mrope_*` fields)

Serve flag: **no `--language-model-only` needed** (MoE text class works natively).

### 3b. 27B dense (Qwen3_5ForCausalLM) — MTP injection + VLM wrapper config
vLLM's dense `Qwen3_5ForCausalLM` class **requires the VLM wrapper config** (same as Jarvis_1) even for text-only inference — flat `qwen3_5_text` config triggers `TypeError: Invalid type of HuggingFace config`. AND: transformers 5.x's `save_pretrained` **strips `mtp.*` weights** during merge (they're not part of the `Qwen3_5ForCausalLM` class definition), so MTP has to be re-injected manually from the base.

Run `scripts/post_merge_qwen27_v2.py`:
- Keep raw merge output's `model.language_model.*` weight keys as-is (DON'T rename for 27B)
- Extract `mtp.*` tensors from base Qwen3.5-27B shards, save as `model-mtp.safetensors` in the output dir
- Append MTP entries to `model.safetensors.index.json`
- Copy the **base Qwen3.5-27B VLM wrapper config** into the output dir (not Cycle 1's `qwen3_next` config — see gotchas below)
- Strip `rope_parameters.mrope_interleaved` + `rope_parameters.mrope_section` (vLLM M-RoPE assertion trigger)

Serve flags: **must include `--language-model-only`** (tells vLLM the `model.language_model.*` keys are the authoritative weights and to skip vision) **and** `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}'` (MTP ~1.5x speedup).

### Watch-outs
- **Do NOT use Cycle 1's `qwen3_next` config for BF16 serving.** `model_type: "qwen3_next"` is specific to the NVFP4 quantized serve path on Jarvis_1 — on BF16 it triggers unbounded memory growth during load (OOM at 97% of the reservation ceiling).
- **27B `gpu-memory-utilization` sweet spot is 0.70** for BF16 + MTP on Spark. Anything higher (0.80, 0.85, 0.90) OOMs during init because model weights + MTP drafter + CUDA workspace overflows the allocator reservation.
- **MoE has no MTP weights** in the HF release — skip `--speculative-config` for MoE serves. Training an Eagle-3 head for MoE is a separate ~$200-500 RunPod job (see `project_eagle3_speculation.md`).

## 4. Create serve script
Template: `~/models/serve_jarvis_moe_trading.sh`. Replace model path for each new adapter. Required flags:

```bash
exec ~/models/vllm-native/bin/python3 -u -m vllm.entrypoints.openai.api_server \
    --model /home/brandonv/models/<merged_dir> \
    --dtype bfloat16 \
    --trust-remote-code \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --host 0.0.0.0
```

## 5. Stop old vLLM, start new
```bash
pkill -9 -f 'vllm.entrypoints'
pkill -9 -f 'VLLM::EngineCore'  # lingering child — will hold RAM otherwise
pkill -9 -f 'multiprocessing.resource_tracker'
sleep 5
nohup bash ~/models/serve_jarvis_moe_trading.sh > ~/models/vllm_trading.log 2>&1 &
```

Load takes ~5-8 min for 35B MoE (21 shards). 27B dense should be faster (~3-4 min, fewer shards).

## 6. Smoke test via API
```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "/home/brandonv/models/<merged_dir>",
  "messages": [{"role": "user", "content": "<trading prompt>"}],
  "max_tokens": 2000
}' > ~/response.json
jq '.choices[0].message | {reasoning_len: (.reasoning // "" | length), content_len: (.content // "" | length)}' ~/response.json
```

Both `reasoning` and `content` fields should be populated.

---

# llama.cpp Deployment Pipeline (Gemma + Qwen MoE)

This is the NVIDIA-official path for DGX Spark (GB10, SM_121, ARM64, CUDA 13). Produces ~2-3x the throughput of vLLM for MoE/Gemma because of Spark-specific kernel optimization + Q4_K_M quantization halving memory bandwidth per token.

## One-time build (already done on Spark)

Per `https://build.nvidia.com/spark/llama-cpp/instructions`:

```bash
export PATH=/usr/local/cuda-13.2/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.2
cd ~/models && git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" -DLLAMA_CURL=OFF
make -j8    # ~5-10 min
```

Produces `~/models/llama.cpp/build/bin/{llama-server, llama-cli, llama-quantize}`. `convert_hf_to_gguf.py` + `convert_lora_to_gguf.py` live in the repo root.

## Per-adapter deployment flow

### 1. Download base + adapter (if not already on Spark)
Use `huggingface_hub.snapshot_download` with `HF_TOKEN` — not git+lfs (git-lfs isn't installed on Spark).

### 2. Merge adapter into base (CPU merge — same as vLLM path)
For Gemma 4: `scripts/merge_adapter_gemma.py`. Critical Gemma specifics:
- `attn_implementation='sdpa'` — Gemma 4 head_dim > 256, exceeds flash-attn v2 limit
- Unwrap `Gemma4ClippableLinear` → `nn.Linear` before PEFT (matches training; clamp bounds would skew merge math)
- Requires **transformers 5.5+** (gemma4 model class wasn't in 5.4.0)

For Qwen MoE: `scripts/merge_adapter_qwen_moe.py` — unchanged, same pattern as for vLLM deploy.

### 3. Convert merged HF → F16 GGUF
```bash
~/models/inference-venv/bin/python3 \
    ~/models/llama.cpp/convert_hf_to_gguf.py \
    /home/brandonv/models/<merged_dir> \
    --outfile /home/brandonv/models/<name>.f16.gguf \
    --outtype f16
```
~30-60s. Produces ~1-1.4x the BF16 size on disk.

### 4. Quantize F16 → Q4_K_M
```bash
~/models/llama.cpp/build/bin/llama-quantize \
    /home/brandonv/models/<name>.f16.gguf \
    /home/brandonv/models/<name>.Q4_K_M.gguf \
    Q4_K_M
```
~1-3 min. Shrinks ~3x (48 GB → 16 GB for Gemma; 66 GB → 20 GB for Qwen MoE). Some tensors fall back to other quants automatically.

### 5. Serve via llama-server
Template (see `scripts/serve_jarvis_{gemma,moe}_trading_llamacpp.sh`):
```bash
exec ~/models/llama.cpp/build/bin/llama-server \
    --model /home/brandonv/models/<name>.Q4_K_M.gguf \
    --host 0.0.0.0 --port <30000+N> \
    --n-gpu-layers 99 --ctx-size 16384 --threads 8 \
    --jinja
```
- `--n-gpu-layers 99` = offload all layers to GPU
- `--jinja` = use the model's chat template directly; enables native `<think>` tag handling, reasoning_content/content split
- `--ctx-size 16384` = same budget as vLLM serves
- No `--reasoning-parser` flag — llama.cpp handles thinking tags natively

### 6. Smoke test
`/v1/models` returns the GGUF filename as the model id. `/v1/chat/completions` accepts the same OpenAI-compatible payload as vLLM. `timings.predicted_per_second` in the response body gives exact tok/s.

## Config gotcha: mrope_section

Qwen3.5 MoE's `config.json` has `rope_parameters.mrope_section` and `rope_parameters.mrope_interleaved`:
- **vLLM** requires these **stripped** (otherwise M-RoPE assertion at load time)
- **llama.cpp** requires these **present** (otherwise `key not found in model: qwen35moe.rope.dimension_sections` during quantize)

Our current `Jarvis_MoE_trading/config.json` has them **present** (optimized for llama.cpp). If reverting to vLLM serve, strip them again (`jq 'del(.rope_parameters.mrope_interleaved, .rope_parameters.mrope_section)'`).

---

# Qwen3.5 Thinking Mode (Critical Gotcha)

**Symptom that bit us on first MoE deploy**: every response began with "Here's a thinking process that leads to...", numbered steps leaked directly into content. Looked like a training-data problem.

**Actual cause**: Qwen3.5 is a reasoning model. Its chat template auto-opens `<think>\n` as part of the assistant's generation prompt. The model fills that with its reasoning → emits `</think>` when done → emits the final answer. Without a reasoning parser, vLLM returns the whole stream as raw `content`, so the user sees the reasoning scaffold.

**Fix (baked into current serve scripts)**: `--reasoning-parser qwen3` splits the stream at `</think>`:
- Everything before → `reasoning` field (visible if you look, invisible to the orchestrator)
- Everything after → clean `content` field

## Two runtime modes

### Thinking ON (default)
Model reasons verbosely, then answers. Best for complex trade decisions.
```json
{ "messages": [...], "max_tokens": 2000 }
```
Budget 1500-2000 tokens per response (reasoning + answer).

### Thinking OFF (use for fast/simple paths)
Chat template inserts empty `<think></think>`, model skips reasoning.
```json
{ "messages": [...], "max_tokens": 500, "chat_template_kwargs": {"enable_thinking": false} }
```
Budget ~500 tokens. Faster, cleaner, but loses reasoning capability.

**Recommended orchestrator policy**: default thinking ON for decision-making heartbeats, thinking OFF for simple lookups, tool-response summarization, and anywhere latency matters more than analytical depth.

## Field naming caveat

vLLM 0.18.2rc1 emits the reasoning under key `reasoning` in the OpenAI message object.
DeepSeek's actual API uses `reasoning_content`.
Inference providers reading responses from Jarvis will need to check both names (or normalize at the provider boundary). Verified 2026-04-12 via smoke test — 5282 chars in `message.reasoning`, 741 in `message.content`.

## Do NOT skip the reasoning parser

Option considered and rejected: always send `enable_thinking=false` per-request. Suppresses the symptom but throws away the reasoning capability — same downgrade as DeepSeek Chat vs DeepSeek Reasoner. The fine-tune's value comes from reasoning with domain knowledge, not just knowing facts. Serve-side parser keeps the reasoning available for debugging while keeping `content` clean for orchestrator parsing.

---

# Training Data Format (for next-cycle planning)

What Cycle 2 training data DID NOT do (and should in Cycle 3 / new adapters):
1. **No `<think>` tags** in 59,048 training examples (0 opening tags, 64 closing). Model inherits Qwen3.5's thinking behavior from pretraining but has no examples teaching it to structure thinking properly. Workaround = serve-side parser; root fix = tag reasoning in training data.
2. **Near-zero agentic examples** (1 of 59,048). If the fine-tuned model is destined for the UT orchestrator role (tool-calling), Cycle 3 must include substantial agentic traces with explicit `<think>why I'm calling this tool</think><tool_call>...</tool_call>` pattern.
3. **No Q4/quantization-aware testing** during data generation. Not an issue yet for MoE (BF16) but will matter when quantizing Jarvis_MoE_trading for latency.

Reference: `memory/project_cycle2_training_fix.md` and `memory/project_finetuning.md`.
