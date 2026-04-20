# Four Post-Export Fixes Required to Serve an NVFP4-Quantized Qwen3.5-27B on DGX Spark

**Brandon Verbeck · AImindPalace · April 2026**

*Companion technical note to [AImindPalace/dgx-spark-nvfp4-serving](https://github.com/AImindPalace/dgx-spark-nvfp4-serving). Ready for cross-posting to a blog or as an informal preprint. Cite via [CITATION.cff](../../CITATION.cff).*

## Abstract

NVIDIA's `modelopt` library (v0.42) is the canonical path for producing NVFP4 checkpoints that vLLM can load. The library's `README` and sample scripts advertise a two-line recipe — quantize with `mtq`, export with `export_hf_checkpoint()` — that works end-to-end for most architectures. It does **not** work end-to-end for Qwen3.5-27B, the first publicly released member of Qwen's new hybrid-attention dense family. The exported artifact is byte-identical to what the docs imply it should be, and vLLM refuses to load it. This note documents the four post-export rewrites required to bridge the gap, consolidated into a single script in the accompanying repository. The resulting checkpoint serves at **11.2 tok/s baseline / 17.1 tok/s with MTP speculative decoding** on a single DGX Spark (GB10, 128 GB UMA, SM121), with no quality regression versus BF16 on a 20-prompt domain benchmark.

## 1. Background

The DGX Spark pairs a 20-core Grace ARM CPU with a GB10 Blackwell GPU over a 273 GB/s unified memory bus. Its SM121 compute capability is missing the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction that production Blackwell parts use for native FP32→E2M1 packing; a software patch maintained in the [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) and [Avarok-Cybersecurity/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm) projects supplies the missing kernel.

Qwen3.5-27B ("dense") is not a vanilla transformer. It uses a 3:1 hybrid attention pattern (48 Gated DeltaNet linear-attention layers, 16 full-attention layers), exposes a Multi-Token Prediction (MTP) draft head for speculative decoding, and its HuggingFace config presents a VLM wrapper (`Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5`) even when used text-only. All three choices interact badly with modelopt's export path.

## 2. The Problem

Running the documented quantize-and-export recipe on a BF16 merged checkpoint produces:

```
<output_dir>/
├── config.json                 # flat Qwen3_5ForCausalLM config
├── hf_quant_config.json
├── model-00001-of-000XX.safetensors
├── ...
└── tokenizer files
```

vLLM 0.18.2 fails to load this directory with one of four symptoms depending on which fault it hits first: (a) `model.safetensors.index.json` not found; (b) unknown architecture `Qwen3_5ForCausalLM`; (c) MTP layer parameters missing when `--speculative-config '{"method":"qwen3_next_mtp"}'` is set; (d) activations silently falling to FP32 at inference time if `save_pretrained()` was used in place of `export_hf_checkpoint()`. Only the last of these is documented in the modelopt issue tracker; the first three are undocumented as of April 2026.

## 3. The Four Fixes

### 3.1 Use `export_hf_checkpoint()`, not `save_pretrained()`

`model.save_pretrained()` on a modelopt-wrapped model writes the parameters **without** the quantization metadata (`hf_quant_config.json`, `input_scale` / `weight_scale_2` tensors). vLLM then loads the checkpoint as BF16 and silently dequantizes at runtime. The fix is to import `modelopt.torch.export.export_hf_checkpoint` and call it instead. This is the only fix that appears in the modelopt examples; it is necessary but not sufficient.

### 3.2 Graft MTP weights from the base checkpoint

`export_hf_checkpoint()` serializes the state_dict of `Qwen3_5ForCausalLM`, which does not include the MTP head (MTP lives on the outer VLM wrapper). Even when the user correctly configures `"*mtp*": {"enable": False}` to keep MTP weights in BF16, those weights are dropped on export. The fix is to open the pre-merge BF16 checkpoint, extract all keys matching `model.mtp.*`, and write them as a sidecar `model-MTP.safetensors` alongside the exported quantized shards. We retain BF16 precision to match vLLM's `qwen3_next_mtp` speculative-decoding expectations.

### 3.3 Rewrite the config to the VLM wrapper form

vLLM's model registry keys Qwen3.5 off `architectures: ["Qwen3_5ForConditionalGeneration"]` and `model_type: "qwen3_5"`, with the text-model hyperparameters nested under a `text_config` sub-object. `export_hf_checkpoint()` emits the flat causal-LM form (`Qwen3_5ForCausalLM` / `qwen3_5_text`). The fix is a one-shot rewrite that reparents the flat config under `text_config`, replaces the architecture string, and preserves the top-level `quantization_config` and `torch_dtype` fields so vLLM's quantization loader still finds them.

### 3.4 Generate `model.safetensors.index.json` for the MTP sidecar

When multiple safetensors shards share a directory, HF expects a `model.safetensors.index.json` mapping every parameter name to its containing shard. `export_hf_checkpoint()` writes this index for the main quantized shards, but once the MTP sidecar is added in step 3.2 the index is out of date. The fix reads every shard, rebuilds the name→file map, and writes a new `model.safetensors.index.json` with a correct `metadata.total_size` field.

All four fixes are consolidated into [`scripts/post_export_jarvis2.py`](../../scripts/post_export_jarvis2.py), which runs in under a second against a finished modelopt export.

## 4. Results

On a single DGX Spark (OS 7.4.0, driver 580.126.09, CUDA 13.0), serving via native vLLM 0.18.2rc1 with the eugr prebuilt wheels:

| Configuration | Decode tok/s | GPU memory | Notes |
|---|---|---|---|
| BF16 (no quant)         | 4.0   | >90 GB | Baseline from NVIDIA developer forum |
| NVFP4, `--enforce-eager` | 11.2  | 55 GB  | This work, baseline |
| NVFP4 + MTP             | **17.1** | 59 GB | This work, speculative decoding enabled |
| NVFP4 theoretical peak  | ~20.2 | —     | Bandwidth-limited ceiling |

MTP delivers a 52% throughput improvement over NVFP4 baseline, lifting utilization from ~56% of the bandwidth ceiling to ~84%. On a 20-prompt domain quality benchmark (five-criterion rubric, blind-scored by Claude Opus against a DeepSeek-Reasoner reference), the MTP-enabled fine-tune scored 4.71/5 across all prompts and 4.91/5 excluding refusals — equal to or higher than the NVFP4 baseline on every dimension. MTP speculative decoding does not degrade output quality on this task mix.

To our knowledge, this is the first verified MTP-enabled NVFP4 measurement for Qwen3.5-27B dense on a DGX Spark in the public record.

## 5. Reproduction

Full pipeline (merge → quantize → post-export → serve) is documented in [`docs/jarvis2-nvfp4-runbook.md`](../jarvis2-nvfp4-runbook.md); the post-export fixer alone is [`scripts/post_export_jarvis2.py`](../../scripts/post_export_jarvis2.py). A complete post-mortem of the pipeline run on which these fixes were developed, including the 45-minute pod run that motivated consolidation, lives in [`docs/2026-04-12-jarvis2-post-mortem.md`](../2026-04-12-jarvis2-post-mortem.md).

## 6. Why This Matters

The gap between "the library's documented recipe" and "a checkpoint vLLM will actually serve" is small in line count (four script-sized fixes) but large in wall-clock time for anyone who hits it without prior knowledge — each fix fails silently or with a misleading error, and the feedback loop on the Spark is slow (a failed serve can leave the GB10 in an unrecoverable D-state requiring a hard reboot). We publish these fixes so that the next team fine-tuning a hybrid-attention Qwen model for Spark-class hardware can skip the forensic work and start from a working baseline.

## Acknowledgements

The SM121 software E2M1 patch is the work of [Avarok-Cybersecurity](https://github.com/Avarok-Cybersecurity/dgx-vllm), packaged for easy installation by [eugr](https://github.com/eugr/spark-vllm-docker). Independent measurements on the NVIDIA developer forums by forum members *cho*, *joshua.dale.warner*, and *josephbreda* provided the comparison points in §4.

## References

1. AImindPalace. *Serving a Fine-Tuned Qwen3.5-27B (Dense, NVFP4) on DGX Spark.* GitHub, 2026. https://github.com/AImindPalace/dgx-spark-nvfp4-serving
2. NVIDIA. *TensorRT Model Optimizer (modelopt) — NVFP4 Quantization Format.* https://github.com/NVIDIA/TensorRT-Model-Optimizer
3. vLLM Project. *vLLM: A high-throughput and memory-efficient inference and serving engine for LLMs.* https://github.com/vllm-project/vllm
4. eugr. *spark-vllm-docker — prebuilt vLLM wheels for DGX Spark.* https://github.com/eugr/spark-vllm-docker
5. Avarok-Cybersecurity. *dgx-vllm — SM121 software E2M1 patch.* https://github.com/Avarok-Cybersecurity/dgx-vllm
6. NVIDIA Developer Forum. *How fast can Qwen3.5-27B be after converting to NVFP4?* https://forums.developer.nvidia.com/t/how-fast-can-qwen3-5-27b-be-after-converting-to-nvfp4/362776

---

*Suggested citation: Verbeck, B. (2026). Four Post-Export Fixes Required to Serve an NVFP4-Quantized Qwen3.5-27B on DGX Spark. AImindPalace Technical Note. https://github.com/AImindPalace/dgx-spark-nvfp4-serving/blob/main/docs/writeups/modelopt-nvfp4-export-fixes.md*
