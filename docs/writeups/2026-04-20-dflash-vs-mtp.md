# DFlash vs MTP on a Fine-Tuned Qwen3.5-27B: Acceptance Survives the Fine-Tune, Speed Doesn't

**Brandon Verbeck · AImindPalace · April 2026**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19673102.svg)](https://doi.org/10.5281/zenodo.19673102)

*Companion to [AImindPalace/dgx-spark-nvfp4-serving](https://github.com/AImindPalace/dgx-spark-nvfp4-serving). Cite via [CITATION.cff](../../CITATION.cff).*

## Summary

We tested [`z-lab/Qwen3.5-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) as the drafter for a DoRA fine-tuned Qwen3.5-27B target on an NVIDIA DGX Spark. The drafter was trained against **base** Qwen3.5-27B, not our fine-tune. The common assumption is that distribution shift from fine-tuning will hurt drafter acceptance enough to make a base-trained speculative head worthless. That assumption was wrong on this setup.

Per-token acceptance rate was **12.6%** (11,104 draft iterations × 15 proposed tokens = 166,560; 20,961 accepted). Mean acceptance length was **1.887** — *higher* than Z-Lab's published 1.77–1.84 on base Qwen3-8B. Position-0 accept rate was **69.3%** vs Z-Lab's 47.8%. Whatever the drafter was trained on, it still tracked the fine-tune well enough to beat the benchmark numbers on the base model.

Speed, however, regressed. DFlash averaged **11.2 tok/s** across 21 prompts (32,046 tokens, 47.8 min). The MTP + NVFP4 baseline on the same Spark is **17.1 tok/s** — DFlash was 35% slower. The speed loss is not a drafter-quality issue; it's structural, and the rest of this note is about why.

## Setup

- Hardware: single NVIDIA DGX Spark, GB10 Blackwell, SM121, 128 GB unified memory, 273 GB/s memory bandwidth.
- vLLM 0.18.2rc1.dev39+gdc0428ebb (eugr prebuilt wheels, software SM121 E2M1 patch).
- Target: `Jarvis_27B_trading`, a DoRA fine-tune of Qwen3.5-27B on 64 trading/finance/decision-science books, merged and served at BF16 (∼55 GB weights).
- Drafter: `z-lab/Qwen3.5-27B-DFlash`, BF16, 3.3 GB.
- Config: `--speculative-config '{"method":"dflash","model":".../Qwen3.5-27B-DFlash","num_speculative_tokens":15}'`, `--attention-backend flash_attn`, `--max-num-batched-tokens 32768`, `--enforce-eager`.
- Workload: 21-prompt harvest suite covering market analysis, strategy, risk management, decision science, statistical reasoning, behavioral finance, and general reasoning. 2048 max tokens per prompt, 1 trial each, batch size 1.
- Baseline: MTP + NVFP4 on the same target-family model, 17.1 tok/s mean, previously measured and documented in [the main serving runbook](../jarvis-serving-runbook.md).

## Acceptance — per-position decay

Drafter proposes 15 tokens; target accepts them in order until the first mismatch. Position-0 is the first token after the verified context; position-14 is the fifteenth speculated token.

| Position | Accepted | % of drafts |
|---:|---:|---:|
| 0  | 7,691 | 69.3% |
| 1  | 4,703 | 42.4% |
| 2  | 2,907 | 26.2% |
| 3  | 1,863 | 16.8% |
| 4  | 1,203 | 10.8% |
| 5  |   823 |  7.4% |
| 6  |   551 |  5.0% |
| 7  |   379 |  3.4% |
| 8  |   260 |  2.3% |
| 9  |   188 |  1.7% |
| 10 |   135 |  1.2% |
| 11 |   102 |  0.9% |
| 12 |    78 |  0.7% |
| 13 |    49 |  0.4% |
| 14 |    29 |  0.3% |

Mean AL (accepted / draft iteration) = **1.887**. For comparison, Z-Lab's paper reports [0.478, 0.181, 0.069, 0.023, ...] per-position on base Qwen3-8B, summing to 0.779 accepted tokens per position before the first reject — which integrates to AL ≈ 1.77. Our drafter, running against a fine-tuned target, outperformed that benchmark.

## Throughput by category

| Category | n | Mean tok/s | Min | Max |
|---|---:|---:|---:|---:|
| strategy             | 4 | 12.62 | 11.10 | 14.20 |
| statistical_reasoning | 2 | 12.65 | 12.10 | 13.20 |
| market_analysis      | 3 | 11.13 |  9.30 | 13.30 |
| risk_management      | 3 | 10.90 | 10.50 | 11.40 |
| behavioral_finance   | 3 | 10.47 | 10.10 | 10.80 |
| decision_science     | 3 |  9.83 |  9.30 | 10.70 |
| general_reasoning    | 3 |  9.33 |  9.10 |  9.50 |
| **overall**          | **21** | **11.20** | **9.10** | **14.20** |

Variance across categories is narrow (~3.3 tok/s from lowest to highest mean). The drafter is not differentially failing on any domain.

## Why DFlash lost despite good acceptance

Three compounding structural reasons:

### 1. Apples-to-oranges precision

Our 17.1 tok/s MTP baseline was measured on an **NVFP4**-quantized target. This DFlash test ran on **BF16**. On a bandwidth-bound device like GB10, BF16 moves roughly 2× more bytes per verification step than NVFP4. Before anything else, that's a ~40% intrinsic penalty. A fair MTP vs DFlash shootout needs both to run on matched precision.

### 2. Drafter overhead is non-trivial on low-throughput hardware

MTP ships a draft head that is ~800 MB and shares the target's embedding/output matrices. Its per-iteration overhead is small. DFlash's drafter is a separate 3.3 GB network with its own forward pass per iteration. On a B200 with ~4,500 GB/s of HBM, that overhead is amortized easily. On a Spark at 273 GB/s, drafter compute eats a proportionally larger slice of wall-clock time.

### 3. Stacking penalties on GB10

Spark-specific constraints (software E2M1 patch for missing SM121 PTX, `--enforce-eager` to avoid CUDA-graph issues with GDN attention, `flash_attn` required for DFlash) compound. Each constraint costs ~5–10% in isolation; stacked, they eat into whatever speculative-decoding gain remains.

## Implications

The interesting finding is **not** that DFlash lost — the paper explicitly benchmarks on B200 and the Spark's hardware ceiling was always going to compress speculative wins. The interesting finding is that a base-trained drafter held up against a fine-tuned target. This has a direct consequence for custom drafter projects.

We had a pending plan to train an [Eagle-3](https://arxiv.org/abs/2503.01840) draft head on our fine-tune's output distribution at ∼$200–500 of RunPod GPU time, on the premise that a base-distribution drafter would be too off-distribution to be useful. This test invalidates the premise on our workload. The expected ROI of training a bespoke drafter is now much lower, since we can get above-spec acceptance from an off-the-shelf drafter at zero additional training cost.

The Eagle-3 project is now deprioritized on Spark. It remains viable if hardware changes to a higher-bandwidth platform, where a 10–20% acceptance edge would compound into a larger absolute throughput win than it does here.

## Reproduce it

```bash
# Drafter (3.3 GB)
hf download z-lab/Qwen3.5-27B-DFlash --local-dir ~/models/Qwen3.5-27B-DFlash

# End-to-end harness (server up + health check + harvest + /metrics snapshot + shutdown)
bash scripts/test_dflash.sh
```

Raw artifacts from this run are in [`benchmarks/results/2026-04-20_dflash_jarvis27/`](../../benchmarks/results/2026-04-20_dflash_jarvis27/): 21 full completions, per-trial metadata, and the post-harvest vLLM Prometheus snapshot with all per-position acceptance counters.

## Citation

> Verbeck, B. (2026). *DFlash vs MTP on a Fine-Tuned Qwen3.5-27B: Acceptance Survives the Fine-Tune, Speed Doesn't.* AImindPalace Technical Note. https://doi.org/10.5281/zenodo.19673102
