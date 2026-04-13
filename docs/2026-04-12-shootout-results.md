# Shootout Results — 2026-04-12

Head-to-head evaluation of the three deployed models on 21 trading/finance prompts (20 scored; one ID-mismatched). Speed measured via benchmark mode (5 trials × 200 tokens); quality via harvest mode (1 trial × 2048 tokens) graded blind by Claude Opus against DeepSeek-Reasoner reference on 6 dimensions (accuracy/depth/reasoning/specificity/domain knowledge/actionability, 10 each, max 60).

## Setup

| Model | Base | Quant | Adapter | Stack | Port |
|---|---|---|---|---|---|
| `jarvis1-nvfp4-mtp` | Qwen3.5-27B dense | NVFP4 | Cycle 1 DoRA (loss 0.967) | vLLM 0.18.2rc1 + qwen3_next_mtp | 8000 |
| `gemma4-q4km-llamacpp` | Gemma 4 26B-A4B-IT | Q4_K_M (4.88 BPW) | Cycle 2 DoRA (loss 0.261) | llama.cpp b8768 | 30000 |
| `qwenmoe-q4km-llamacpp` | Qwen 3.5-35B-A3B | Q4_K_M (4.88 BPW) | Cycle 2 DoRA (loss 1.151) | llama.cpp b8768 | 30001 |

Reference: DeepSeek-Reasoner (20 prompts harvested 2026-04-03).

## Speed

Benchmark mode (5 trials × 200 tokens × 21 prompts; trial 1 discarded as warmup):

| Model | Throughput | TTFT | Memory |
|---|---|---|---|
| Qwen MoE | **72.5 tok/s** (±0.8) | 0 ms (prompt-cached) | 102.8 GB |
| Gemma 4 | 65.4 tok/s (±0.8) | 16.8 ms | 99.8 GB |
| Jarvis_1 | 17.6 tok/s (±0.6) | 227 ms | 97.0 GB |

Harvest mode (1 trial × 2048 tokens × 21 prompts):

| Model | Throughput | Total wall time |
|---|---|---|
| Qwen MoE | 72.1 tok/s | 9.7 min |
| Gemma 4 | 62.2 tok/s | 10.4 min |
| Jarvis_1 | 16.8 tok/s | 41.0 min |

Qwen MoE is 4.1x faster than Jarvis_1; Gemma 4 is 3.7x. Both MoE quantized variants win cleanly on throughput.

## Quality

Blind Opus scoring (fair run, all 3 candidates sent full reasoning + content):

| Model | Avg Score | First-place wins |
|---|---|---|
| **Jarvis_1** (NVFP4, Cycle 1) | **49.2 / 60** | **15 / 19** |
| Qwen MoE (Q4_K_M, Cycle 2) | 44.4 / 60 | 3 / 19 |
| Gemma 4 (Q4_K_M, Cycle 2) | 43.3 / 60 | 1 / 19 |

Jarvis_1 dominates across almost every category — market analysis, risk management, decision science, statistical reasoning, behavioral finance, general reasoning. MoE takes 3 strategy/market-analysis prompts; Gemma takes 1 risk_management.

## Why Jarvis_1 wins quality despite Cycle 1 training

Multiple factors stack:

1. **Quantization precision** — NVFP4 uses block-wise FP4 with scale metadata, preserving more weight information than Q4_K_M's mixed 4/5-bit per-tensor quant. The precision gap likely matters most on numerical/R-R reasoning, which the test prompts heavily feature.
2. **Training loss is not an apples-to-apples quality signal.** Gemma's 0.261 loss (lowest) corresponds to the lowest shootout score; Jarvis_1's 0.967 (highest) wins. Loss measures fit to training distribution, not transfer to novel trading scenarios.
3. **Gemma's IT base may hurt on trading depth.** Instruction-tuned bases are pulled toward generic helpful-assistant style; the DoRA adapter can reinforce domain knowledge but can't undo a general conversational anchor.
4. **Qwen MoE's active-parameter constraint.** Only 3B of 35B params contribute per token. Fewer effective parameters → thinner numerical reasoning than the 27B dense activation in Jarvis_1.

## Operational takeaways

1. **Quality ≠ speed for the orchestrator role.** A 17 tok/s model that reasons better on each decision beats a 72 tok/s model that makes worse calls — if decision latency isn't blocking trades. For the UT daemon, heartbeat cycles are ~30 min, so per-call latency is not a primary constraint; decision quality is.
2. **The real next step is Jarvis_2.** Quantize today's Cycle 2 27B BF16 merge (`Jarvis_27B_trading`) to NVFP4 on RunPod (~$20-40, 1-2h). Expected result: Jarvis_1-architecture speed (17-20 tok/s with MTP) combined with Cycle 2's better training coverage (65 books vs 38). Should score ≥ Jarvis_1's 49.2 and hold the 17 tok/s baseline.
3. **Consider bumping quant on MoE/Gemma to Q5_K_M or Q6_K.** If the quality gap is driven by quant aggressiveness, Q5/Q6 could close 2-4 points while keeping most of the speed (~55-65 tok/s expected). Worth a quick re-run.
4. **MoE and Gemma remain valid for parallel/sanity tasks.** Use them as second-opinion cross-checks during pre-market planning, not as primary decision makers.

## Artifacts

- `benchmarks/results/2026-04-12_jarvis1-nvfp4-mtp/` — Jarvis_1 harvest + benchmark JSON (gitignored, lives in AImindPalace)
- `benchmarks/results/2026-04-12_gemma4-q4km-llamacpp/` — Gemma 4 harvest + benchmark JSON
- `benchmarks/results/2026-04-12_qwenmoe-q4km-llamacpp/` — Qwen MoE harvest + benchmark JSON
- `benchmarks/results/shootout_scores_2026-04-12_fair.json` — full blind Opus scoring output with per-prompt scores and rankings

## Methodology note

First scoring pass was biased: the harvest script only saved `message.content`. For llama.cpp serves with `<think>` tag splitting, thinking content goes to `reasoning_content` and is separate. Gemma/MoE saved partial or empty completions while Jarvis_1 (no reasoning parser on its vLLM serve) saved full raw output. `scripts/shootout_harvest.py` was patched to concatenate `reasoning_content` + `content` into `completion` for parity. The 2048-token results above are from that corrected re-run.

## Fair4k addendum — all 3 models re-harvested at 4096 max_tokens

Follow-up concern: with 2048 tokens, MoE/Gemma might have been constrained on reasoning length (many responses hit the 2048 cap). Re-harvested all three at 4096 budget to eliminate that variable.

| Rank | Model | Score (/60) | Wins | Δ from 2048 | Speed |
|---|---|---|---|---|---|
| 🥇 | Jarvis_1 (NVFP4 27B, Cycle 1) | **51.3** | **18/19** | +2.1, +3 wins | 16.7 tok/s |
| 🥈 | Qwen MoE (Q4_K_M, Cycle 2) | 45.5 | 1/19 | +1.1, -2 wins | 71.8 tok/s |
| 🥉 | Gemma 4 (Q4_K_M, Cycle 2) | 42.0 | 0/19 | −1.3, −1 win | 61.8 tok/s |

Results:
- Jarvis_1 scored **higher** with more room to elaborate (51.3 vs 49.2).
- MoE scored slightly higher (45.5 vs 44.4) — was mildly budget-constrained before.
- **Gemma scored LOWER with more budget**. Its formulaic `DECISION FRAMEWORK` template amplified with room to pad, and Opus penalized the repetition.

Key conclusion: **the quality gap is structural, not a token-budget artifact.** Jarvis_1 has more encoded trading knowledge than Gemma or MoE — NVFP4 quant precision, 27B dense activation, and Cycle 1 Qwen base all compound.

Results: `benchmarks/results/shootout_scores_2026-04-12_fair4k.json`
Completions: `benchmarks/results/2026-04-12_*-fair4k/`

## Q5+imatrix MoE addendum — does precision close the gap?

Follow-up experiment: quantize the merged Qwen MoE at **Q5_K_M with in-domain imatrix calibration** (256 trading samples used to compute per-layer importance matrix via `llama-imatrix`, then `llama-quantize ... Q5_K_M --imatrix`). Goal: trade some MoE speed for quality, see if it closes the gap to Jarvis_1.

| Rank | Model | Score (/60) | Wins (of 19) | Speed |
|---|---|---|---|---|
| 🥇 | Jarvis_1 (NVFP4 27B, Cycle 1) | **51.0** | 15 | 17 tok/s |
| 🥈 | **Qwen MoE Q5_K_M + imatrix** (Cycle 2) | **46.1** | **2** | **56 tok/s** |
| 🥉 | Qwen MoE Q4_K_M (Cycle 2) | 44.9 | 2 | 72 tok/s |
| 4 | Gemma 4 Q4_K_M (Cycle 2) | 42.0 | 0 | 62 tok/s |

**Result: +1.2 quality points, −16 tok/s (22% slower).** Marginal.

Interpretation:
- Q5_K_M+imatrix (5.71 BPW) vs Q4_K_M (4.88 BPW) = 17% more bits per weight
- Quality improvement was only +2.7% relative (44.9 → 46.1)
- Cannot close the 4.9-point gap to Jarvis_1 via precision alone
- Remaining gap is structural: MoE's 3B active param count + Gemma's IT-base anchor are the real limits

**Takeaway**: the MoE's quality ceiling is architecturally bounded. Going from Q4 → Q5 → Q6 → Q8 or even BF16 won't buy you Jarvis_1-equivalent quality because MoE's 3B active per token is an information capacity bottleneck, not a quantization artifact.

**Production recommendation**:
- **Keep MoE on Q5_K_M+imatrix** (56 tok/s, 46.1 quality) if marginal quality gain matters
- **Keep MoE on Q4_K_M** (72 tok/s, 44.9 quality) if volume matters more
- Neither catches Jarvis_1 — that requires the 27B dense NVFP4 path (Jarvis_2)

## The cycle 2 dense open question

The shootout doesn't directly measure **Cycle 2 27B dense NVFP4** (Jarvis_2) — that model doesn't exist yet. Jarvis_1 is Cycle 1. The clean comparison we're setting up:

| Model | Expected score | Status |
|---|---|---|
| Jarvis_1 (Cycle 1 NVFP4 27B) | 51.0 measured | Deployed |
| **Jarvis_2 (Cycle 2 NVFP4 27B)** | **53-55 projected** (same architecture + better training) | Quantization job in flight 2026-04-12 PM |

If Jarvis_2 lands at ≥51, it's the new quality leader and the Eagle-3 training target.
