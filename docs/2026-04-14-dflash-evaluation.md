# DFlash on DGX Spark — Evaluation Notes (2026-04-14)

Research-only. No code changes yet. Goal: decide whether to replace (or stack alongside) our current NVFP4 + MTP setup (17.1 tok/s on Jarvis_1) with **DFlash** speculative decoding from z-lab.

## What DFlash is

DFlash is a **diffusion-based speculative decoding drafter** from [z-lab](https://github.com/z-lab/dflash). It is **not** a new quantization scheme and it is **not** Qwen's native MTP. It's a separate ~2B-param BF16 drafter trained specifically to speculate for Qwen3.5-27B dense.

- Drafter weights: [`z-lab/Qwen3.5-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) (2B BF16 safetensors)
- Target model: Qwen3.5-27B dense (exact match for Jarvis_1 / Jarvis_2)
- Trained context: 4096 tokens (sliding-window flag extends it)
- Paper claims: 1.3×–5.2× speedup over AR baseline depending on task + batch size
- Merged into vLLM nightly (≥ 0.19.x); present in SGLang 0.5.6+

## How it compares to what we have

| Setup | Target quant | Drafter | Reported tok/s | Notes |
|---|---|---|---|---|
| **Jarvis_1 baseline (this repo)** | NVFP4 | — | 11.2 | measured |
| **Jarvis_1 + MTP (this repo)** | NVFP4 | Qwen native MTP (BF16) | **17.1** | measured, 52% over baseline |
| Forum report ([optimisation thread](https://forums.developer.nvidia.com/t/qwen3-5-27b-optimisation-thread-starting-at-30-t-s-tp-1/366009)) | int4-AutoRound | DFlash | ~30 | `n_spec=12–16`, 15–30% accept |
| Forum report ([DFlash thread](https://forums.developer.nvidia.com/t/dflash-llm-for-dgx-spark-too-good-to-be-true/366445)) | INT4/FP8 (Qwen3.5-35B) | DFlash | 31 (complex) / >70 (simple) | Task-dependent |
| HF discussion reference numbers | BF16 target | DFlash | — | 47.24% accept, 8.09 mean accept len |
| HF discussion reference numbers | FP8 target | DFlash | — | 46.52% accept, 7.98 mean accept len |
| HF discussion — Blackwell users | FP8 | DFlash | — | 12–20% accept (low), workaround: `--no-enable-prefix-caching` → 30–35% |

So the **headline** is: DFlash on Qwen3.5-27B has been reported at ~30 tok/s on Spark with INT4 weights — a ~2× over our MTP number. But the comparison is **not** apples-to-apples, and several gotchas apply to our stack.

## Gotchas for our stack specifically

1. **NVFP4 + DFlash is unverified.** Every reported number on Spark uses INT4 or FP8. No public measurement of DFlash against an NVFP4 target exists (as of 2026-04-14, searched HF, the two NVIDIA forum threads, and the z-lab GitHub). The HF discussion shows FP8 target works; the SM121 E2M1 software patch that makes NVFP4 work on our box is a separate hot path from the drafter, so there's no *theoretical* blocker, but this is the first real risk.
2. **vLLM version mismatch.** Forum DFlash runs use vLLM `0.19.1rc1.dev46+gc5e3454e5.d20260406.cu132`. We're on eugr-packaged 0.18.2rc1. DFlash landed post-0.19, so we'd need either:
   - a newer eugr drop (check [spark-vllm-docker releases](https://github.com/eugr/spark-vllm-docker/releases))
   - or a source build against CUDA 13.2 + torch nightly (the 20–40 min Blackwell kernel compile the agent-quoted article describes)
3. **DFlash and MTP are mutually exclusive.** `--speculative-config` takes one `method`. You can't stack MTP + DFlash in the same serve. A DFlash swap would *replace* the 52% MTP win.
4. **Attention backend constraint.** DFlash requires `--attention-backend flash_attn` (vLLM) / `fa3` (SGLang). Our current runbook lets vLLM auto-select FLASHINFER_CUTLASS. That backend swap has unknown interaction with the eugr SM121 E2M1 patch, which lives on the GEMM path, not the attention path — so likely fine, but worth verifying before claiming a win.
5. **Prefix caching bug on Blackwell.** HF discussion #2 documents that prefix caching tanks acceptance to <20% on Blackwell/H100. Fix is `--no-enable-prefix-caching`. Losing prefix cache hurts TTFT on multi-turn chat — relevant for our trading-desk agentic use.
6. **+2 GB drafter overhead.** 2B BF16 = ~4 GB. MTP head is 811 MB. On our 0.40 utilization budget (~48 GB) that's still fine (19 GB NVFP4 + 4 GB drafter + KV + overhead) but leaves less headroom than MTP.
7. **Acceptance rate is workload-dependent.** Reported range 15–35% on Spark. Our trading-prompt workload (long-form structured reasoning) is closer to the "complex context" case where the forum reports ~31 tok/s, not the ~70 tok/s "hello world" case. Realistic expectation is **~25–30 tok/s** *if* it works on NVFP4 — still a big win over 17.1 if it holds.

## Decision framework

DFlash is worth a pod run, but *not* a blind swap into production. Before we touch Jarvis_1's serve:

- [ ] Confirm eugr or upstream vLLM ≥ 0.19 runs on Spark with our NVFP4 checkpoint at all (no DFlash yet — just the version bump). Baseline tok/s on current config must still reproduce 11.2/17.1 on the newer vLLM before drafter work means anything.
- [ ] A/B test: same checkpoint, same prompts (`benchmarks/prompts.json`, full 21-prompt harness), three configs on same vLLM build:
  - `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` (current)
  - `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-27B-DFlash","num_speculative_tokens":8}' --attention-backend flash_attn --no-enable-prefix-caching`
  - same DFlash config with `num_speculative_tokens=15` (HF recommended)
- [ ] Track: steady-state tok/s, TTFT, mean acceptance length, memory used, vLLM logged acceptance rate.
- [ ] Quality check: re-score at least the 15 prompts from the 2026-04-12 shootout to confirm no regression (DFlash is lossless in theory — verification rejects wrong drafts — but exposure on NVFP4 + SM121 patch has no prior datapoints).
- [ ] If DFlash wins on throughput + ties on quality, also measure with prefix caching *enabled* to see if recent vLLM patches fixed the <20% accept bug on Blackwell for NVFP4 targets.

## Open questions (worth asking before spending pod hours)

1. Does z-lab's drafter tokenizer exactly match our fine-tuned Jarvis_1 tokenizer? DoRA merge doesn't change the tokenizer, and DFlash uses Qwen/Qwen3.5-27B's tokenizer, so this should be fine — but confirm `added_tokens.json` and chat template parity.
2. Does the drafter's 4096-token training window cause rejection explosions on our 8192 `--max-model-len`? Sliding-window flag `--speculative-dflash-draft-window-size` is the lever; HF page mentions it for long-context workloads.
3. Does DFlash compose with our `--kv-cache-dtype fp8`? The drafter is BF16; KV is shared with the target. Unverified.

## Not the story the shared context claimed

The article snippet pasted into the chat conflates three separate things:

- **DFlash** = z-lab's diffusion drafter (the thing we care about here)
- **FlashInfer** = the CUDA backend our setup already uses (FLASHINFER_CUTLASS, via eugr SM121 patch)
- **Docker vs native install** = a completely orthogonal packaging question that this repo already resolved (native, eugr wheels, documented in README §Setup)

The "use Docker because Blackwell is hard" paragraph is correct general advice for a cold start but **not** our situation — Jarvis_1 is already serving natively and stable. The DFlash angle is the only genuinely new lever in the referenced material.

## Recommendation

1. **Don't rewrite the runbook or serve scripts yet.** Our current NVFP4 + MTP path is the only publicly verified dense-27B NVFP4 measurement on Spark and it works.
2. **Spin a branch for a DFlash A/B** — probably worth one ~2-hour session on Spark: version-bump vLLM, pull the drafter, run the 21-prompt harness 3 ways, record numbers.
3. **If the A/B shows ≥25 tok/s with no quality regression on NVFP4**, write it up as a 2026-04-XX update section in the README and add a `scripts/serve_jarvis_27b_dflash.sh` sibling to the MTP serve script — keep both, default to the winner.
4. **If DFlash refuses to run on NVFP4** (most likely failure mode: drafter-verifier logit mismatch on the SM121 E2M1 software path), document it as a "things that didn't work" row and stay on MTP. The ~30 tok/s community numbers are on INT4-AutoRound, which is a different quantization we'd have to re-produce to match them — at which point we're not serving our fine-tune's NVFP4 checkpoint anymore, and that's a much bigger scope change than a speculative-decoding swap.

## Sources

- z-lab drafter: [z-lab/Qwen3.5-27B-DFlash on HuggingFace](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash)
- FP8/BF16 acceptance numbers + Blackwell prefix-cache workaround: [HF discussion #2](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash/discussions/2)
- z-lab GitHub: [github.com/z-lab/dflash](https://github.com/z-lab/dflash)
- Spark community numbers: [NVIDIA forum — DFlash thread](https://forums.developer.nvidia.com/t/dflash-llm-for-dgx-spark-too-good-to-be-true/366445)
- Spark 27B optimization thread (30+ tok/s TP=1): [NVIDIA forum — Qwen3.5-27B optimization](https://forums.developer.nvidia.com/t/qwen3-5-27b-optimisation-thread-starting-at-30-t-s-tp-1/366009)
