# Jarvis_2 NVFP4 Quantization — Post-Mortem (2026-04-12)

Second time through the NVFP4 pipeline for a fine-tuned Qwen3.5-27B dense. Jarvis_1 shipped on 2026-04-01; Jarvis_2 is the Cycle 2 merge (`bverbeck/trading-dora-adapter-v2`, final loss 0.945) run on 4×H200 NVL at "max quality" config (NVFP4_AWQ_LITE_CFG, 512 × 4096 calibration, stratified across 65 books × 4 templates).

This documents everything that went wrong, what the fix was, and how to avoid repeating it next time.

## 1. `model.save_pretrained()` silently drops quant state (cost: one 45-min pod run, ~$8)

**Symptom:** Output directory was 53.8 GB of BF16 safetensors — ~10x larger than the expected ~19 GB NVFP4. No `*_scale` / `*_amax` tensors. No `quantization_config` in `config.json`.

**Root cause:** `modelopt 0.42`'s `mtq.quantize()` returns a model wrapped with `QuantLinear` / `QuantConv` modules. `model.save_pretrained()` strips those wrappers and serializes the *underlying BF16 weights*, silently. No warning, no error.

**Fix:** Use `mte.export_hf_checkpoint()` instead. Also call `mto.save(model, path)` first to checkpoint the modelopt state so you can resume without redoing the ~45 min calibration if the export fails:

```python
import modelopt.torch.opt as mto
import modelopt.torch.export as mte

mto.save(model, os.path.join(output_dir, "modelopt_state.pth"))
mte.export_hf_checkpoint(model, export_dir=output_dir)
```

**Why this was avoidable:** The memory entry `project_jarvis_quantization.md` from Jarvis_1 on 2026-04-01 already documented this exact fix. Wasn't consulted before writing `scripts/quantize_nvfp4_jarvis2.py`. The memory is now updated and this is the #1 gotcha documented in the public repo's README.

## 2. `mte.export_hf_checkpoint()` drops `mtp.*` weights

**Symptom:** After the correct export ran, the output had 0 `mtp.*` tensors — even though we had explicitly kept MTP in BF16 by setting `config["quant_cfg"]["*mtp*"] = {"enable": False}`.

**Root cause:** modelopt's `export_hf_checkpoint()` only serializes weights that are part of the registered `state_dict()` of the HF model class. We load the base as `Qwen3_5ForCausalLM`, which does not include the MTP head in its state_dict — MTP lives outside the standard text-model hierarchy. The "exclude from quant" directive correctly kept MTP in BF16 in memory but did not cause it to be serialized.

**Fix:** MTP is never touched by the DoRA adapter (excluded by module name in the training config), so the base model's MTP tensors are byte-identical to what a correctly-exported merge+quant would have produced. Extract them from the HF base and save as a sidecar shard:

```python
from safetensors.torch import load_file, save_file
import glob

mtp = {}
for shard in sorted(glob.glob("Qwen3.5-27B/model-*.safetensors")):
    for k, v in load_file(shard).items():
        if "mtp" in k.lower():
            mtp[k] = v.clone()

save_file(mtp, "Jarvis_2/model-mtp.safetensors")
```

You get 15 tensors, ~849 MB.

**Why it wasn't caught earlier:** Jarvis_1's 2026-04-01 script grafted MTP via `extra_state_dict` parameter to `mte.export_hf_checkpoint()`. That codepath worked. The Jarvis_2 script dropped the graft step because "we excluded MTP from quant, so it should be there." It wasn't. Exclude-from-quant ≠ include-in-export.

## 3. Exported `config.json` is flat, needs VLM wrapper overlay

**Symptom:** vLLM refused to load the exported model with "Invalid type of HuggingFace config" / routing errors. The exported config had:

```json
{
  "architectures": ["Qwen3_5ForCausalLM"],
  "model_type": "qwen3_5_text",
  "hidden_size": 5120,
  ...
}
```

**Root cause:** modelopt's exporter writes the flat CausalLM config matching the class the model was loaded as. But vLLM's Qwen3.5 loader requires the VLM wrapper format — `Qwen3_5ForConditionalGeneration` / `qwen3_5`, with `text_config` (model_type: `qwen3_5_text`) and `vision_config` nested under the top level — even when serving text-only with `--language-model-only`.

**Fix:** Overlay the working Jarvis_1 config as a template and preserve only modelopt's newly-written `quantization_config`:

```python
j2 = json.load(open("Jarvis_2/config.json"))
j1 = json.load(open("Jarvis_1/config.json"))  # known-working wrapper template

out = dict(j1)
out["quantization_config"] = j2["quantization_config"]
if "transformers_version" in j2:
    out["transformers_version"] = j2["transformers_version"]

json.dump(out, open("Jarvis_2/config.json", "w"), indent=2)
```

**Alternative (no template needed):** The public repo has `config-fixes/fix_config.py` which rebuilds the wrapper from the flat config's own fields. Either approach works; using the template means you guarantee vision_config compatibility with whatever Jarvis_1 uses.

## 4. Missing `model.safetensors.index.json`

**Symptom:** Even after the config was fixed, vLLM couldn't find the MTP sidecar. The single-file `model.safetensors` worked but it only contained main-model weights; MTP was in `model-mtp.safetensors` as a separate shard and there was no index telling the loader about it.

**Root cause:** modelopt exports a single `model.safetensors` with no index because there's only one shard. Once we added a sidecar, we needed an index mapping every key to its shard.

**Fix:** Enumerate the keys from both files and write `model.safetensors.index.json`:

```python
from safetensors import safe_open

weight_map = {}
with safe_open("Jarvis_2/model.safetensors", framework="pt") as f:
    for k in f.keys():
        weight_map[k] = "model.safetensors"
for k in mtp.keys():
    weight_map[k] = "model-mtp.safetensors"

total_size = sum((Path("Jarvis_2") / fn).stat().st_size
                 for fn in {"model.safetensors", "model-mtp.safetensors"})
json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map},
          open("Jarvis_2/model.safetensors.index.json", "w"), indent=2)
```

## Consolidated fixer

All four post-export fixes are now in `scripts/post_export_jarvis2.py`. One command handles the complete pipeline end-to-end:

```bash
python scripts/post_export_jarvis2.py \
    --jarvis2-dir /path/to/Jarvis_2 \
    --template-config /path/to/Jarvis_1/config.json \
    --base-model /path/to/Qwen3.5-27B
```

It rewrites the config, extracts+writes the MTP sidecar, generates the index, and verifies the final layout matches a known-working NVFP4 Qwen3.5-27B deploy (same as Jarvis_1).

## Operational gotchas (less severe)

**5. HuggingFace private storage limit (403 Forbidden on upload).** The private tier cap was reached mid-upload. Freed space by deleting `bverbeck/Jarvis-MoE-35B` (65 GB) and `bverbeck/trading-dora-adapter-gemma-moe` (510 MB) — both kept losing the shootout (Gemma 42.0, MoE 45.5 vs Jarvis_1 51.0), so losing them is cheap. For next quantization run: clean up stale repos first.

**6. `flash-attn` compile failed on pod.** Takes ~30 min to build from source against torch 2.x + CUDA 13 on H200. Gave up, ran calibration on SDPA fallback instead — about 2x slower on pass 2 but not worth fixing at run time. For next pod: pre-install `flash-attn` during image setup, not at calibration start.

**7. FLA (`flash-linear-attention`) fast path not available at first forward pass.** Qwen3.5's GDN linear_attention layers need FLA kernels or they fall back to a naive implementation that's very slow. Must `pip install flash-linear-attention` before the model ever runs a forward pass. Already in the memory as a durable "install FLA before any Qwen3.5 forward" rule; on pod this was done correctly this time.

## What made the difference

**Scripts that did the right thing on Jarvis_2:**
- `scripts/quantize_nvfp4_jarvis2.py` — AWQ-lite + stratified calibration + layer exclusions + `mte.export_hf_checkpoint()` (after fix)
- `scripts/post_merge_minimal.py` — MTP graft before quant + strip `mrope_*` from flat config
- `scripts/post_export_jarvis2.py` — the consolidated post-export fixer (new today)

**Pipeline invariant:** Next time a third Jarvis run happens, the post-export layer should be a single command with no manual steps. That's what `post_export_jarvis2.py` encodes.

**Memory durability:** The `project_jarvis_quantization.md` memory should always be read at the start of any new quant run, not just when something fails. Adding a "SAVE STEP" section at the very top helped, but the real fix was making the post-export fixer a single script so the MTP/config/index work doesn't depend on remembering three separate manual fixes.

## What's ready to test

- `/home/brandonv/models/Jarvis_2/` on Spark:
  - `model.safetensors` — 21 GB, NVFP4 packed weights + float8 scales
  - `model-mtp.safetensors` — 849 MB, BF16 MTP head (15 tensors)
  - `model.safetensors.index.json` — 2,381 keys mapped
  - `config.json` — VLM wrapper format, modelopt `quantization_config` preserved
  - `hf_quant_config.json`, `chat_template.jinja`, `tokenizer*`, `generation_config.json`
- `~/models/serve_jarvis_2.sh` on Spark — same flags as Jarvis_1 serve (`--quantization modelopt`, `--language-model-only`, MTP speculative config)
- Shootout harvest + scoring already wired (from 2026-04-12 Jarvis_1 / MoE / Gemma run)

Next session: run the 21-prompt shootout with Jarvis_2 added as a fourth candidate. If it lands ≥ Jarvis_1's 51.0/60, it's the new quality leader and the Eagle-3 training target.
