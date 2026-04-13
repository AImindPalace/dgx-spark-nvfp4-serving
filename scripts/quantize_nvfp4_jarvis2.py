#!/usr/bin/env python3
"""
NVFP4 quantization for Jarvis_27B_trading via nvidia-modelopt — MAX QUALITY config.

Strategy for Jarvis_2 (Cycle 2 NVFP4):
- 512 calibration samples × 4096 seq_len = ~2M tokens calibrated (matches inference
  distribution of reasoning+answer on long trading prompts)
- Stratified sampling across 65 books + 4 templates (no single domain dominates)
- Layer-wise exclusions: first 2, last 2, and MTP head kept in higher precision
  (these layers are most sensitive to quantization — embeddings narrow, logits sharp)
- AWQ-lite preprocessing stage before NVFP4 (redistributes activation outliers,
  significantly improves quant quality on LLMs with narrow distributions)
- Multi-GPU ready: device_map='auto' splits across available GPUs (4x H200 recommended)

Output is vLLM-compatible (serve with --quantization modelopt --trust-remote-code).

Usage:
    # Single GPU (H200 141GB):
    python quantize_nvfp4_jarvis2.py

    # Multi-GPU (4x H200):
    python quantize_nvfp4_jarvis2.py  # device_map=auto uses all visible GPUs

    # Override defaults:
    python quantize_nvfp4_jarvis2.py --num-calib-samples 512 --seq-len 4096 --no-awq
"""

import argparse
import copy
import glob
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch


# Default paths assume Spark layout. Override with CLI args on other hosts
# (e.g., RunPod: --input /workspace/... --output /workspace/... --training-data-dir /workspace/training_data).
INPUT_DIR = os.path.expanduser("~/models/Jarvis_27B_trading")
OUTPUT_DIR = os.path.expanduser("~/models/Jarvis_2")
TRAINING_DATA_DIR = os.path.expanduser("~/trader/training_data")


def load_calibration_texts_stratified(n_samples: int) -> list[str]:
    """Stratified sampling across 65 books + 4 templates.

    Ensures equal representation so no single author/domain dominates the
    activation statistics. Each sample is a full assistant turn from the
    training corpus (the text the model actually learned to produce).
    """
    by_book_template: dict[tuple[str, str], list[str]] = defaultdict(list)

    json_files = glob.glob(os.path.join(TRAINING_DATA_DIR, "**/*.json"), recursive=True)
    print(f"Found {len(json_files)} training data files")

    for fpath in json_files:
        path = Path(fpath)
        book = path.parent.name  # e.g. "murphy_ta", "aronson"
        # Template encoded in filename suffix (concept_qa, scenario_reasoning,
        # multi_turn_dialogue, chart_analysis) — strip chapter prefix, use suffix.
        stem = path.stem
        for template in ("concept_qa", "scenario_reasoning", "multi_turn_dialogue", "chart_analysis"):
            if stem.endswith(template):
                break
        else:
            template = "other"

        try:
            with open(fpath) as f:
                data = json.load(f)
            for item in data:
                convs = item.get("conversations", item.get("messages", []))
                for msg in convs:
                    role = msg.get("from") or msg.get("role", "")
                    if role in ("gpt", "assistant"):
                        content = msg.get("value") or msg.get("content", "")
                        if content and len(content) > 200:
                            by_book_template[(book, template)].append(content)
        except Exception:
            continue

    buckets = list(by_book_template.keys())
    print(f"Found {len(buckets)} (book, template) buckets")

    # Round-robin sample equally from each bucket until we hit n_samples.
    random.seed(42)
    random.shuffle(buckets)
    for key in buckets:
        random.shuffle(by_book_template[key])

    texts: list[str] = []
    round_idx = 0
    while len(texts) < n_samples:
        progress = False
        for key in buckets:
            if round_idx < len(by_book_template[key]):
                texts.append(by_book_template[key][round_idx])
                progress = True
                if len(texts) >= n_samples:
                    break
        if not progress:
            break  # every bucket exhausted
        round_idx += 1

    print(f"Stratified sample: {len(texts)} texts from {len(buckets)} buckets")
    return texts


def make_calib_dataloader(tokenizer, texts: list[str], seq_len: int):
    """Tokenize calibration texts into single-item batches on GPU."""
    batches = []
    for text in texts:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding=False,
        )
        batches.append(tokens["input_ids"].cuda())
    return batches


def main():
    parser = argparse.ArgumentParser(description="NVFP4 quantization — max quality config")
    parser.add_argument("--num-calib-samples", type=int, default=512,
                        help="Calibration samples (default 512; diminishing returns past 1024)")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="Max tokens per calibration sample (default 4096 matches reasoning length)")
    parser.add_argument("--no-awq", action="store_true",
                        help="Skip AWQ-lite preprocessing (use if modelopt version lacks NVFP4_AWQ_LITE_CFG)")
    parser.add_argument("--input", default=INPUT_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--training-data-dir", default=TRAINING_DATA_DIR,
                        help="Directory tree with book/*.json training files for stratified calibration")
    args = parser.parse_args()
    # Let load_calibration_texts_stratified see the override
    globals()["TRAINING_DATA_DIR"] = args.training_data_dir

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Calibration: {args.num_calib_samples} samples × {args.seq_len} seq_len "
          f"= {args.num_calib_samples * args.seq_len / 1e6:.1f}M tokens")
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.0f} GB")

    # ── Load calibration texts first (before loading model) ──
    texts = load_calibration_texts_stratified(args.num_calib_samples)

    # ── Load tokenizer ──
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)

    # ── Tokenize calibration data ──
    print(f"Tokenizing {len(texts)} calibration samples at seq_len={args.seq_len}...")
    calib_batches = make_calib_dataloader(tokenizer, texts, args.seq_len)
    total_tokens = sum(b.numel() for b in calib_batches)
    print(f"Prepared {len(calib_batches)} batches, {total_tokens:,} total tokens")

    # ── Load model (multi-GPU via device_map=auto) ──
    from transformers import AutoModelForCausalLM
    print(f"Loading model (~54 GB BF16, distributing across {torch.cuda.device_count()} GPU(s))...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s — class: {type(model).__name__}")

    # ── Configure NVFP4 quantization ──
    import modelopt.torch.quantization as mtq

    # Check for AWQ-lite variant (best quality); fall back to DEFAULT.
    use_awq = False
    if not args.no_awq:
        for cfg_name in ("NVFP4_AWQ_LITE_CFG", "NVFP4_AWQ_CFG"):
            if hasattr(mtq, cfg_name):
                config = copy.deepcopy(getattr(mtq, cfg_name))
                print(f"Quant config: {cfg_name} (AWQ-lite preprocessing + NVFP4)")
                use_awq = True
                break
    if not use_awq:
        config = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
        print("Quant config: NVFP4_DEFAULT_CFG (no AWQ preprocessing)")

    # Preserve MTP weights in higher precision (needed for speculative decoding accuracy).
    config["quant_cfg"]["*mtp*"] = {"enable": False}

    # Preserve first 2 + last 2 layers — these are the most quant-sensitive
    # (early layers narrow embedding distributions; last layers produce sharp
    # logits over 248K-token vocab). Qwen3.5-27B has 64 hidden layers.
    for layer_idx in (0, 1, 62, 63):
        config["quant_cfg"][f"*layers.{layer_idx}.*"] = {"enable": False}
    print("Layer exclusions: MTP head + layers 0, 1, 62, 63 (kept in BF16)")

    # ── Calibration forward loop ──
    def forward_loop(model):
        print(f"Running calibration: {len(calib_batches)} batches × {args.seq_len} tokens...")
        t0 = time.time()
        for i, input_ids in enumerate(calib_batches):
            with torch.no_grad():
                model(input_ids)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(calib_batches) - i - 1) / rate
                print(f"  Calib [{i + 1}/{len(calib_batches)}]  "
                      f"{rate:.1f} samples/s  ETA {eta:.0f}s")
        print(f"Calibration complete in {time.time() - t0:.0f}s")

    # ── Quantize ──
    print("Starting NVFP4 quantization...")
    t0 = time.time()
    model = mtq.quantize(model, config, forward_loop=forward_loop)
    elapsed = time.time() - t0
    print(f"Quantization done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # ── Save ──
    # CRITICAL: modelopt 0.42+ requires mte.export_hf_checkpoint() to properly
    # serialize quantized weights + scales. model.save_pretrained() silently
    # drops the quant state and writes BF16 (~54 GB) instead of NVFP4 (~19 GB).
    os.makedirs(args.output, exist_ok=True)
    print(f"Saving to {args.output}...")
    t0 = time.time()

    import modelopt.torch.opt as mto
    import modelopt.torch.export as mte

    # Checkpoint the modelopt state first — recovery point if export_hf_checkpoint fails.
    state_path = os.path.join(args.output, "modelopt_state.pth")
    mto.save(model, state_path)
    print(f"  modelopt state saved → {state_path}")

    # HuggingFace-compatible NVFP4 export (vLLM-loadable with --quantization modelopt).
    mte.export_hf_checkpoint(model, export_dir=args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved in {time.time() - t0:.0f}s")

    # ── Summary ──
    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith(".safetensors")
    )
    print(f"\nModel size: {total_size / 1e9:.2f} GB")
    print(f"Output: {args.output}")
    print("Serve with: vllm serve <path> --quantization modelopt --trust-remote-code")
    print("            (plus --speculative-config for MTP, same flags as Jarvis_1)")


if __name__ == "__main__":
    main()
