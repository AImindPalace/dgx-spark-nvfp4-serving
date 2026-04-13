#!/usr/bin/env python3
"""
NVFP4 quantization for Qwen3.5-27B-trading via nvidia-modelopt.

Uses NVFP4_DEFAULT_CFG (E2M1 weights + E4M3 block scales, group size 16).
Calibration uses domain-specific trading data from our training corpus.

Output is vLLM-compatible (serve with --quantization modelopt).

Usage:
    source ~/models/inference-venv/bin/activate
    python ~/models/quantize_nvfp4.py [--num-calib-samples 256] [--seq-len 256]
"""

import argparse
import copy
import gc
import glob
import json
import os
import random
import time

import torch


INPUT_DIR = os.path.expanduser("~/models/Qwen3.5-27B-trading")
OUTPUT_DIR = os.path.expanduser("~/models/Jarvis_1")
TRAINING_DATA_DIR = os.path.expanduser("~/trader/training_data")


def load_calibration_texts(n_samples: int = 256) -> list[str]:
    """Load calibration texts from training data corpus."""
    texts = []
    json_files = glob.glob(os.path.join(TRAINING_DATA_DIR, "**/*.json"), recursive=True)
    random.shuffle(json_files)
    print(f"Found {len(json_files)} training data files")

    for fpath in json_files:
        if len(texts) >= n_samples * 3:  # gather extra, then sample
            break
        try:
            with open(fpath) as f:
                data = json.load(f)
            for item in data:
                convs = item.get("conversations", item.get("messages", []))
                for msg in convs:
                    content = msg.get("content", msg.get("value", ""))
                    if content and len(content) > 100:
                        texts.append(content)
        except Exception:
            continue

    random.shuffle(texts)
    texts = texts[:n_samples]
    print(f"Loaded {len(texts)} calibration texts")
    return texts


def make_calib_dataloader(tokenizer, texts: list[str], seq_len: int, batch_size: int = 1):
    """Tokenize calibration texts into batches."""
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
    parser = argparse.ArgumentParser(description="NVFP4 quantization via modelopt")
    parser.add_argument("--num-calib-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Calibration: {args.num_calib_samples} samples, seq_len={args.seq_len}")

    # ── Load calibration texts first (before loading model to save memory) ──
    texts = load_calibration_texts(args.num_calib_samples)

    # ── Load tokenizer ──
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(INPUT_DIR, trust_remote_code=True)

    # ── Tokenize calibration data ──
    print("Tokenizing calibration data...")
    calib_batches = make_calib_dataloader(tokenizer, texts, args.seq_len)
    print(f"Prepared {len(calib_batches)} calibration batches")

    # ── Load model ──
    from transformers import AutoModelForCausalLM
    print("Loading model (this will use ~51 GB)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        INPUT_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ── Configure NVFP4 quantization ──
    import modelopt.torch.quantization as mtq

    config = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    # Exclude MTP head from quantization (keep BF16 for speculative decoding)
    config["quant_cfg"]["*mtp*"] = {"enable": False}
    print("Quantization config: NVFP4_DEFAULT_CFG + MTP excluded")

    # ── Calibration forward loop ──
    def forward_loop(model):
        print(f"Running calibration with {len(calib_batches)} batches...")
        for i, input_ids in enumerate(calib_batches):
            with torch.no_grad():
                model(input_ids)
            if (i + 1) % 50 == 0:
                print(f"  Calibration [{i + 1}/{len(calib_batches)}]")
        print("Calibration complete")

    # ── Quantize ──
    print("Starting NVFP4 quantization...")
    t0 = time.time()
    model = mtq.quantize(model, config, forward_loop=forward_loop)
    elapsed = time.time() - t0
    print(f"Quantization done in {elapsed:.1f}s")

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving to {OUTPUT_DIR}...")
    t0 = time.time()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved in {time.time() - t0:.1f}s")

    # ── Summary ──
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".safetensors")
    )
    print(f"\nModel size: {total_size / 1e9:.2f} GB")
    print(f"Output: {OUTPUT_DIR}")
    print("Serve with: --quantization modelopt --trust-remote-code")


if __name__ == "__main__":
    main()
