#!/usr/bin/env python3
"""Merge Qwen 3.5-35B-A3B MoE DoRA adapter into base model (CPU).

CPU-only to avoid CUDA VM reservation on Spark unified memory.
Output is in VLM wrapper format — run fix_keys + config flatten after.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/home/brandonv/models/Qwen3.5-35B-A3B-base"
ADAPTER = "/home/brandonv/models/trading-dora-adapter-qwen-moe"
OUTPUT = "/home/brandonv/models/Jarvis_MoE_trading_raw"


def log(msg):
    print(f"[{time.strftime(chr(37)+chr(72)+chr(58)+chr(37)+chr(77)+chr(58)+chr(37)+chr(83))}] {msg}", flush=True)


log("=== Loading tokenizer ===")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

log("=== Loading base model (CPU, BF16) ===")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
log(f"Base loaded in {time.time()-t0:.1f}s — class: {type(model).__name__}")

log("=== Loading DoRA adapter ===")
t0 = time.time()
model = PeftModel.from_pretrained(model, ADAPTER)
log(f"Adapter loaded in {time.time()-t0:.1f}s")

log("=== Merging weights ===")
t0 = time.time()
model = model.merge_and_unload()
log(f"Merged in {time.time()-t0:.1f}s")

log(f"=== Saving merged model to {OUTPUT} ===")
t0 = time.time()
model.save_pretrained(OUTPUT, safe_serialization=True, max_shard_size="4GB")
tokenizer.save_pretrained(OUTPUT)
log(f"Saved in {time.time()-t0:.1f}s")

log("=== MERGE COMPLETE ===")
