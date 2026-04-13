#!/usr/bin/env python3
"""Merge Gemma 4 26B-A4B DoRA adapter into base model (CPU).

Key Gemma 4 specifics:
- attn_implementation must be sdpa (Gemma 4 head_dim > 256, exceeds flash-attn v2 limit)
- ClippableLinear must be unwrapped before PEFT (matches training setup; clamp
  bounds zero out LoRA gradients during training AND skew merge math)
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/home/brandonv/models/gemma-4-26B-A4B-it"
ADAPTER = "/home/brandonv/models/trading-dora-adapter-gemma-moe"
OUTPUT = "/home/brandonv/models/Jarvis_Gemma_trading_raw"


def log(msg):
    print(f"[{time.strftime(chr(37)+chr(72)+chr(58)+chr(37)+chr(77)+chr(58)+chr(37)+chr(83))}] {msg}", flush=True)


log("=== Loading tokenizer ===")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

log("=== Loading base model (CPU, BF16, SDPA) ===")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    attn_implementation="sdpa",
)
log(f"Base loaded in {time.time()-t0:.1f}s -- class: {type(model).__name__}")

log("=== Unwrapping Gemma4ClippableLinear -> nn.Linear ===")
unwrap_count = 0
for name, module in list(model.named_modules()):
    if type(module).__name__ == "Gemma4ClippableLinear":
        inner = module.linear
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            setattr(parent, parts[1], inner)
            unwrap_count += 1
log(f"Unwrapped {unwrap_count} ClippableLinear modules")

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
