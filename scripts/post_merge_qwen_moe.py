#!/usr/bin/env python3
"""Post-process merged MoE: rename keys + swap config for text-only serving.

After merge_adapter_qwen_moe.py writes Jarvis_MoE_trading_raw/ (VLM wrapper format),
run this to rename model.language_model.X -> model.X across shards and copy the
proven serving config/tokenizer/chat_template from the already-working Jarvis_MoE.
"""
import json
import os
import glob
import shutil
import time
from safetensors.torch import load_file, save_file

RAW = "/home/brandonv/models/Jarvis_MoE_trading_raw"
OUT = "/home/brandonv/models/Jarvis_MoE_trading"
REF = "/home/brandonv/models/Jarvis_MoE"  # known-good serving layout


def log(msg):
    print(f"[{time.strftime(chr(37)+chr(72)+chr(58)+chr(37)+chr(77)+chr(58)+chr(37)+chr(83))}] {msg}", flush=True)


def rename_key(key):
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key


os.makedirs(OUT, exist_ok=True)

shard_paths = sorted(glob.glob(os.path.join(RAW, "model-*.safetensors")))
log(f"Processing {len(shard_paths)} shards")

for shard_path in shard_paths:
    name = os.path.basename(shard_path)
    log(f"  {name}")
    data = load_file(shard_path)
    renamed = {rename_key(k): v for k, v in data.items()}
    changed = sum(1 for k in data if k != rename_key(k))
    out_path = os.path.join(OUT, name)
    save_file(renamed, out_path)
    log(f"    renamed {changed}/{len(data)} keys -> {out_path}")
    del data, renamed

log("Updating model.safetensors.index.json")
idx_in = os.path.join(RAW, "model.safetensors.index.json")
with open(idx_in) as f:
    idx = json.load(f)
idx["weight_map"] = {rename_key(k): v for k, v in idx["weight_map"].items()}
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump(idx, f, indent=2)

log("Copying serving files from known-good Jarvis_MoE")
for fname in ["config.json", "chat_template.jinja", "tokenizer_config.json",
              "tokenizer.json", "generation_config.json"]:
    src = os.path.join(REF, fname)
    if not os.path.exists(src):
        log(f"  skip {fname} (missing in REF)")
        continue
    shutil.copy2(src, os.path.join(OUT, fname))
    log(f"  copied {fname}")

log(f"=== POST-MERGE COMPLETE: {OUT} ===")
log("Next: update serve script path, restart vLLM")
