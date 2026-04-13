#!/usr/bin/env python3
"""Post-process merged 27B: rename keys + inject MTP weights from base.

transformers 5.x save_pretrained strips mtp.* keys (not in Qwen3_5ForCausalLM
class definition). This script restores them from the base model + renames
model.language_model.* -> model.* for serving.
"""
import json
import os
import glob
import shutil
import time
from safetensors.torch import load_file, save_file

RAW = "/home/brandonv/models/Jarvis_27B_trading_raw"
OUT = "/home/brandonv/models/Jarvis_27B_trading"
BASE = "/home/brandonv/models/Qwen3.5-27B"
REF = "/home/brandonv/models/Qwen3.5-27B-trading"  # known-good BF16 serving layout (Cycle 1)


def log(msg):
    print(f"[{time.strftime(chr(37)+chr(72)+chr(58)+chr(37)+chr(77)+chr(58)+chr(37)+chr(83))}] {msg}", flush=True)


def rename_key(k):
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    return k


os.makedirs(OUT, exist_ok=True)

# Step 1: Collect MTP weights from base
log("Collecting MTP weights from base")
mtp_weights = {}
for shard in sorted(glob.glob(os.path.join(BASE, "model.safetensors*.safetensors"))):
    d = load_file(shard)
    for k, v in d.items():
        if "mtp" in k.lower():
            mtp_weights[k] = v.clone()
    del d
log(f"  collected {len(mtp_weights)} mtp tensors")

# Step 2: Process merged shards — rename keys
raw_shards = sorted(glob.glob(os.path.join(RAW, "model-*.safetensors")))
log(f"Processing {len(raw_shards)} merged shards")
for shard_path in raw_shards:
    name = os.path.basename(shard_path)
    data = load_file(shard_path)
    renamed = {rename_key(k): v for k, v in data.items()}
    changed = sum(1 for k in data if k != rename_key(k))
    save_file(renamed, os.path.join(OUT, name))
    log(f"  {name} — renamed {changed}/{len(data)} keys")
    del data, renamed

# Step 3: Write MTP weights as new shard
mtp_shard_name = f"model-{len(raw_shards)+1:05d}-of-{len(raw_shards)+1:05d}.safetensors"
# Actually safer: use a unique name and update index
mtp_shard_name = f"model-mtp.safetensors"
save_file(mtp_weights, os.path.join(OUT, mtp_shard_name))
log(f"  {mtp_shard_name} — wrote {len(mtp_weights)} mtp keys")

# Step 4: Rebuild index.json with renamed main keys + mtp keys
log("Building model.safetensors.index.json")
with open(os.path.join(RAW, "model.safetensors.index.json")) as f:
    idx = json.load(f)
weight_map = {rename_key(k): v for k, v in idx["weight_map"].items()}
for k in mtp_weights:
    weight_map[k] = mtp_shard_name
idx["weight_map"] = weight_map
# metadata.total_size may be wrong now — recompute
total_size = 0
for shard in sorted(glob.glob(os.path.join(OUT, "model-*.safetensors"))):
    total_size += os.path.getsize(shard)
idx["metadata"] = idx.get("metadata", {})
idx["metadata"]["total_size"] = total_size
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump(idx, f, indent=2)
log(f"  index: {len(weight_map)} keys, total_size={total_size/(1024**3):.1f} GiB")

# Step 5: Copy serving files from known-good Cycle 1 merge
log("Copying serving files from Qwen3.5-27B-trading")
for fname in ["config.json", "chat_template.jinja", "tokenizer_config.json",
              "tokenizer.json", "generation_config.json", "merges.txt", "vocab.json"]:
    src = os.path.join(REF, fname)
    if not os.path.exists(src):
        log(f"  skip {fname}")
        continue
    shutil.copy2(src, os.path.join(OUT, fname))
    log(f"  copied {fname}")

log(f"=== POST-MERGE COMPLETE: {OUT} ===")
