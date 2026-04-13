#!/usr/bin/env python3
"""Minimal post-merge: inject MTP weights + strip mrope from config. No key rename.

Keeps model.language_model.* weight keys as-is (vLLM --language-model-only handles them).
"""
import json, os, glob, shutil, sys
from safetensors.torch import load_file, save_file

RAW = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Jarvis_27B_trading_raw"
BASE = sys.argv[2] if len(sys.argv) > 2 else "/workspace/Qwen3.5-27B"

print(f"RAW: {RAW}")
print(f"BASE: {BASE}")

# Extract MTP weights from base (transformers 5.5 save_pretrained strips mtp.*)
print("Extracting MTP weights from base...")
mtp_weights = {}
for shard in sorted(glob.glob(os.path.join(BASE, "model.safetensors*.safetensors"))):
    d = load_file(shard)
    for k, v in d.items():
        if "mtp" in k.lower():
            mtp_weights[k] = v.clone()
    del d
print(f"  collected {len(mtp_weights)} mtp tensors")

if mtp_weights:
    # Append MTP as extra shard to RAW
    mtp_shard = os.path.join(RAW, "model-mtp.safetensors")
    save_file(mtp_weights, mtp_shard)
    print(f"  wrote {mtp_shard}")

    # Update index.json
    idx_path = os.path.join(RAW, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    for k in mtp_weights:
        idx["weight_map"][k] = "model-mtp.safetensors"
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    print(f"  index updated, now {len(idx[chr(119)+chr(101)+chr(105)+chr(103)+chr(104)+chr(116)+chr(95)+chr(109)+chr(97)+chr(112)])} keys")

# Strip mrope from config (vLLM rejects mrope_section/mrope_interleaved)
cfg_path = os.path.join(RAW, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)

modified = False
rope = cfg.get("rope_parameters") or {}
if "mrope_interleaved" in rope:
    del rope["mrope_interleaved"]; modified = True
if "mrope_section" in rope:
    del rope["mrope_section"]; modified = True

# Also check nested text_config
tc = cfg.get("text_config")
if tc:
    tc_rope = tc.get("rope_parameters") or {}
    if "mrope_interleaved" in tc_rope:
        del tc_rope["mrope_interleaved"]; modified = True
    if "mrope_section" in tc_rope:
        del tc_rope["mrope_section"]; modified = True

if modified:
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Stripped mrope_* from config")
else:
    print("No mrope fields to strip")

print("POST-MERGE MINIMAL DONE")
