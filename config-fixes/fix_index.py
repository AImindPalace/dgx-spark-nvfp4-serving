#!/usr/bin/env python3
"""
Fix model.safetensors.index.json after nvidia-modelopt export_hf_checkpoint().

ModelOpt may create an index file referencing shard filenames
(model-00001-of-00002.safetensors, etc.) while the actual output
is a single model.safetensors file. This causes vLLM to fail with
"Cannot find any model weights".

Usage:
    python fix_index.py /path/to/your/model/

This will update the index in-place. A backup is saved as
model.safetensors.index.json.bak.
"""

import json
import shutil
import sys
from pathlib import Path


def fix_index(model_dir: str) -> None:
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"
    safetensors_path = model_path / "model.safetensors"

    if not index_path.exists():
        print(f"No index file found at {index_path}, nothing to fix.")
        return

    if not safetensors_path.exists():
        print(f"No model.safetensors found at {safetensors_path}.")
        print("This script expects a single safetensors file from modelopt export.")
        return

    index = json.load(open(index_path))
    weight_map = index.get("weight_map", {})

    # Check if already correct
    files_referenced = set(weight_map.values())
    if files_referenced == {"model.safetensors"}:
        print("Index already points to model.safetensors. Nothing to fix.")
        return

    # Check for nonexistent shard files
    missing = [f for f in files_referenced if not (model_path / f).exists()]
    if not missing:
        print(f"All referenced files exist: {files_referenced}. No fix needed.")
        return

    print(f"Index references {len(files_referenced)} file(s): {files_referenced}")
    print(f"Missing files: {missing}")
    print(f"Redirecting all {len(weight_map)} entries to model.safetensors...")

    # Backup
    shutil.copy2(index_path, index_path.with_suffix(".json.bak"))

    # Fix
    index["weight_map"] = {k: "model.safetensors" for k in weight_map}
    json.dump(index, open(index_path, "w"), indent=2)

    print(f"Fixed. Backup saved to {index_path.with_suffix('.json.bak')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/model/")
        sys.exit(1)
    fix_index(sys.argv[1])
