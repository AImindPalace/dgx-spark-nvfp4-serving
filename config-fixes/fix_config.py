#!/usr/bin/env python3
"""
Fix config.json for serving a ModelOpt NVFP4 Qwen3.5 checkpoint on DGX Spark.

ModelOpt's export_hf_checkpoint() may produce a flattened config or you may
have manually changed model_type to "qwen3_next" based on other guides.
Both cause vLLM to dequantize NVFP4 weights to FP32 (~112 GB instead of ~19 GB).

This script restructures config.json into the VLM wrapper format that vLLM
expects for Qwen3.5 models:
  - architectures: ["Qwen3_5ForConditionalGeneration"]
  - model_type: "qwen3_5"
  - Model params nested under text_config (model_type: "qwen3_5_text")
  - vision_config section present (required by vLLM's model registry)

Usage:
    python fix_config.py /path/to/your/model/

This will update config.json in-place. A backup is saved as config.json.bak.
"""

import json
import shutil
import sys
from pathlib import Path

# Standard Qwen3.5 vision config (required even for text-only serving)
VISION_CONFIG = {
    "model_type": "qwen3_5",
    "depth": 27,
    "hidden_size": 1152,
    "hidden_act": "gelu_pytorch_tanh",
    "intermediate_size": 4304,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 5120,
    "num_position_embeddings": 2304,
    "initializer_range": 0.02,
}

# Fields that belong at the top level (not in text_config)
TOP_LEVEL_FIELDS = {
    "architectures", "_name_or_path", "transformers_version",
    "quantization_config", "tie_word_embeddings",
    "image_token_id", "video_token_id",
    "vision_start_token_id", "vision_end_token_id",
    "vision_config", "text_config", "model_type",
}


def fix_config(model_dir: str) -> None:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"

    if not config_path.exists():
        print(f"No config.json found at {config_path}")
        return

    c = json.load(open(config_path))

    # Check if already in VLM wrapper format
    if (c.get("model_type") == "qwen3_5"
            and c.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
            and "text_config" in c
            and "vision_config" in c):
        print("Config already in VLM wrapper format. Nothing to fix.")
        return

    print(f"Current: architectures={c.get('architectures')}, model_type={c.get('model_type')}")

    # Backup
    shutil.copy2(config_path, config_path.with_suffix(".json.bak"))

    # If there's already a text_config, use it; otherwise build from top-level fields
    if "text_config" in c:
        text_config = c["text_config"]
    else:
        text_config = {k: v for k, v in c.items() if k not in TOP_LEVEL_FIELDS}

    text_config["model_type"] = "qwen3_5_text"

    # Build the wrapper config
    new_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "_name_or_path": c.get("_name_or_path", "Qwen/Qwen3.5-27B"),
        "transformers_version": c.get("transformers_version", "5.4.0"),
        "tie_word_embeddings": c.get("tie_word_embeddings", False),
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "vision_config": VISION_CONFIG,
        "text_config": text_config,
    }

    # Preserve quantization_config
    if "quantization_config" in c:
        new_config["quantization_config"] = c["quantization_config"]

    json.dump(new_config, open(config_path, "w"), indent=2)
    print(f"Fixed: architectures=['Qwen3_5ForConditionalGeneration'], model_type='qwen3_5'")
    print(f"Backup saved to {config_path.with_suffix('.json.bak')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/model/")
        sys.exit(1)
    fix_config(sys.argv[1])
