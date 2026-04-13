#!/usr/bin/env python3
"""Post-modelopt-export fixes for Jarvis_2 NVFP4 deploy.

modelopt's `mte.export_hf_checkpoint()` produces a technically-correct NVFP4
model, but the output needs three patches before vLLM (specifically the
Spark NVFP4 serving path that Jarvis_1 uses) will load it:

1. Config architecture/model_type is flat (`Qwen3_5ForCausalLM` /
   `qwen3_5_text`), but vLLM requires the VLM-wrapper format
   (`Qwen3_5ForConditionalGeneration` / `qwen3_5`) with nested text_config
   + vision_config. Overlay Jarvis_1's working config.json as a template,
   preserving the exported `quantization_config`.

2. MTP head weights (`mtp.*`) are missing from the output even though we
   excluded them from quantization — modelopt's exporter only serializes
   weights that are part of the HF model class's registered state_dict,
   and `Qwen3_5ForCausalLM` doesn't expose MTP. Inject the 15 MTP tensors
   from the base Qwen3.5-27B model as a sidecar `model-mtp.safetensors`
   shard. (DoRA never touched MTP, so base MTP weights are byte-identical
   to what a correctly-exported merge+quant would have produced.)

3. The single-file safetensors needs a `.safetensors.index.json` mapping
   so vLLM's multi-file weight loader handles the sidecar shard. Generate
   one that maps all main-model keys to `model.safetensors` and all
   `mtp.*` keys to `model-mtp.safetensors`.

Usage:
    python post_export_jarvis2.py \\
        --jarvis2-dir /home/brandonv/models/Jarvis_2 \\
        --template-config /home/brandonv/models/Jarvis_1/config.json \\
        --base-model /home/brandonv/models/Qwen3.5-27B

Verifies final state is vLLM-compatible (matches Jarvis_1's layout).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
except ImportError:
    print("Need safetensors. Run: pip install safetensors")
    sys.exit(1)


def fix_config(jarvis2_dir: Path, template_config: Path) -> None:
    """Overlay template (Jarvis_1 VLM-wrapper) config, keep modelopt's quant_config."""
    j2_path = jarvis2_dir / "config.json"
    j2 = json.load(j2_path.open())
    j1 = json.load(template_config.open())

    print(f"config.json before: architectures={j2['architectures']}, "
          f"model_type={j2.get('model_type')}")

    qcfg = j2.get("quantization_config")
    if qcfg is None:
        print("  WARNING: no quantization_config in exported model — modelopt export may have failed")
        return

    # Start from template (has proper VLM wrapper structure), swap in the quant config.
    out = dict(j1)
    out["quantization_config"] = qcfg
    if "transformers_version" in j2:
        out["transformers_version"] = j2["transformers_version"]

    json.dump(out, j2_path.open("w"), indent=2)
    print(f"config.json after:  architectures={out['architectures']}, "
          f"model_type={out['model_type']}, "
          f"has text_config={('text_config' in out)}, "
          f"has vision_config={('vision_config' in out)}")


def extract_mtp_from_base(base_dir: Path) -> dict:
    """Pull mtp.* tensors out of the HF base (split across multiple shards)."""
    mtp: dict = {}
    shards = sorted(glob.glob(str(base_dir / "model.safetensors*.safetensors")))
    if not shards:
        shards = sorted(glob.glob(str(base_dir / "model-*.safetensors")))
    for shard in shards:
        d = load_file(shard)
        for k, v in d.items():
            if "mtp" in k.lower():
                mtp[k] = v.clone()
        del d
    return mtp


def write_mtp_sidecar(jarvis2_dir: Path, mtp: dict) -> Path:
    sidecar = jarvis2_dir / "model-mtp.safetensors"
    save_file(mtp, str(sidecar))
    print(f"wrote {sidecar.name}: {len(mtp)} tensors, "
          f"{sidecar.stat().st_size / 1e9:.2f} GB")
    return sidecar


def build_index(jarvis2_dir: Path, mtp_keys: list[str]) -> None:
    """Generate model.safetensors.index.json routing main keys → model.safetensors
    and mtp.* keys → model-mtp.safetensors."""
    main = jarvis2_dir / "model.safetensors"
    sidecar = jarvis2_dir / "model-mtp.safetensors"
    weight_map: dict[str, str] = {}

    with safe_open(str(main), framework="pt") as f:
        for k in f.keys():
            weight_map[k] = "model.safetensors"

    for k in mtp_keys:
        weight_map[k] = "model-mtp.safetensors"

    total_size = main.stat().st_size + (sidecar.stat().st_size if sidecar.exists() else 0)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}

    idx_path = jarvis2_dir / "model.safetensors.index.json"
    json.dump(index, idx_path.open("w"), indent=2)
    print(f"wrote {idx_path.name}: {len(weight_map)} keys, total_size={total_size / 1e9:.1f} GB")


def verify(jarvis2_dir: Path) -> None:
    """Sanity-check final state matches Jarvis_1 layout expectations."""
    cfg = json.load((jarvis2_dir / "config.json").open())
    assert cfg["architectures"] == ["Qwen3_5ForConditionalGeneration"], \
        f"arch not Qwen3_5ForConditionalGeneration: {cfg['architectures']}"
    assert cfg["model_type"] == "qwen3_5", f"model_type not qwen3_5: {cfg['model_type']}"
    assert "text_config" in cfg, "missing text_config"
    assert "vision_config" in cfg, "missing vision_config"
    assert "quantization_config" in cfg, "missing quantization_config"
    assert cfg["quantization_config"]["quant_method"] == "modelopt"

    idx = json.load((jarvis2_dir / "model.safetensors.index.json").open())
    wm = idx["weight_map"]
    mtp_keys = [k for k in wm if k.startswith("mtp.")]
    assert len(mtp_keys) > 0, "no mtp.* keys in index"
    assert all(wm[k] == "model-mtp.safetensors" for k in mtp_keys), \
        "mtp keys not routed to sidecar"

    print("\n=== POST-EXPORT VERIFY: OK ===")
    print(f"  architectures:        {cfg['architectures'][0]}")
    print(f"  model_type:           {cfg['model_type']}")
    print(f"  quantization_config:  {cfg['quantization_config']['quant_algo']}")
    print(f"  weight_map keys:      {len(wm)} ({len(mtp_keys)} mtp.*)")
    print(f"  sidecar present:      {(jarvis2_dir / 'model-mtp.safetensors').exists()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jarvis2-dir", required=True,
                   help="Directory containing modelopt export (model.safetensors, config.json, etc.)")
    p.add_argument("--template-config", required=True,
                   help="Path to Jarvis_1's config.json (the known-working VLM-wrapper template)")
    p.add_argument("--base-model", required=True,
                   help="HF base model dir with mtp.* weights (e.g. ~/models/Qwen3.5-27B)")
    args = p.parse_args()

    j2 = Path(args.jarvis2_dir)
    template = Path(args.template_config)
    base = Path(args.base_model)

    assert j2.is_dir(), f"missing: {j2}"
    assert template.is_file(), f"missing: {template}"
    assert base.is_dir(), f"missing: {base}"

    print("=" * 60)
    print("Step 1: rewrite config.json → VLM wrapper format")
    print("=" * 60)
    fix_config(j2, template)

    print()
    print("=" * 60)
    print("Step 2: extract mtp.* weights from base + write sidecar")
    print("=" * 60)
    mtp = extract_mtp_from_base(base)
    print(f"extracted {len(mtp)} mtp tensors from {base.name}/")
    if not mtp:
        print("WARNING: no mtp.* tensors found in base. Check paths.")
        sys.exit(1)
    write_mtp_sidecar(j2, mtp)

    print()
    print("=" * 60)
    print("Step 3: generate model.safetensors.index.json")
    print("=" * 60)
    build_index(j2, list(mtp.keys()))

    print()
    verify(j2)


if __name__ == "__main__":
    main()
