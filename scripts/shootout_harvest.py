#!/usr/bin/env python3
"""Harvest completions from any OpenAI-compatible model endpoint.

Runs all 20 benchmark prompts, saves full completions + metadata.
Designed for the 27B vs 35B-A3B vs Gemma 4 shootout.

Usage:
    python scripts/shootout_harvest.py \
        --label jarvis_1 \
        --base-url http://localhost:8000 \
        --model /home/brandonv/models/Jarvis_1 \
        --max-tokens 2048

    # For cloud APIs:
    python scripts/shootout_harvest.py \
        --label deepseek \
        --base-url https://api.deepseek.com \
        --model deepseek-reasoner \
        --api-key $DEEPSEEK_API_KEY
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROMPTS_FILE = REPO_ROOT / "benchmarks" / "prompts.json"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

SYSTEM_PROMPT = (
    "You are an expert trader and portfolio manager with deep knowledge of "
    "technical analysis, fundamental analysis, risk management, behavioral "
    "finance, decision science, and quantitative methods. You have studied "
    "the works of Murphy, Elder, Schwager, Taleb, Dalio, Aronson, DePrado, "
    "Kahneman, Tetlock, Marks, and other masters of trading and decision-making. "
    "Provide thorough, specific, and actionable analysis."
)


def load_prompts() -> list[dict]:
    with open(PROMPTS_FILE) as f:
        return json.load(f)


def run_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None,
    temperature: float,
) -> dict:
    """Run a single completion and return metadata."""
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, headers=headers, timeout=300)
    t1 = time.perf_counter()

    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    usage = data.get("usage", {})
    msg = choice["message"]
    content = msg.get("content") or ""
    # llama.cpp/vLLM with reasoning parser split <think>...</think> into a
    # separate field (reasoning_content or reasoning). Concatenate so the
    # full model output is captured for scoring.
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning:
        completion = f"<think>\n{reasoning}\n</think>\n\n{content}"
    else:
        completion = content
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    wall_time = t1 - t0

    return {
        "completion": completion,
        "content_only": content,
        "reasoning_only": reasoning,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "wall_time_s": round(wall_time, 3),
        "tok_per_s": round(completion_tokens / wall_time, 1) if wall_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Harvest model completions for shootout")
    parser.add_argument("--label", required=True, help="Label for this run (e.g., jarvis_1, qwen_moe_base, gemma_base)")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--model", required=True, help="Model name/path for the API")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per completion")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--api-key", default=None, help="API key (optional)")
    args = parser.parse_args()

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_dir = RESULTS_DIR / f"{timestamp}_{args.label}"
    completions_dir = out_dir / "completions"
    completions_dir.mkdir(parents=True, exist_ok=True)

    # Warmup
    print("Warmup request...")
    try:
        run_completion(args.base_url, args.model, "Hello", 20, args.api_key, 0.3)
    except Exception as e:
        print(f"Warmup failed: {e}")
        print("Is the model server running?")
        sys.exit(1)

    # Run all prompts
    results = []
    total_tokens = 0
    total_time = 0.0

    for i, p in enumerate(prompts):
        pid = p["prompt_id"]
        cat = p["category"]
        print(f"[{i+1}/{len(prompts)}] {pid}...", end=" ", flush=True)

        try:
            r = run_completion(
                args.base_url, args.model, p["content"],
                args.max_tokens, args.api_key, args.temperature,
            )
            print(f"{r['completion_tokens']} tokens, {r['tok_per_s']} tok/s")

            # Save individual completion
            comp_file = completions_dir / f"{i:02d}_{pid}.txt"
            comp_file.write_text(r["completion"], encoding="utf-8")

            results.append({
                "prompt_id": pid,
                "category": cat,
                "prompt": p["content"],
                **r,
            })
            total_tokens += r["completion_tokens"]
            total_time += r["wall_time_s"]

        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "prompt_id": pid,
                "category": cat,
                "prompt": p["content"],
                "completion": f"ERROR: {e}",
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "wall_time_s": 0,
                "tok_per_s": 0,
            })

    # Save harvest results
    harvest = {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "system_prompt": SYSTEM_PROMPT,
        "total_prompts": len(prompts),
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 2),
        "avg_tok_per_s": round(total_tokens / total_time, 1) if total_time > 0 else 0,
        "trials": results,
    }

    harvest_file = out_dir / "harvest_results.json"
    with open(harvest_file, "w") as f:
        json.dump(harvest, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Label:       {args.label}")
    print(f"Model:       {args.model}")
    print(f"Prompts:     {len(prompts)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time:  {total_time:.1f}s")
    print(f"Avg tok/s:   {total_tokens / total_time:.1f}" if total_time > 0 else "N/A")
    print(f"Output:      {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
