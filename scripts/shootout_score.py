#!/usr/bin/env python3
"""Blind quality scoring of model completions using Claude Opus.

Compares completions from multiple models against a reference (DeepSeek-Reasoner).
Scores are blind — Opus doesn't know which model produced which completion.

Usage:
    python scripts/shootout_score.py \
        --reference benchmarks/results/2026-04-03_deepseek-reasoner \
        --candidates \
            benchmarks/results/2026-04-09_jarvis_1 \
            benchmarks/results/2026-04-09_qwen_moe_base \
            benchmarks/results/2026-04-09_gemma_base \
        --output benchmarks/results/shootout_scores.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("pip install anthropic")
    sys.exit(1)

SCORING_PROMPT = """You are an expert evaluator assessing trading/finance model responses.

You will see a PROMPT, a REFERENCE response (from an established model), and {n_candidates} CANDIDATE responses labeled A through {last_label}. The candidates are in random order — you do NOT know which model produced which response.

Score each candidate on these dimensions (1-10 scale):

1. **Accuracy** — Are the facts, calculations, and market mechanics correct?
2. **Depth** — Does it go beyond surface-level analysis? Does it show genuine domain expertise?
3. **Reasoning** — Is the logic chain sound? Are conclusions well-supported by evidence?
4. **Specificity** — Does it give concrete numbers, price levels, position sizes — not vague generalities?
5. **Domain Knowledge** — Does it reference relevant frameworks, authors, and concepts from trading literature?
6. **Actionability** — Could a trader actually use this analysis to make a decision?

Respond with ONLY a JSON object (no markdown, no explanation) in this format:
{{
  "scores": {{
    "A": {{"accuracy": N, "depth": N, "reasoning": N, "specificity": N, "domain_knowledge": N, "actionability": N, "total": N, "brief_note": "one sentence"}},
    "B": {{"accuracy": N, "depth": N, "reasoning": N, "specificity": N, "domain_knowledge": N, "actionability": N, "total": N, "brief_note": "one sentence"}}
  }},
  "ranking": ["B", "A"],
  "reference_comparison": "brief note on how candidates compare to reference"
}}

Total = sum of all 6 scores (max 60).
Ranking = candidates ordered best to worst by total score."""


def load_harvest(result_dir: Path) -> dict:
    """Load harvest results from a results directory."""
    harvest_file = result_dir / "harvest_results.json"
    if harvest_file.exists():
        with open(harvest_file) as f:
            return json.load(f)

    # Fallback: load individual completions
    completions_dir = result_dir / "completions"
    if not completions_dir.exists():
        raise FileNotFoundError(f"No harvest_results.json or completions/ in {result_dir}")

    trials = []
    for f in sorted(completions_dir.glob("*.txt")):
        parts = f.stem.split("_", 1)
        pid = parts[1] if len(parts) > 1 else f.stem
        trials.append({
            "prompt_id": pid,
            "completion": f.read_text(encoding="utf-8"),
        })

    return {"label": result_dir.name, "trials": trials}


def score_prompt(
    client: anthropic.Anthropic,
    prompt_text: str,
    reference_completion: str,
    candidates: list[tuple[str, str]],  # [(label, completion), ...]
) -> dict:
    """Score candidates for a single prompt using Opus."""

    # Shuffle candidates for blind scoring
    shuffled = list(candidates)
    random.shuffle(shuffled)
    label_map = {}  # letter -> real label
    letters = "ABCDEFGH"

    candidate_text = ""
    for i, (real_label, completion) in enumerate(shuffled):
        letter = letters[i]
        label_map[letter] = real_label
        candidate_text += f"\n--- CANDIDATE {letter} ---\n{completion}\n"

    n = len(shuffled)
    last_label = letters[n - 1]

    user_msg = f"""PROMPT:
{prompt_text}

--- REFERENCE RESPONSE ---
{reference_completion}

{candidate_text}

Score each candidate (A through {last_label}) on the 6 dimensions. Return ONLY the JSON object."""

    formatted_system = SCORING_PROMPT.format(n_candidates=n, last_label=last_label)

    resp = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        temperature=0,
        system=formatted_system,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = resp.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]

    scores = json.loads(raw)

    # Remap letter labels back to real model labels
    remapped_scores = {}
    for letter, score_data in scores.get("scores", {}).items():
        real_label = label_map.get(letter, letter)
        remapped_scores[real_label] = score_data

    remapped_ranking = [label_map.get(l, l) for l in scores.get("ranking", [])]

    return {
        "scores": remapped_scores,
        "ranking": remapped_ranking,
        "reference_comparison": scores.get("reference_comparison", ""),
        "label_map": label_map,
    }


def main():
    parser = argparse.ArgumentParser(description="Blind quality scoring with Claude Opus")
    parser.add_argument("--reference", required=True, help="Path to reference results dir")
    parser.add_argument("--candidates", nargs="+", required=True, help="Paths to candidate results dirs")
    parser.add_argument("--output", default=None, help="Output file for scores")
    parser.add_argument("--prompts", default=None, help="Only score these prompt_ids (comma-separated)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load reference
    ref_dir = Path(args.reference)
    ref_data = load_harvest(ref_dir)
    ref_by_pid = {t["prompt_id"]: t for t in ref_data["trials"]}
    print(f"Reference: {ref_dir.name} ({len(ref_by_pid)} prompts)")

    # Load candidates
    cand_data = {}
    for cand_path in args.candidates:
        cand_dir = Path(cand_path)
        data = load_harvest(cand_dir)
        label = data.get("label", cand_dir.name)
        cand_data[label] = {t["prompt_id"]: t for t in data["trials"]}
        print(f"Candidate: {label} ({len(cand_data[label])} prompts)")

    # Filter prompts if specified
    prompt_ids = list(ref_by_pid.keys())
    if args.prompts:
        filter_pids = set(args.prompts.split(","))
        prompt_ids = [p for p in prompt_ids if p in filter_pids]

    # Score each prompt
    all_scores = []
    model_totals = {label: [] for label in cand_data}

    for i, pid in enumerate(prompt_ids):
        ref_trial = ref_by_pid.get(pid)
        if not ref_trial:
            continue

        print(f"[{i+1}/{len(prompt_ids)}] Scoring {pid}...", end=" ", flush=True)

        candidates = []
        for label, trials in cand_data.items():
            trial = trials.get(pid)
            if trial and trial.get("completion") and not trial["completion"].startswith("ERROR"):
                candidates.append((label, trial["completion"]))

        if not candidates:
            print("SKIP (no candidates)")
            continue

        try:
            result = score_prompt(
                client, ref_trial.get("prompt", pid),
                ref_trial["completion"], candidates,
            )
            print(f"Winner: {result['ranking'][0] if result['ranking'] else 'N/A'}")

            for label, score_data in result["scores"].items():
                model_totals[label].append(score_data.get("total", 0))

            all_scores.append({
                "prompt_id": pid,
                "category": ref_trial.get("category", ""),
                **result,
            })

            time.sleep(1)  # Rate limit courtesy

        except Exception as e:
            print(f"FAILED: {e}")
            all_scores.append({"prompt_id": pid, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("AGGREGATE SCORES")
    print(f"{'='*60}")
    summary = {}
    for label, totals in model_totals.items():
        if totals:
            avg = sum(totals) / len(totals)
            summary[label] = {
                "avg_total": round(avg, 1),
                "min": min(totals),
                "max": max(totals),
                "n_scored": len(totals),
            }
            print(f"  {label:30s}  avg={avg:.1f}/60  n={len(totals)}")
    print(f"{'='*60}")

    # Count wins
    win_counts = {label: 0 for label in cand_data}
    for s in all_scores:
        ranking = s.get("ranking", [])
        if ranking:
            win_counts[ranking[0]] = win_counts.get(ranking[0], 0) + 1

    print("\nWIN COUNTS (first place):")
    for label, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:30s}  {wins} wins")

    # Save output
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": ref_dir.name,
        "candidates": list(cand_data.keys()),
        "n_prompts_scored": len(all_scores),
        "summary": summary,
        "win_counts": win_counts,
        "per_prompt": all_scores,
    }

    out_file = Path(args.output) if args.output else (
        Path("benchmarks/results") / f"shootout_scores_{time.strftime('%Y-%m-%d')}.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nScores saved to {out_file}")


if __name__ == "__main__":
    main()
