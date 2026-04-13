"""CLI orchestration for the Jarvis benchmark suite.

Usage:
    python -m benchmarks {benchmark|harvest|full} --label LABEL [options]

Modes:
    benchmark   200 max_tokens, 5 trials per prompt (trial 1 is warmup, excluded from stats)
    harvest     2048 max_tokens, 1 trial per prompt
    full        runs benchmark then harvest in sequence
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

from benchmarks.capture import capture_environment, capture_serve_flags
from benchmarks.client import TrialResult, run_trial
from benchmarks.report import (
    ensure_output_dir,
    write_config,
    write_quality,
    write_results,
    write_summary,
)
from benchmarks.stats import Stats, compute_stats

PROMPTS_FILE = Path(__file__).parent / "prompts.json"

# Minimum tokens reserved for the visible answer in harvest mode.
# Thinking tokens are capped at (max_tokens - MIN_ANSWER_BUDGET) so the
# model cannot exhaust the entire budget on its <think> trace.
MIN_ANSWER_BUDGET = 2048

MODE_CONFIG = {
    "benchmark": {"max_tokens": 200, "trials": 5},
    "harvest": {"max_tokens": 7168, "trials": 1},
}


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(path: Optional[Path] = None) -> list[dict]:
    """Load prompts from *path* (defaults to the bundled prompts.json)."""
    source = path if path is not None else PROMPTS_FILE
    return json.loads(source.read_text())


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with benchmark/harvest/full subcommands."""
    parser = argparse.ArgumentParser(
        description="Jarvis benchmark suite — measure throughput and output quality."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--label", required=True, help="Short label for this run (used in filenames)")
        sub.add_argument("--phase", default="independent", help="Experiment phase (default: independent)")
        sub.add_argument("--optimization", default="", help="Optimization label (e.g. MTP, nvfp4)")
        sub.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
        sub.add_argument("--model", default="/home/brandonv/models/Jarvis_1", help="Model path as registered in vLLM")
        sub.add_argument("--baseline", default=None, help="Label of a previous run to compare against")

    benchmark_sub = subparsers.add_parser("benchmark", help="Run throughput benchmark (5 trials, first is warmup)")
    _add_common_args(benchmark_sub)

    harvest_sub = subparsers.add_parser("harvest", help="Harvest long-form completions (1 trial, 2048 tokens)")
    _add_common_args(harvest_sub)

    full_sub = subparsers.add_parser("full", help="Run benchmark then harvest in sequence")
    _add_common_args(full_sub)

    return parser


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_mode(
    mode: str,
    base_url: str,
    model: str,
    prompts: list[dict],
) -> list[TrialResult]:
    """Run all prompts for a given mode, printing progress as we go.

    Returns the complete list of TrialResult objects (including warmup trials).
    """
    cfg = MODE_CONFIG[mode]
    max_tokens: int = cfg["max_tokens"]
    num_trials: int = cfg["trials"]

    # Cap thinking tokens in harvest mode so the answer isn't starved.
    thinking_budget: int | None = None
    if mode == "harvest" and max_tokens > MIN_ANSWER_BUDGET:
        thinking_budget = max_tokens - MIN_ANSWER_BUDGET

    results: list[TrialResult] = []
    total = len(prompts) * num_trials
    seq = 0

    for prompt in prompts:
        prompt_id = prompt["prompt_id"]
        category = prompt["category"]
        content = prompt["content"]

        for trial_num in range(1, num_trials + 1):
            seq += 1
            is_warmup = mode == "benchmark" and trial_num == 1
            warmup_tag = " (warmup)" if is_warmup else ""
            print(f"[{seq}/{total}] {prompt_id} trial {trial_num}{warmup_tag}")

            result = run_trial(
                base_url=base_url,
                model=model,
                prompt_id=prompt_id,
                category=category,
                content=content,
                max_tokens=max_tokens,
                trial_num=trial_num,
                is_warmup=is_warmup,
                thinking_budget=thinking_budget,
            )
            results.append(result)
            print(f"  → {result.tok_per_s} tok/s, TTFT {result.ttft_ms:.0f}ms")

    return results


# ---------------------------------------------------------------------------
# Statistics aggregation
# ---------------------------------------------------------------------------

def aggregate_stats(
    trials: list[TrialResult],
) -> tuple[Optional[Stats], Optional[Stats]]:
    """Compute tok_per_s and ttft_ms stats from non-warmup trials.

    Returns ``(None, None)`` when there are no active (non-warmup) trials.
    """
    active = [t for t in trials if not t.is_warmup]
    if not active:
        return None, None

    tok_stats = compute_stats([t.tok_per_s for t in active])
    ttft_stats = compute_stats([t.ttft_ms for t in active])
    return tok_stats, ttft_stats


# ---------------------------------------------------------------------------
# Baseline loaders
# ---------------------------------------------------------------------------

def _find_run_dir(label: str) -> Optional[Path]:
    """Search benchmarks/results/ for dirs ending with _{label}.

    Returns the most recent match (sorted by name, last wins), or None.
    """
    results_root = Path("benchmarks") / "results"
    if not results_root.exists():
        return None

    matches = sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and d.name.endswith(f"_{label}")
    )
    return matches[-1] if matches else None


def load_baseline_trials(label: str) -> Optional[list[TrialResult]]:
    """Load harvest TrialResult objects from a previous run's harvest_results.json.

    Returns None when no matching run directory or file is found.
    """
    run_dir = _find_run_dir(label)
    if run_dir is None:
        return None

    harvest_file = run_dir / "harvest_results.json"
    if not harvest_file.exists():
        return None

    data = json.loads(harvest_file.read_text())
    return [TrialResult(**t) for t in data["trials"]]


def load_baseline_stats(label: str, mode: str) -> Optional[dict]:
    """Load the stats dict from a previous run's {mode}_results.json.

    Returns None when no matching run directory or file is found.
    """
    run_dir = _find_run_dir(label)
    if run_dir is None:
        return None

    results_file = run_dir / f"{mode}_results.json"
    if not results_file.exists():
        return None

    data = json.loads(results_file.read_text())
    return data.get("stats")


def load_baseline_memory(label: str) -> Optional[float]:
    """Load memory_before_gb from a previous run's config.json.

    Returns None when no matching run directory or file is found.
    """
    run_dir = _find_run_dir(label)
    if run_dir is None:
        return None

    config_file = run_dir / "config.json"
    if not config_file.exists():
        return None

    data = json.loads(config_file.read_text())
    return data.get("environment", {}).get("memory_before_gb")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """Orchestrate benchmark and/or harvest runs and write output files."""
    args = build_parser().parse_args(argv)

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts. Label: {args.label}")

    # Capture environment once per run
    environment = capture_environment()
    serve_flags = capture_serve_flags()

    # Determine which modes to run
    modes = ["benchmark", "harvest"] if args.command == "full" else [args.command]

    # Create output directory (shared for all modes in this run)
    # Use cwd-relative path so tests (which chdir to tmp_path) land in the right place
    results_root = Path("benchmarks") / "results"
    output_dir = ensure_output_dir(args.label, results_root=results_root)

    # Write config once — includes prompt suite metadata for tracking
    prompt_ids = [p["prompt_id"] for p in prompts]
    prompts_path = PROMPTS_FILE if PROMPTS_FILE.exists() else None
    prompts_hash = (
        hashlib.sha256(prompts_path.read_bytes()).hexdigest()[:12]
        if prompts_path else ""
    )
    write_config(
        output_dir=output_dir,
        label=args.label,
        phase=args.phase,
        optimization=args.optimization or None,
        serve_flags=serve_flags,
        environment=environment,
        prompt_ids=prompt_ids,
        prompts_hash=prompts_hash,
    )

    for mode in modes:
        trials = run_mode(mode, args.base_url, args.model, prompts)
        tok_stats, ttft_stats = aggregate_stats(trials)

        write_results(
            output_dir=output_dir,
            label=args.label,
            mode=mode,
            trials=trials,
            tok_stats=tok_stats,
            ttft_stats=ttft_stats,
        )

        if mode == "benchmark" and tok_stats is not None:
            # Load baseline stats for comparison if requested
            baseline_tok_mean: Optional[float] = None
            baseline_ttft_mean: Optional[float] = None
            if args.baseline:
                base_stats = load_baseline_stats(args.baseline, "benchmark")
                if base_stats:
                    tok_entry = base_stats.get("tok_per_s")
                    ttft_entry = base_stats.get("ttft_ms")
                    if tok_entry:
                        baseline_tok_mean = tok_entry.get("mean")
                    if ttft_entry:
                        baseline_ttft_mean = ttft_entry.get("mean")

            # Memory from last trial
            memory_gb = trials[-1].memory_after_gb if trials else 0.0

            # Baseline memory from config.json
            baseline_memory_gb: Optional[float] = None
            if args.baseline:
                baseline_memory_gb = load_baseline_memory(args.baseline)

            write_summary(
                output_dir=output_dir,
                label=args.label,
                optimization=args.optimization or None,
                phase=args.phase,
                tok_stats=tok_stats,
                ttft_stats=ttft_stats,
                memory_gb=memory_gb,
                baseline_tok_mean=baseline_tok_mean,
                baseline_ttft_mean=baseline_ttft_mean,
                baseline_memory_gb=baseline_memory_gb,
            )

        if mode == "harvest" and args.baseline:
            baseline_trials = load_baseline_trials(args.baseline)
            if baseline_trials is not None:
                write_quality(
                    output_dir=output_dir,
                    baseline_label=args.baseline,
                    test_label=args.label,
                    baseline_trials=baseline_trials,
                    test_trials=trials,
                )

    print(f"Done. Results → {output_dir}")


if __name__ == "__main__":
    main()
