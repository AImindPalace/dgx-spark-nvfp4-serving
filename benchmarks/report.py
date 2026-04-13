"""Output generation for benchmark runs.

Writes JSON and plain-text output files to a dated results directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmarks.capture import Environment, ServeFlags
from benchmarks.client import TrialResult
from benchmarks.stats import Stats


# ---------------------------------------------------------------------------
# Internal helpers (patched in tests)
# ---------------------------------------------------------------------------

def _today_utc() -> str:
    """Return today's UTC date as YYYY-MM-DD (extracted for test patching)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_utc_iso() -> str:
    """Return current UTC timestamp as ISO 8601 string ending in Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

_DEFAULT_RESULTS_ROOT = Path(__file__).parent / "results"


def ensure_output_dir(label: str, results_root: Optional[Path] = None) -> Path:
    """Create and return a dated output directory for this benchmark run.

    Directory name: ``{YYYY-MM-DD}_{label}`` under *results_root*
    (defaults to ``benchmarks/results/``).
    """
    root = results_root if results_root is not None else _DEFAULT_RESULTS_ROOT
    date = _today_utc()
    output_dir = root / f"{date}_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# config.json
# ---------------------------------------------------------------------------

def write_config(
    output_dir: Path,
    label: str,
    phase: str,
    optimization: Optional[str],
    serve_flags: ServeFlags,
    environment: Environment,
    prompt_ids: Optional[list[str]] = None,
    prompts_hash: Optional[str] = None,
) -> Path:
    """Write ``config.json`` describing this benchmark run.

    Includes prompt suite metadata so every run records exactly which
    prompts were used (``prompt_ids``) and a sha256 hash of prompts.json
    for quick version comparison across runs.

    Returns the Path to the written file.
    """
    data = {
        "label": label,
        "timestamp": _now_utc_iso(),
        "phase": phase,
        "optimization": optimization,
        "serve_flags": serve_flags.to_dict(),
        "environment": environment.to_dict(),
        "prompts": {
            "count": len(prompt_ids) if prompt_ids else 0,
            "ids": prompt_ids or [],
            "suite_hash": prompts_hash or "",
        },
    }
    out_path = output_dir / "config.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# {mode}_results.json
# ---------------------------------------------------------------------------

def write_results(
    output_dir: Path,
    label: str,
    mode: str,
    trials: list[TrialResult],
    tok_stats: Optional[Stats],
    ttft_stats: Optional[Stats],
) -> Path:
    """Write ``{mode}_results.json`` with trial data and aggregate stats.

    Schema::

        {
            "label": ...,
            "mode": ...,
            "trials": [...],
            "stats": {
                "tok_per_s": {...},   # omitted when tok_stats is None
                "ttft_ms": {...}      # omitted when ttft_stats is None
            }
        }

    When both stats are None the ``"stats"`` key is an empty dict.
    Returns the Path to the written file.
    """
    stats_dict: dict = {}
    if tok_stats is not None:
        stats_dict["tok_per_s"] = tok_stats.to_dict()
    if ttft_stats is not None:
        stats_dict["ttft_ms"] = ttft_stats.to_dict()

    data = {
        "label": label,
        "mode": mode,
        "trials": [t.to_dict() for t in trials],
        "stats": stats_dict,
    }
    out_path = output_dir / f"{mode}_results.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# quality.json
# ---------------------------------------------------------------------------

def write_quality(
    output_dir: Path,
    baseline_label: str,
    test_label: str,
    baseline_trials: list[TrialResult],
    test_trials: list[TrialResult],
) -> Path:
    """Write ``quality.json`` comparing baseline vs. test token counts.

    Trials are matched by ``prompt_id``. Unmatched baseline prompts are
    skipped.

    ``token_delta_pct`` = ``round((test_tokens - baseline_tokens) /
    baseline_tokens * 100, 1)`` per matched prompt.

    Schema::

        {
            "baseline_label": ...,
            "test_label": ...,
            "comparisons": [
                {
                    "prompt_id": ...,
                    "category": ...,
                    "baseline_tokens": ...,
                    "test_tokens": ...,
                    "token_delta_pct": ...,
                    "notes": ""
                },
                ...
            ],
            "overall_assessment": "",
            "degradation_detected": false
        }

    Returns the Path to the written file.
    """
    # Index test trials by prompt_id for O(1) lookup
    test_by_id: dict[str, TrialResult] = {t.prompt_id: t for t in test_trials}

    comparisons = []
    for bt in baseline_trials:
        tt = test_by_id.get(bt.prompt_id)
        if tt is None:
            continue
        delta_pct = round(
            (tt.tokens_generated - bt.tokens_generated) / bt.tokens_generated * 100,
            1,
        )
        comparisons.append(
            {
                "prompt_id": bt.prompt_id,
                "category": bt.category,
                "baseline_tokens": bt.tokens_generated,
                "test_tokens": tt.tokens_generated,
                "token_delta_pct": delta_pct,
                "notes": "",
            }
        )

    data = {
        "baseline_label": baseline_label,
        "test_label": test_label,
        "comparisons": comparisons,
        "overall_assessment": "",
        "degradation_detected": False,
    }
    out_path = output_dir / "quality.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# summary.txt
# ---------------------------------------------------------------------------

def write_summary(
    output_dir: Path,
    label: str,
    optimization: Optional[str],
    phase: str,
    tok_stats: Optional[Stats],
    ttft_stats: Optional[Stats],
    memory_gb: float,
    baseline_tok_mean: Optional[float] = None,
    baseline_ttft_mean: Optional[float] = None,
    baseline_memory_gb: Optional[float] = None,
) -> Path:
    """Write ``summary.txt`` — human-readable benchmark summary.

    Format (with baseline)::

        Experiment: {optimization or label} ({phase})
        Date: YYYY-MM-DD
        Baseline: {baseline_tok_mean} tok/s | This run: {mean} tok/s (+{delta}%)
        TTFT: {mean}ms (baseline: {baseline_ttft_mean}ms)
        Memory: {memory_gb} GB (baseline: {baseline_memory_gb} GB, +{delta} GB)
        Status: PASS

    Format (no baseline)::

        Experiment: {optimization or label} ({phase})
        Date: YYYY-MM-DD
        Throughput: {mean} tok/s (±{std})
        TTFT: {mean}ms (±{std})
        Memory: {memory_gb} GB
        Status: PASS

    When tok_stats or ttft_stats is None the corresponding line is omitted.
    Returns the Path to the written file.
    """
    experiment_name = optimization if optimization is not None else label
    lines: list[str] = [
        f"Experiment: {experiment_name} ({phase})",
        f"Date: {_today_utc()}",
    ]

    has_regression = False

    if tok_stats is not None:
        if baseline_tok_mean is not None:
            delta_pct = round(
                (tok_stats.mean - baseline_tok_mean) / baseline_tok_mean * 100, 1
            )
            sign = "+" if delta_pct >= 0 else ""
            lines.append(
                f"Baseline: {baseline_tok_mean} tok/s | This run: {tok_stats.mean} tok/s ({sign}{delta_pct}%)"
            )
            if delta_pct < 0:
                has_regression = True
        else:
            lines.append(f"Throughput: {tok_stats.mean} tok/s (±{tok_stats.std})")

    if ttft_stats is not None:
        if baseline_ttft_mean is not None:
            lines.append(f"TTFT: {ttft_stats.mean}ms (baseline: {baseline_ttft_mean}ms)")
        else:
            lines.append(f"TTFT: {ttft_stats.mean}ms (±{ttft_stats.std})")

    if baseline_memory_gb is not None:
        mem_delta = round(memory_gb - baseline_memory_gb, 1)
        sign = "+" if mem_delta >= 0 else ""
        lines.append(
            f"Memory: {memory_gb} GB (baseline: {baseline_memory_gb} GB, {sign}{mem_delta} GB)"
        )
    else:
        lines.append(f"Memory: {memory_gb} GB")

    lines.append(f"Status: {'REGRESS' if has_regression else 'PASS'}")

    out_path = output_dir / "summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
