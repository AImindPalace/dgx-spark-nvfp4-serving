"""vLLM streaming client for benchmark trials.

Makes a single streaming POST to vLLM's OpenAI-compatible endpoint, measures
time-to-first-token (TTFT), assembles the full completion, and captures a
post-generation memory snapshot.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import requests

from benchmarks.capture import capture_memory


@dataclass
class TrialResult:
    prompt_id: str
    category: str
    trial: int
    is_warmup: bool
    prompt: str
    completion: str
    reasoning: str
    max_tokens: int
    tokens_generated: int
    ttft_ms: float
    total_time_s: float
    tok_per_s: float
    memory_after_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# SSE line parser
# ---------------------------------------------------------------------------

def parse_sse_line(line: str) -> dict | None:
    """Parse a single SSE line from a vLLM streaming response.

    Returns:
        Parsed JSON dict when the line carries a data payload.
        None for empty lines, non-data lines, and the ``[DONE]`` sentinel.
    """
    if not line:
        return None
    if not line.startswith("data: "):
        return None
    payload = line[6:]  # strip "data: " prefix
    if payload == "[DONE]":
        return None
    return json.loads(payload)


# ---------------------------------------------------------------------------
# Core trial runner
# ---------------------------------------------------------------------------

def run_trial(
    base_url: str,
    model: str,
    prompt_id: str,
    category: str,
    content: str,
    max_tokens: int,
    trial_num: int,
    is_warmup: bool,
    thinking_budget: int | None = None,
) -> TrialResult:
    """Run one inference trial against a vLLM server and return metrics.

    Args:
        base_url:       Root URL of the vLLM server (e.g. ``"http://localhost:8000"``).
        model:          Model name/path as registered in vLLM.
        prompt_id:      Identifier for the prompt (for logging/grouping).
        category:       Prompt category label.
        content:        The user message text to send.
        max_tokens:     Maximum tokens to generate.
        trial_num:      Trial index (0-based warmup, 1-based measured).
        is_warmup:      If True this trial is not counted in statistics.
        thinking_budget: Optional cap on thinking tokens (Qwen3.5).  When set,
                        tells the model to limit its ``<think>`` trace to at most
                        this many tokens, reserving the remainder of *max_tokens*
                        for the visible answer.

    Returns:
        TrialResult with timing, completion text, and memory usage.
    """
    url = f"{base_url}/v1/chat/completions"
    chat_kwargs: dict = {"enable_thinking": True}
    if thinking_budget is not None:
        chat_kwargs["thinking_budget"] = thinking_budget
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": chat_kwargs,
    }

    completion_parts: list[str] = []
    reasoning_parts: list[str] = []
    ttft_ms: float = 0.0
    tokens_generated: int = 0

    start = time.perf_counter()

    response = requests.post(url, json=payload, stream=True, timeout=300)
    response.raise_for_status()

    for raw_line in response.iter_lines(decode_unicode=True):
        chunk = parse_sse_line(raw_line)
        if chunk is None:
            continue

        # Extract content and reasoning from the delta
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})

            # Reasoning content (thinking trace) — separate field
            reasoning_text = delta.get("reasoning_content", "")
            if reasoning_text:
                reasoning_parts.append(reasoning_text)

            # Answer content
            token_text = delta.get("content", "")
            if token_text:
                if not completion_parts:
                    # First non-empty content — record TTFT
                    ttft_ms = (time.perf_counter() - start) * 1000
                completion_parts.append(token_text)

        # Usage appears on the final chunk
        usage = chunk.get("usage")
        if usage:
            tokens_generated = usage.get("completion_tokens", 0)

    total_time = time.perf_counter() - start

    # Fallback token count if server did not include usage
    if tokens_generated == 0:
        tokens_generated = len(completion_parts)

    tok_per_s = tokens_generated / total_time if total_time > 0 else 0.0
    raw_completion = "".join(completion_parts)
    reasoning = "".join(reasoning_parts)

    # Post-process: split thinking trace from answer.
    # Path 1: vLLM native reasoning_content (populated via SSE delta) — already separated.
    # Path 2: Model emits free-form scratchpad ending with </think> in the
    #          completion stream (learned behavior from training data).
    #          Split on </think> and route scratchpad → reasoning, answer → completion.
    if not reasoning and "</think>" in raw_completion:
        parts = raw_completion.split("</think>", 1)
        reasoning = parts[0].strip()
        completion = parts[1].strip()
    else:
        completion = raw_completion

    mem = capture_memory()

    return TrialResult(
        prompt_id=prompt_id,
        category=category,
        trial=trial_num,
        is_warmup=is_warmup,
        prompt=content,
        completion=completion,
        reasoning=reasoning,
        max_tokens=max_tokens,
        tokens_generated=tokens_generated,
        ttft_ms=ttft_ms,
        total_time_s=total_time,
        tok_per_s=tok_per_s,
        memory_after_gb=mem.used_gb,
    )
