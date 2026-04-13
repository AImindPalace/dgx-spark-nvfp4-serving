from __future__ import annotations

import math
import statistics as _stats
from dataclasses import asdict, dataclass


@dataclass
class Stats:
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_stats(values: list[float]) -> Stats:
    """Aggregate stats with linear-interpolation percentiles.

    Raises ValueError if values is empty.
    All fields are rounded to 2 decimal places.
    """
    if not values:
        raise ValueError("empty: cannot compute stats on an empty list")

    if len(values) == 1:
        v = round(values[0], 2)
        return Stats(mean=v, std=0.0, min=v, max=v, p50=v, p95=v)

    sorted_vals = sorted(values)
    return Stats(
        mean=round(_stats.mean(sorted_vals), 2),
        std=round(_stats.stdev(sorted_vals), 2),
        min=round(sorted_vals[0], 2),
        max=round(sorted_vals[-1], 2),
        p50=round(_percentile(sorted_vals, 0.50), 2),
        p95=round(_percentile(sorted_vals, 0.95), 2),
    )


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear interpolation percentile.

    idx = pct * (n - 1), then lerp between floor and ceil indices.
    """
    n = len(sorted_vals)
    idx = pct * (n - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])
