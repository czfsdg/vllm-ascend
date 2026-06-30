#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze D-Cut verifier logs by input shape.

This script parses D-Cut runtime logs and summarizes whether smaller verifier
shapes (``total_tokens`` / Q) actually reduce latency. It is intentionally
standalone and dependency-free so it can run on a server log file without
importing vLLM, torch, or the D-Cut plugin.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_VERIFIER_RE = re.compile(
    r"D-Cut verifier timing: elapsed_ms=(?P<elapsed>[0-9.]+) "
    r"total_tokens=(?P<total_tokens>\d+) .*?num_reqs=(?P<num_reqs>\d+) "
    r"query_lens_sum=(?P<query_lens_sum>\d+) query_lens_max=(?P<query_lens_max>\d+)"
)
_BREAKDOWN_RE = re.compile(
    r"D-Cut verifier breakdown: .*?total_tokens=(?P<total_tokens>\d+) "
    r".*?num_reqs=(?P<num_reqs>\d+) phases=(?P<phases>\{.*?\}) "
    r"model_forward_shape_stats=(?P<model_forward_stats>\{.*?\})"
)
_ATTENTION_RE = re.compile(
    r"D-Cut attention timing: elapsed_ms=(?P<elapsed>[0-9.]+) .*?"
    r"total_tokens=(?P<total_tokens>\d+) .*?num_reqs=(?P<num_reqs>\d+)"
)


@dataclass
class ShapeStats:
    num_reqs: int
    total_tokens: int
    query_lens_max: int = 0
    verifier_ms: list[float] = field(default_factory=list)
    model_forward_ms: list[float] = field(default_factory=list)
    model_call_ms: list[float] = field(default_factory=list)
    attention_ms: list[float] = field(default_factory=list)

    @property
    def key(self) -> tuple[int, int]:
        return self.num_reqs, self.total_tokens


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _parse_float_dict(raw_value: str) -> dict[str, float]:
    parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, dict):
        return {}
    return {str(key): float(value) for key, value in parsed.items() if isinstance(value, (int, float))}


def parse_log_lines(lines: Iterable[str]) -> dict[tuple[int, int], ShapeStats]:
    shapes: dict[tuple[int, int], ShapeStats] = {}

    def get_shape(num_reqs: int, total_tokens: int) -> ShapeStats:
        key = (num_reqs, total_tokens)
        if key not in shapes:
            shapes[key] = ShapeStats(num_reqs=num_reqs, total_tokens=total_tokens)
        return shapes[key]

    for line in lines:
        if match := _VERIFIER_RE.search(line):
            num_reqs = int(match.group("num_reqs"))
            total_tokens = int(match.group("total_tokens"))
            shape = get_shape(num_reqs, total_tokens)
            shape.query_lens_max = max(shape.query_lens_max, int(match.group("query_lens_max")))
            shape.verifier_ms.append(float(match.group("elapsed")))
            continue
        if match := _BREAKDOWN_RE.search(line):
            num_reqs = int(match.group("num_reqs"))
            total_tokens = int(match.group("total_tokens"))
            shape = get_shape(num_reqs, total_tokens)
            phases = _parse_float_dict(match.group("phases"))
            model_forward_stats = _parse_float_dict(match.group("model_forward_stats"))
            if "model_forward" in phases:
                shape.model_forward_ms.append(phases["model_forward"])
            if "model_forward.model_call" in phases:
                shape.model_call_ms.append(phases["model_forward.model_call"])
            elif "elapsed_ms" in model_forward_stats:
                shape.model_call_ms.append(model_forward_stats["elapsed_ms"])
            continue
        if match := _ATTENTION_RE.search(line):
            num_reqs = int(match.group("num_reqs"))
            total_tokens = int(match.group("total_tokens"))
            shape = get_shape(num_reqs, total_tokens)
            shape.attention_ms.append(float(match.group("elapsed")))
    return shapes


def _linear_fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _print_table(shapes: list[ShapeStats]) -> None:
    print(
        "| num_reqs | Q/total_tokens | query_lens_max | samples | verifier_ms | "
        "model_forward_ms | model_call_ms | attention_ms |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for shape in shapes:
        print(
            "| "
            f"{shape.num_reqs} | {shape.total_tokens} | {shape.query_lens_max} | {len(shape.verifier_ms)} | "
            f"{_format_ms(_mean(shape.verifier_ms))} | "
            f"{_format_ms(_mean(shape.model_forward_ms))} | "
            f"{_format_ms(_mean(shape.model_call_ms))} | "
            f"{_format_ms(_mean(shape.attention_ms))} |"
        )


def _print_fit(shapes: list[ShapeStats], field_name: str, values: list[tuple[float, float]]) -> None:
    fit = _linear_fit(values)
    if fit is None:
        print(f"\n{field_name}: not enough distinct shapes for a linear fit.")
        return
    slope, intercept = fit
    min_q = min(q for q, _ in values)
    max_q = max(q for q, _ in values)
    predicted_min = intercept + slope * min_q
    predicted_max = intercept + slope * max_q
    predicted_saving = predicted_max - predicted_min
    fixed_ratio = math.inf if predicted_max == 0.0 else intercept / predicted_max
    print(f"\n{field_name} linear fit: ms ~= {intercept:.3f} + {slope:.6f} * Q")
    print(f"{field_name} predicted saving from Q={max_q:.0f} to Q={min_q:.0f}: {predicted_saving:.3f} ms")
    print(f"{field_name} fixed-cost ratio at Q={max_q:.0f}: {fixed_ratio:.2%}")


def analyze_log(path: Path) -> int:
    shapes = sorted(
        parse_log_lines(path.read_text(encoding="utf-8", errors="replace").splitlines()).values(),
        key=lambda item: (item.total_tokens, item.num_reqs),
    )
    if not shapes:
        print("No D-Cut verifier timing records found.")
        return 1
    _print_table(shapes)
    fit_specs = [
        ("verifier", [(shape.total_tokens, _mean(shape.verifier_ms)) for shape in shapes]),
        ("model_forward", [(shape.total_tokens, _mean(shape.model_forward_ms)) for shape in shapes]),
        ("model_call", [(shape.total_tokens, _mean(shape.model_call_ms)) for shape in shapes]),
        ("attention", [(shape.total_tokens, _mean(shape.attention_ms)) for shape in shapes]),
    ]
    for field_name, raw_points in fit_specs:
        points = [(float(q), float(value)) for q, value in raw_points if value is not None]
        _print_fit(shapes, field_name, points)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze D-Cut verifier latency by Q/shape from a log file.")
    parser.add_argument("log_file", type=Path, help="Path to a vLLM/vLLM Ascend log containing D-Cut timing lines.")
    args = parser.parse_args()
    return analyze_log(args.log_file)


if __name__ == "__main__":
    raise SystemExit(main())
