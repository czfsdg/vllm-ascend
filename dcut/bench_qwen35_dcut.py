#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ALLaVA-style client benchmark for Qwen3.5 DFlash / D-Cut serving."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_MARKERS = ("<image>", "<|image|>", "<image>", "")


@dataclass(frozen=True)
class Sample:
    sample_id: int
    prompt: str
    source_id: str | None = None


def _clean_prompt(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    for marker in IMAGE_MARKERS:
        text = text.replace(marker, " ")
    return " ".join(text.split()).strip()


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return _clean_prompt(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type in ("text", "input_text"):
                    parts.append(part.get("text") or "")
                elif "text" in part and part_type not in ("image", "image_url"):
                    parts.append(part.get("text") or "")
        return _clean_prompt(" ".join(parts))
    return _clean_prompt(content)


def _first_user_prompt(record: dict[str, Any]) -> str:
    for key in ("prompt", "text", "question", "instruction", "query"):
        if key in record:
            prompt = _clean_prompt(record[key])
            if prompt:
                return prompt
    conv = record.get("conversations") or record.get("messages") or []
    if isinstance(conv, dict):
        conv = conv.get("conversations") or conv.get("messages") or []
    if not isinstance(conv, list):
        return ""
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("from") or turn.get("speaker") or "").lower()
        if role not in ("user", "human", "prompter"):
            continue
        prompt = (_text_from_content(turn["content"]) if "content" in turn
                  else _clean_prompt(turn.get("value") or turn.get("text") or ""))
        if prompt:
            return prompt
    return ""


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        return []
    if path.suffix.lower() != ".jsonl" and stripped[0] in "[{":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            if isinstance(obj, dict):
                for key in ("data", "items", "samples", "conversations"):
                    if isinstance(obj.get(key), list):
                        obj = obj[key]
                        break
                else:
                    obj = [obj]
            if not isinstance(obj, list):
                raise ValueError(f"{path} did not contain a JSON list/dict dataset")
            return [x for x in obj if isinstance(x, dict)]
    rows = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSONL") from exc
        if isinstance(item, dict):
            rows.append(item)
    return rows


def load_samples(path: str, *, num: int, skip: int, tail_frac: float,
                 shuffle_seed: int | None, min_chars: int,
                 max_chars: int) -> list[Sample]:
    records = _read_json_or_jsonl(Path(path))
    if tail_frac > 0:
        if not 0 < tail_frac <= 1:
            raise ValueError("--tail-frac must be in (0, 1]")
        records = records[int(len(records) * (1.0 - tail_frac)):]
    if skip:
        records = records[skip:]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(records)
    samples = []
    for idx, record in enumerate(records):
        prompt = _first_user_prompt(record)
        if len(prompt) < min_chars:
            continue
        if max_chars > 0 and len(prompt) > max_chars:
            prompt = prompt[:max_chars].rstrip()
        source_id = record.get("id") or record.get("uid") or record.get("image") or record.get("image_path")
        samples.append(Sample(idx, prompt, str(source_id)))
        if num and len(samples) >= num:
            break
    if not samples:
        raise ValueError(f"No usable text prompts found in {path}")
    return samples


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    xs = sorted(xs)
    pos = (len(xs) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _post_stream_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.perf_counter()
    ttft = None
    text_parts = []
    usage = None
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                piece = ((choice.get("delta") or {}).get("content")
                         if "delta" in choice else choice.get("text"))
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(piece)
                    chunks += 1
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    total = time.perf_counter() - t0
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens") or chunks
    return {
        "ok": True,
        "ttft_s": ttft,
        "latency_s": total,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_chars": sum(len(x) for x in text_parts),
        "output_preview": "".join(text_parts)[:240],
    }


def run_one(sample: Sample, *, base_url: str, model: str, endpoint: str,
            max_tokens: int, temperature: float, timeout: float,
            enable_thinking: bool) -> dict[str, Any]:
    if endpoint == "chat":
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": sample.prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    elif endpoint == "completions":
        url = base_url.rstrip("/") + "/completions"
        payload = {
            "model": model,
            "prompt": sample.prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        raise ValueError(f"unknown endpoint: {endpoint}")
    started = time.time()
    try:
        result = _post_stream_json(url, payload, timeout)
    except Exception as exc:  # noqa: BLE001 - keep benchmark running
        result = {"ok": False, "error": str(exc)}
    result.update({
        "sample_id": sample.sample_id,
        "source_id": sample.source_id,
        "prompt_chars": len(sample.prompt),
        "prompt_preview": sample.prompt[:240],
        "started_unix": started,
    })
    return result


def summarize(records: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    ok = [r for r in records if r.get("ok")]
    bad = [r for r in records if not r.get("ok")]
    lat = [float(r["latency_s"]) for r in ok if r.get("latency_s") is not None]
    ttft = [float(r["ttft_s"]) for r in ok if r.get("ttft_s") is not None]
    out_tok = [int(r["completion_tokens"]) for r in ok if isinstance(r.get("completion_tokens"), int)]
    in_tok = [int(r["prompt_tokens"]) for r in ok if isinstance(r.get("prompt_tokens"), int)]
    total_out = sum(out_tok)
    total_in = sum(in_tok)
    summary = {
        "requests": len(records),
        "succeeded": len(ok),
        "failed": len(bad),
        "wall_s": wall_s,
        "request_per_s": len(ok) / wall_s if wall_s > 0 else None,
        "output_tok_per_s": total_out / wall_s if wall_s > 0 and total_out else None,
        "total_tok_per_s": (total_in + total_out) / wall_s if wall_s > 0 and (total_in or total_out) else None,
        "prompt_tokens": total_in or None,
        "completion_tokens": total_out or None,
        "latency_mean_s": statistics.mean(lat) if lat else None,
        "latency_p50_s": _percentile(lat, 0.50) if lat else None,
        "latency_p90_s": _percentile(lat, 0.90) if lat else None,
        "latency_p99_s": _percentile(lat, 0.99) if lat else None,
        "ttft_mean_s": statistics.mean(ttft) if ttft else None,
        "ttft_p50_s": _percentile(ttft, 0.50) if ttft else None,
        "ttft_p90_s": _percentile(ttft, 0.90) if ttft else None,
        "ttft_p99_s": _percentile(ttft, 0.99) if ttft else None,
    }
    if bad:
        summary["first_error"] = bad[0].get("error")
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    def fmt(key: str, digits: int = 3) -> str:
        val = summary.get(key)
        if val is None:
            return "NA"
        if isinstance(val, float):
            return f"{val:.{digits}f}"
        return str(val)

    print(
        "SUMMARY "
        f"ok={fmt('succeeded', 0)}/{fmt('requests', 0)} "
        f"wall={fmt('wall_s')}s "
        f"req/s={fmt('request_per_s')} "
        f"out_tok/s={fmt('output_tok_per_s', 1)} "
        f"total_tok/s={fmt('total_tok_per_s', 1)} "
        f"ttft_p50={fmt('ttft_p50_s')}s "
        f"ttft_p90={fmt('ttft_p90_s')}s "
        f"lat_p50={fmt('latency_p50_s')}s "
        f"lat_p90={fmt('latency_p90_s')}s")
    if summary.get("first_error"):
        print(f"FIRST_ERROR {summary['first_error']}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a running Qwen3.5 vLLM server with ALLaVA prompts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="qwen35")
    parser.add_argument("--endpoint", choices=("chat", "completions"), default="chat")
    parser.add_argument("--num", type=int, default=256)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--tail-frac", type=float, default=0.0)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--min-prompt-chars", type=int, default=1)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    args = parser.parse_args()

    samples = load_samples(
        args.dataset,
        num=args.num + args.warmup,
        skip=args.skip,
        tail_frac=args.tail_frac,
        shuffle_seed=args.shuffle_seed,
        min_chars=args.min_prompt_chars,
        max_chars=args.max_prompt_chars,
    )
    warmup = samples[:args.warmup]
    measured = samples[args.warmup:]
    if not measured:
        raise SystemExit("No measured samples left after --warmup")
    print(f"[{args.tag}] loaded prompts={len(samples)} warmup={len(warmup)} "
          f"measured={len(measured)} concurrency={args.concurrency} base_url={args.base_url}")
    common = {
        "base_url": args.base_url,
        "model": args.model,
        "endpoint": args.endpoint,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "enable_thinking": args.enable_thinking,
    }
    for sample in warmup:
        run_one(sample, **common)
    records = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(run_one, sample, **common) for sample in measured]
        with open(args.out, "w", encoding="utf-8") as fout:
            for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                rec = fut.result()
                rec["tag"] = args.tag
                records.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if done % max(1, min(16, args.concurrency)) == 0 or done == len(futs):
                    ok = sum(1 for r in records if r.get("ok"))
                    print(f"[{args.tag}] completed {done}/{len(futs)} ok={ok}")
    wall_s = time.perf_counter() - t0
    summary = summarize(records, wall_s)
    summary.update({
        "tag": args.tag,
        "dataset": os.path.abspath(args.dataset),
        "base_url": args.base_url,
        "model": args.model,
        "endpoint": args.endpoint,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "tail_frac": args.tail_frac,
    })
    if args.summary_out:
        Path(args.summary_out).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    _print_summary(summary)
    if summary["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
