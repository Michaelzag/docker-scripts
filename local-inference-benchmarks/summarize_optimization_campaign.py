#!/usr/bin/env python3
"""Summarize every saved local-inference optimization artifact.

The script reads, but never mutates, benchmark JSON. It emits one compact JSON
index and a Markdown evidence table suitable for the final ubuntu-admin record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rounded(value: Any, digits: int = 2) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def review_rows(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("reviews") or []
    cold = rows[0] if rows else {}
    warm_long = next(
        (
            row
            for row in rows[1:]
            if int((row.get("usage") or {}).get("completion_tokens") or 0) >= 1000
        ),
        {},
    )
    cold_timing = cold.get("timings") or {}
    warm_timing = warm_long.get("timings") or {}
    return {
        "kind": "review",
        "cold_wall_s": rounded(cold.get("wall_seconds")),
        "cold_pass_60s": cold.get("within_usability_cutoff"),
        "cold_prompt_tokens": (cold.get("usage") or {}).get("prompt_tokens"),
        "cold_prompt_tps": rounded(cold_timing.get("prompt_per_second")),
        "cold_decode_tps": rounded(cold_timing.get("predicted_per_second")),
        "cold_cache_tokens": cold.get("cache_tokens"),
        "warm_long_wall_s": rounded(warm_long.get("wall_seconds")),
        "warm_long_output_tokens": (warm_long.get("usage") or {}).get("completion_tokens"),
        "warm_long_decode_tps": rounded(warm_timing.get("predicted_per_second")),
        "warm_long_cache_tokens": warm_long.get("cache_tokens"),
        "all_requests_pass_60s": all(
            bool(row.get("within_usability_cutoff")) for row in rows
        )
        if rows
        else None,
    }


def tool_rows(data: dict[str, Any]) -> dict[str, Any]:
    loop = data.get("tool_loop") or {}
    turns = loop.get("turns") or []
    first = turns[0] if turns else {}
    warm_walls = [
        float(row["wall_seconds"])
        for row in turns[1:]
        if row.get("wall_seconds") is not None
    ]
    final = loop.get("final") or {}
    return {
        "kind": "tool_loop",
        "tool_valid": loop.get("valid_tool_calls"),
        "tool_requested": loop.get("requested_turns"),
        "all_turns_pass_60s": loop.get("all_turns_within_cutoff"),
        "first_turn_wall_s": rounded(first.get("wall_seconds")),
        "first_turn_prompt_tps": rounded((first.get("timings") or {}).get("prompt_per_second")),
        "warm_turn_max_s": rounded(max(warm_walls)) if warm_walls else None,
        "final_wall_s": rounded(final.get("wall_seconds")),
        "final_cache_tokens": final.get("cache_tokens"),
    }


def quality_rows(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "quality",
        "quality_passed": data.get("passed"),
        "quality_total": data.get("total"),
        "all_requests_pass_60s": data.get("all_within_usability_cutoff"),
        "categories": data.get("categories"),
    }


def concurrency_rows(data: dict[str, Any]) -> dict[str, Any]:
    run = data.get("concurrency") or {}
    return {
        "kind": "concurrency",
        "workers": run.get("workers"),
        "aggregate_decode_tps": rounded(run.get("aggregate_completion_tokens_per_second")),
        "all_requests_pass_60s": run.get("all_within_cutoff"),
        "wall_s": rounded(run.get("wall_seconds")),
    }


def context_rows(data: dict[str, Any]) -> dict[str, Any]:
    run = data.get("context_probe") or {}
    return {
        "kind": "context",
        "raw_prompt_tokens": run.get("raw_prompt_tokens"),
        "wall_s": rounded(run.get("wall_seconds")),
        "pass_60s": run.get("within_usability_cutoff"),
        "all_needles_recalled": run.get("all_needles_recalled"),
        "needle_recall": run.get("needle_recall"),
    }


def classify(data: dict[str, Any]) -> dict[str, Any]:
    if "cases" in data and "passed" in data:
        return quality_rows(data)
    if data.get("reviews"):
        return review_rows(data)
    if data.get("tool_loop"):
        return tool_rows(data)
    if data.get("concurrency"):
        return concurrency_rows(data)
    if data.get("context_probe"):
        return context_rows(data)
    return {"kind": "unknown"}


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Local inference optimization evidence",
        "",
        "Generated from saved JSON artifacts. A dash means the metric does not apply to that workload.",
        "",
        "| Model | Candidate | Workload | Cold/first wall | Prompt TPS | Decode TPS | Tool calls | Quality | 60s | Artifact |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        sixty = row.get("all_requests_pass_60s")
        if sixty is None:
            sixty = row.get("all_turns_pass_60s")
        if sixty is None:
            sixty = row.get("pass_60s")
        quality = "-"
        if row.get("quality_total") is not None:
            quality = f"{row.get('quality_passed')}/{row.get('quality_total')}"
        tools = "-"
        if row.get("tool_requested") is not None:
            tools = f"{row.get('tool_valid')}/{row.get('tool_requested')}"
        lines.append(
            "| {model} | {candidate} | {kind} | {wall} | {prompt} | {decode} | {tools} | {quality} | {sixty} | `{artifact}` |".format(
                model=row.get("model", "-"),
                candidate=row.get("candidate", "-"),
                kind=row.get("kind", "-"),
                wall=row.get("cold_wall_s", row.get("first_turn_wall_s", row.get("wall_s", "-"))),
                prompt=row.get("cold_prompt_tps", row.get("first_turn_prompt_tps", "-")),
                decode=row.get("cold_decode_tps", row.get("aggregate_decode_tps", "-")),
                tools=tools,
                quality=quality,
                sixty=sixty if sixty is not None else "-",
                artifact=row["artifact"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.results.rglob("*.json")):
        if path.name.startswith("harness-") or path.resolve() == args.json_output.resolve():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        relative = path.relative_to(args.results)
        parts = relative.parts
        row = classify(data)
        row.update(
            {
                "model": parts[0] if parts else "unknown",
                "candidate": "/".join(parts[1:-1]) or "unclassified",
                "label": data.get("label"),
                "artifact": str(relative),
            }
        )
        rows.append(row)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps({"artifacts": rows}, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(markdown(rows))
    print(f"indexed {len(rows)} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
