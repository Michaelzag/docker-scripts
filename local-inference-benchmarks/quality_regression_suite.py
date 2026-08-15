#!/usr/bin/env python3
"""Deterministic 26-case regression suite for local inference tuning.

This is intentionally source-controlled: the earlier 26-case comparison saved
category totals but not the case definitions.  Runtime, quantization, KV, and
speculative-decoding experiments now have a reproducible quality gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

from agent_workload_benchmark import ChatClient, request_body, set_request_body_extra


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    category: str
    prompt: str
    expected: Any
    validator: str = "exact"


def fixed_cases() -> list[Case]:
    cases = [
        Case("arithmetic", "reasoning", "Compute (173 * 29) - ((84 / 7) * 13). Reply with only the integer.", "4861"),
        Case("syllogism", "reasoning", "All cranes are birds. No birds are mammals. Can any crane be a mammal? Reply only YES or NO.", "NO"),
        Case("set_intersection", "reasoning", "A={2,3,5,7,11}; B={1,3,6,7,12}; C={0,3,7,9}. Return A intersection B intersection C as ascending comma-separated integers only.", [3, 7], "csv_ints"),
        Case("shortest_path", "reasoning", "Weighted edges: A-B 4, A-C 2, C-B 1, B-D 5, C-D 8, D-E 2, B-E 9. What is the shortest A-to-E distance? Reply with only the integer.", "10"),
        Case("probability", "reasoning", "A fair six-sided die is rolled twice. What is the reduced fraction for exactly one roll being a 6? Reply only numerator/denominator.", "5/18"),
        Case("sequence", "reasoning", "Sequence: 2, 6, 12, 20, 30, ?. Reply with only the next integer.", "42"),
        Case("logic_order", "reasoning", "Mira finished before Sol. Taro finished after Sol but before Inez. Who finished second? Reply with only the name.", "Sol"),
        Case("base_conversion", "reasoning", "Convert hexadecimal 2F7 to decimal. Reply only with the integer.", "759"),
        Case("dict_mutation", "code_review", "Python: `for key in cache:\n    if key.startswith(prefix): del cache[key]`. Return only the primary defect label: MUTATION_DURING_ITERATION, DATA_RACE, or MEMORY_LEAK.", "MUTATION_DURING_ITERATION"),
        Case("sql_injection", "code_review", "Python: `cursor.execute(\"SELECT * FROM users WHERE name = '%s'\" % request.args['name'])`. Return only: SQL_INJECTION, XSS, or CSRF.", "SQL_INJECTION"),
        Case("lost_update", "code_review", "Two workers execute `value = read_counter(); write_counter(value + 1)` concurrently without a transaction. Return only: LOST_UPDATE, DEADLOCK, or BUFFER_OVERFLOW.", "LOST_UPDATE"),
        Case("path_traversal", "code_review", "A download handler opens `BASE / request.query['filename']` without resolving or checking containment. Return only: PATH_TRAVERSAL, SSRF, or OPEN_REDIRECT.", "PATH_TRAVERSAL"),
        Case("async_blocking", "code_review", "Inside `async def handle()`, code calls `subprocess.run(cmd)` for five minutes on the event-loop thread. Return only: ASYNC_BLOCKING, USE_AFTER_FREE, or TYPE_CONFUSION.", "ASYNC_BLOCKING"),
        Case("prefix_auth", "code_review", "Authorization accepts a resource when `resource_id.startswith(allowed_id)`, so allowed `tenant-1` also accepts `tenant-10`. Return only: PREFIX_AUTH_BYPASS, INTEGER_OVERFLOW, or CLICKJACKING.", "PREFIX_AUTH_BYPASS"),
        Case("judge_requirement", "judge", "Requirement: reject missing API keys with HTTP 401. Candidate A returns 401. Candidate B returns 500. Reply only A or B.", "A"),
        Case("judge_scope", "judge", "Instruction: change only README.md. Candidate A changes README.md and app.py. Candidate B changes only README.md. Reply only A or B.", "B"),
        Case("judge_evidence", "judge", "Claim must cite a measured latency. Candidate A says 'probably fast'. Candidate B says 'p95 was 42 ms over 10,000 requests'. Reply only A or B.", "B"),
        Case("judge_security", "judge", "Choose the safer implementation. A interpolates user input into SQL. B uses a parameterized query. Reply only A or B.", "B"),
        Case("judge_reversibility", "judge", "Maintenance must be reversible. A deletes the only data copy. B takes a snapshot and changes an alias with rollback retained. Reply only A or B.", "B"),
        Case("judge_correctness", "judge", "Need stable ascending numeric order for [10,2,1]. A sorts strings and returns [1,10,2]. B sorts integers and returns [1,2,10]. Reply only A or B.", "B"),
        Case("json_exact", "instruction", "Return exactly one compact JSON object with keys alpha then beta, values 7 and false. No markdown.", {"alpha": 7, "beta": False}, "json"),
        Case("csv_exact", "instruction", "Return these records as CSV with exactly this header order name,count and rows sorted by name: zed=2, amy=9. No code fence.", "name,count\namy,9\nzed,2"),
        Case("sorted_unique", "instruction", "Return the unique words from `pear apple pear kiwi apple` alphabetically, lowercase, separated by one comma and no spaces.", "apple,kiwi,pear"),
        Case("format_exact", "instruction", "Reply with exactly three lines: first `BEGIN`, second the number 17, third `END`. No other text.", "BEGIN\n17\nEND"),
    ]
    return cases


def normalize(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


def validate(actual: str, expected: Any, validator: str) -> bool:
    actual = normalize(actual)
    if validator == "exact":
        return actual.casefold() == str(expected).strip().casefold()
    if validator == "json":
        try:
            return json.loads(actual) == expected
        except json.JSONDecodeError:
            return False
    if validator == "contains":
        return str(expected) in actual
    if validator == "csv_ints":
        try:
            return [int(item.strip()) for item in actual.split(",")] == expected
        except ValueError:
            return False
    raise ValueError(f"unknown validator: {validator}")


def long_context_case(client: ChatClient, target_tokens: int, name: str) -> tuple[str, str, int | None]:
    needle = hashlib.sha256(f"{name}:{target_tokens}".encode()).hexdigest()[:24]
    block = (
        "module telemetry record: tenant isolation is enabled; revision checks are monotonic; "
        "workers release leases after bounded retries; audit sequence continues.\n"
    )
    low, high = 1, max(8, target_tokens // 20)
    best_text, best_count = block, client.tokenize(block)
    while low <= high:
        mid = (low + high) // 2
        filler = block * mid
        count = client.tokenize(filler)
        if count is None:
            mid = max(1, target_tokens // 25)
            filler = block * mid
            best_text, best_count = filler, None
            break
        if best_count is None or abs(count - target_tokens) < abs(best_count - target_tokens):
            best_text, best_count = filler, count
        if count < target_tokens:
            low = mid + 1
        else:
            high = mid - 1
    point = len(best_text) // 2 if name.endswith("middle") else len(best_text) // 20
    prompt = (
        best_text[:point]
        + f"\nUNIQUE_AUDIT_VALUE={needle}\n"
        + best_text[point:]
        + "\nReply with only the exact UNIQUE_AUDIT_VALUE from the text.\n"
    )
    return prompt, needle, best_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-timeout", type=float, default=900)
    parser.add_argument("--usability-cutoff", type=float, default=60)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Completion budget for non-reasoning cases",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=4096,
        help="Completion budget for reasoning cases that may emit hidden reasoning first",
    )
    parser.add_argument("--long-context-tokens", default="16384,120000")
    parser.add_argument(
        "--categories",
        default="",
        help="Optional comma-separated category filter for candidate pre-screening",
    )
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help="JSON object merged into every chat request (for example reasoning controls)",
    )
    parser.add_argument("--label", default="unspecified")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        extra_body = json.loads(args.extra_body_json)
    except json.JSONDecodeError as exc:
        print(f"error: invalid --extra-body-json: {exc}", file=sys.stderr)
        return 2
    if not isinstance(extra_body, dict):
        print("error: --extra-body-json must decode to an object", file=sys.stderr)
        return 2
    set_request_body_extra(extra_body)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"error: {args.api_key_env} is unset", file=sys.stderr)
        return 2
    client = ChatClient(args.base_url, api_key, args.measurement_timeout)
    cases = fixed_cases()
    selected_categories = {
        value.strip() for value in args.categories.split(",") if value.strip()
    }
    if selected_categories:
        cases = [case for case in cases if case.category in selected_categories]
    long_sizes = [int(value) for value in args.long_context_tokens.split(",") if value]
    if selected_categories and "long_context" not in selected_categories:
        long_sizes = []
    for index, size in enumerate(long_sizes):
        prompt, needle, count = long_context_case(
            client, size, "long_middle" if index else "long_early"
        )
        cases.append(
            Case(f"long_context_{size}", "long_context", prompt, needle, "contains")
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_max_tokens = (
            args.reasoning_max_tokens if case.category == "reasoning" else args.max_tokens
        )
        result = client.chat(
            request_body(
                args.model,
                [{"role": "user", "content": case.prompt}],
                case_max_tokens,
                0,
            )
        )
        actual = ""
        response: dict[str, Any] = result.response or {}
        if result.ok:
            message = ((response.get("choices") or [{}])[0].get("message")) or {}
            actual = message.get("content") or ""
        passed = result.ok and validate(actual, case.expected, case.validator)
        rows.append(
            {
                "name": case.name,
                "category": case.category,
                "passed": passed,
                "ok": result.ok,
                "wall_seconds": round(result.wall_seconds, 6),
                "within_usability_cutoff": result.ok
                and result.wall_seconds <= args.usability_cutoff,
                "max_tokens": case_max_tokens,
                "finish_reason": ((response.get("choices") or [{}])[0]).get("finish_reason"),
                "usage": response.get("usage"),
                "timings": response.get("timings"),
                "actual": normalize(actual)[:512],
                "expected": case.expected,
                "error": result.error,
            }
        )

    categories: dict[str, dict[str, int]] = {}
    for row in rows:
        category = categories.setdefault(row["category"], {"passed": 0, "total": 0})
        category["total"] += 1
        category["passed"] += int(row["passed"])
    report = {
        "schema_version": 1,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": args.label,
        "endpoint": args.base_url,
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "extra_body": extra_body,
        "usability_cutoff_seconds": args.usability_cutoff,
        "passed": sum(int(row["passed"]) for row in rows),
        "total": len(rows),
        "all_within_usability_cutoff": all(row["within_usability_cutoff"] for row in rows),
        "categories": categories,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "total", "all_within_usability_cutoff", "categories")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
