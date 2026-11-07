"""Run the versioned, exploratory 30-case campaign task set (not an IQ test).

Seeded prompts and independent gold go in a manifest; every inference request
uses the existing ChatClient and opt-in full evidence. Model Python runs only
through code_sandbox, never through host exec/eval/import.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

from agent_workload_benchmark import (
    ApiResult,
    ChatClient,
    redact_persistence,
    request_body,
    set_request_body_extra,
    timestamp_pair,
)
from challenge_cases import CATEGORIES, VERSION, Case, build_cases, grade_answer
from code_sandbox import check_sandbox, docker_argv, grade_code


@dataclasses.dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    message: dict[str, object]
    failure: str | None


def completion(result: ApiResult) -> Completion:
    if not result.ok:
        return Completion("", None, {}, "request_error")
    choices = (result.response or {}).get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return Completion("", None, {}, "malformed_response")
    choice = choices[0]
    reason = choice.get("finish_reason")
    message = choice.get("message")
    if not isinstance(reason, str) or not isinstance(message, dict):
        return Completion("", None, {}, "malformed_response")
    text = message.get("content")
    if text is not None and not isinstance(text, str):
        return Completion("", reason, message, "malformed_content")
    failure = None if reason in ("stop", "tool_calls") else f"finish_{reason}"
    return Completion(text or "", reason, message, failure)


@dataclasses.dataclass
class Depot:
    stock: int
    reserve: int
    capacity: int
    locked: bool


class DepotTask:
    """Own the mutable synthetic world, not a prescribed tool sequence."""

    def __init__(self, fixture: str):
        values = json.loads(fixture)
        self.target = values["target"]
        self.demand = values["demand"]
        self.depots = {name: Depot(**data) for name, data in values["depots"].items()}
        self.errors = 0
        self.operations = 0
        self.events: list[dict[str, object]] = []

    def snapshot(self) -> dict[str, object]:
        return {name: dataclasses.asdict(depot) for name, depot in self.depots.items()}

    def apply(self, name: str, arguments: object) -> dict[str, object]:
        self.operations += 1
        before = self.snapshot()
        error = None
        answer: dict[str, object] = {}
        if self.operations > 20:
            error = "operation limit exceeded"
        elif not isinstance(arguments, dict):
            error = "arguments must be an object"
        elif name == "list_depots" and not arguments:
            answer = {"depots": sorted(self.depots)}
        elif name == "inspect_depot" and set(arguments) == {"name"}:
            depot = self.depots.get(arguments["name"]) if isinstance(arguments["name"], str) else None
            if depot is None:
                error = "unknown depot"
            else:
                answer = dataclasses.asdict(depot)
        elif name == "transfer" and set(arguments) == {"source", "destination", "count"}:
            source, destination, count = arguments["source"], arguments["destination"], arguments["count"]
            if not isinstance(source, str) or not isinstance(destination, str) or type(count) is not int:
                error = "source/destination must be names and count a positive integer"
            elif source == destination or source not in self.depots or destination not in self.depots or count <= 0:
                error = "invalid transfer"
            else:
                src, dst = self.depots[source], self.depots[destination]
                if src.locked or dst.locked or src.stock - count < src.reserve or dst.stock + count > dst.capacity:
                    error = "transfer violates lock, reserve, or capacity"
                else:
                    src.stock -= count
                    dst.stock += count
                    answer = {"transferred": count, "source_stock": src.stock, "destination_stock": dst.stock}
        else:
            error = "unknown tool or invalid argument keys"
        if error:
            self.errors += 1
            answer = {"error": error}
        self.events.append({"tool": name, "arguments": arguments, "before": before, "result": answer, "after": self.snapshot()})
        return answer

    def goal_met(self) -> bool:
        return self.errors == 0 and self.depots[self.target].stock == self.demand


def tool_schema() -> list[dict[str, object]]:
    def tool(name: str, description: str, properties: dict[str, object]) -> dict[str, object]:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}}
    return [
        tool("list_depots", "List existing depot names.", {}),
        tool("inspect_depot", "Read widget stock, reserve minimum, capacity maximum, and lock status for one depot.", {"name": {"type": "string"}}),
        tool("transfer", "Atomically move widgets; rejects locked endpoints, reserve violation, capacity overflow, same source/destination or nonpositive count. No side effect on rejection.", {"source": {"type": "string"}, "destination": {"type": "string"}, "count": {"type": "integer", "minimum": 1}}),
    ]


def call_record(result: ApiResult, answer: Completion) -> dict[str, object]:
    return {
        "ok": result.ok, "wall_seconds": result.wall_seconds,
        "http_status": result.status, "error": result.error,
        "failure": answer.failure, "finish_reason": answer.finish_reason,
        "usage": (result.response or {}).get("usage"),
        "artifact": result.artifact,
    }


def run_case(case: Case, client: ChatClient, args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    messages: list[dict[str, object]] = [{"role": "user", "content": case.prompt}]
    calls: list[dict[str, object]] = []
    text = ""
    failure = None
    passed = False
    sandbox = None
    inference_seconds = 0.0
    world = DepotTask(case.tool_fixture) if case.tool_fixture else None
    for turn in range(10 if world else 1):
        body = request_body(args.model, messages, args.max_tokens, 0, tool_schema() if world else None)
        result = client.chat(body, case_id=f"{case.name}/turn-{turn + 1}")
        answer = completion(result)
        inference_seconds += result.wall_seconds
        calls.append(call_record(result, answer))
        text = answer.text
        if answer.failure:
            failure = answer.failure
            break
        tool_calls = answer.message.get("tool_calls")
        if world and tool_calls:
            if not isinstance(tool_calls, list) or answer.finish_reason not in ("tool_calls", "stop"):
                failure = "malformed_tool_calls"
                break
            messages.append(answer.message)
            for call in tool_calls:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not isinstance(call.get("function"), dict):
                    failure = "malformed_tool_call"
                    break
                function = call["function"]
                name, raw_args = function.get("name"), function.get("arguments")
                if not isinstance(name, str) or not isinstance(raw_args, str):
                    failure = "malformed_tool_call"
                    break
                try:
                    arguments = json.loads(raw_args)
                except ValueError:
                    failure = "invalid_tool_arguments_json"
                    break
                response = world.apply(name, arguments)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(response)})
                if world.operations > 20:
                    failure = "tool_operation_limit"
                    break
            if failure:
                break
            continue
        if answer.finish_reason != "stop" or tool_calls:
            failure = "unexpected_finish_or_tool_call"
            break
        if case.category == "code":
            source = text.strip()
            sandbox = grade_code(
                source, case.checks, command=docker_argv(args.docker_command),
                image=args.sandbox_image, timeout=args.sandbox_timeout,
                require_unchanged=case.name in ("merge_intervals", "latest_revision"),
            )
            passed = sandbox.passed
            if not passed:
                failure = sandbox.state
        else:
            passed = grade_answer(text, case.expected) and (world is None or world.goal_met())
            if not passed:
                failure = "wrong_answer_or_goal"
        break
    else:
        failure = "tool_turn_limit"
    return {
        "name": case.name, "category": case.category, "passed": passed,
        "failure": failure, "answer": text, "calls": calls,
        "inference_seconds": inference_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "sandbox": dataclasses.asdict(sandbox) if sandbox else None,
        "tool_events": world.events if world else None,
        "tool_final_state": world.snapshot() if world else None,
        "prompt_chars": len(case.prompt),
        "prefix_cache_state": "uncontrolled; no forced cache flush",
    }


class CategorySummary(TypedDict):
    passed: int
    total: int
    failed: int
    request_errors: int
    timeout_requests: int
    length_finishes: int
    inference_seconds_total: float
    inference_seconds_median: float
    inference_seconds_p95_nearest_rank: float
    elapsed_seconds_total: float
    usage_reported_sum: dict[str, int]
    requests_with_usage: int
    over_60s_case_count: int


def summarize(rows: list[dict[str, object]]) -> dict[str, CategorySummary]:
    """Every selected case is in the denominator, including failed requests."""
    categories: dict[str, CategorySummary] = {}
    for category in CATEGORIES:
        selected = [row for row in rows if row["category"] == category]
        if not selected:
            continue
        latencies = sorted(float(str(row["inference_seconds"])) for row in selected)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage_reports = 0
        error_requests = 0
        length_requests = 0
        timeout_requests = 0
        for row in selected:
            calls = row["calls"]
            if not isinstance(calls, list):
                raise TypeError("case calls must be a list")
            for call in calls:
                error_requests += int(not call["ok"])
                length_requests += int(call["finish_reason"] == "length")
                timeout_requests += int(
                    not call["ok"] and any(marker in str(call["error"]).casefold() for marker in ("timeout", "timed out"))
                )
                if isinstance(call["usage"], dict):
                    usage_reports += 1
                    for key in usage:
                        value = call["usage"].get(key)
                        if type(value) is int:
                            usage[key] += value
        categories[category] = {
            "passed": sum(row["passed"] is True for row in selected),
            "total": len(selected), "failed": sum(row["passed"] is not True for row in selected),
            "request_errors": error_requests, "length_finishes": length_requests,
            "timeout_requests": timeout_requests,
            "inference_seconds_total": sum(latencies),
            "elapsed_seconds_total": sum(float(str(row["elapsed_seconds"])) for row in selected),
            "inference_seconds_median": statistics.median(latencies),
            "inference_seconds_p95_nearest_rank": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "usage_reported_sum": usage, "requests_with_usage": usage_reports,
            "over_60s_case_count": sum(value > 60 for value in latencies),
        }
    return categories


def reserve_output(path: Path) -> None:
    """Exclusively reserve every final destination before any inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    destinations = sorted({path, path.with_suffix(".manifest.json"), path.with_suffix(".run-owner.json")})
    claimed: list[Path] = []
    reservation = {"pid": os.getpid(), "claimed": timestamp_pair(), "state": "reserved"}
    try:
        for destination in destinations:
            with destination.open("x") as claim:
                claimed.append(destination)
                json.dump(reservation, claim)
                claim.write("\n")
    except OSError:
        # Only paths exclusively created by this attempt can be rolled back.
        # Existing evidence, including another run's report/manifest, is intact.
        for destination in reversed(claimed):
            destination.unlink()
        raise


def write_json(path: Path, value: object, *, api_key: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as output:
        json.dump(redact_persistence(value, api_key), output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    parser.add_argument("--output", type=Path, required=True, help="New run report path; do not reuse a run directory")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Explicit runtime/model/hash/host/GPU provenance JSON; never inferred")
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--nonce", default="", help="Run identifier, retained but not injected into identical seeded prompts")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--cases", default="", help="Comma-separated exact case names")
    parser.add_argument("--categories", default="", help=",".join(CATEGORIES))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--measurement-timeout", type=float, default=600)
    parser.add_argument(
        "--extra-body-json", default="{}",
        help="Native sampling/profile JSON: temperature, top_p, top_k, min_p, presence_penalty, frequency_penalty, repeat_penalty, repetition_penalty, seed, chat_template_kwargs. No request-shape or budget overrides.",
    )
    parser.add_argument("--sandbox-image", default="", help="Parent-supplied Python image@sha256 pin; required for code cases")
    parser.add_argument("--docker-command", default="docker", help="Only docker or 'sudo -n docker'")
    parser.add_argument("--sandbox-timeout", type=float, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
            raise ValueError("output or scoring manifest already exists; choose a fresh run path")
        if args.max_tokens <= 0 or args.measurement_timeout <= 0 or not 0 < args.sandbox_timeout <= 10:
            raise ValueError("positive inference budgets and sandbox timeout <=10s required")
        command = docker_argv(args.docker_command)
        extra = json.loads(args.extra_body_json)
        supported_profile_keys = {
            "temperature", "top_p", "top_k", "min_p", "presence_penalty",
            "frequency_penalty", "repeat_penalty", "repetition_penalty",
            "seed", "chat_template_kwargs",
        }
        if not isinstance(extra, dict):
            raise ValueError("extra body must be a native sampling/profile JSON object")
        unsupported = set(extra) - supported_profile_keys
        if unsupported:
            raise ValueError(f"unsupported native profile keys (request-shape/budget overrides are forbidden): {', '.join(sorted(unsupported))}")
        set_request_body_extra(extra)
        all_cases = build_cases(args.seed)
        names = {name.strip() for name in args.cases.split(",") if name.strip()}
        categories = {name.strip() for name in args.categories.split(",") if name.strip()}
        if names - {case.name for case in all_cases} or categories - set(CATEGORIES):
            raise ValueError("unknown case/category filter")
        cases = [case for case in all_cases if (not names or case.name in names) and (not categories or case.category in categories)]
        if not cases:
            raise ValueError("filters selected no cases")
        if any(case.category == "code" for case in cases):
            check_sandbox(command, args.sandbox_image)
        provenance = json.loads(args.manifest.read_text()) if args.manifest else None
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("manifest must be a JSON object")
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"error: {args.api_key_env} is unset", file=sys.stderr)
        return 2
    try:
        reserve_output(args.output)
    except OSError as exc:
        print(f"error: cannot claim output namespace: {exc}", file=sys.stderr)
        return 2
    started_at = timestamp_pair()
    nonce = args.nonce or args.output.stem
    client = ChatClient(args.base_url, api_key, args.measurement_timeout, artifacts_dir=args.artifacts_dir, label=args.label, nonce=nonce)
    scoring_manifest = {
        "suite_version": VERSION, "seed": args.seed,
        "fixtures": [dataclasses.asdict(case) for case in cases],
        "max_tokens": args.max_tokens, "measurement_timeout_seconds": args.measurement_timeout,
        "temperature": extra.get("temperature", 0), "extra_body": extra,
        "sandbox_image": args.sandbox_image or None, "docker_command": list(command),
        "sandbox_timeout_seconds": args.sandbox_timeout,
        "provenance": provenance, "provenance_status": "operator supplied, not independently verified" if provenance else "not supplied; runtime identities unknown",
        "limitations": "Exploratory task set; not standardized IQ/general intelligence. No LLM judge. Nonstreaming EOF, no TTFT. Seeds are repetitions, not independent benchmark validation. No cache flush; usage tokenization is model-specific. Failed cases stay in denominator; missing usage is not zero-token inference.",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    write_json(manifest_path, scoring_manifest, api_key=api_key)
    fixture_hash = hashlib.sha256(json.dumps(scoring_manifest["fixtures"], sort_keys=True).encode()).hexdigest()
    rows = []
    for case in cases:
        rows.append(run_case(case, client, args))
        report = {
            "suite_version": VERSION, "seed": args.seed, "label": args.label,
            "nonce": nonce, "model": args.model, "started": started_at, "updated": timestamp_pair(),
            "scoring_manifest": str(manifest_path), "fixtures_sha256": fixture_hash,
            "completed": len(rows), "selected_total": len(cases), "complete": len(rows) == len(cases),
            "categories": summarize(rows), "cases": rows,
        }
        write_json(args.output, report, api_key=api_key)
    print(json.dumps({"categories": summarize(rows), "report": str(args.output)}, indent=2))
    # Quality failure is evidence, not a transport/process failure. Parent uses counts.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
