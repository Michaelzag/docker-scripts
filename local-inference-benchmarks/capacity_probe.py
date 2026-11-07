#!/usr/bin/env python3
"""Use case: one loaded endpoint/native profile and fixed or pooled token budgets
produce retained calibration, concurrent retrieval and observed scheduling evidence.
Invariants: formatted-token counts before inference, no retries/answer repair,
all selected requests in the denominator. Side effects: HTTP and evidence files;
only worker count, pool mode, token targets and native sampling actually vary.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import cast

from agent_workload_benchmark import (
    ApiResult,
    ChatClient,
    redact_persistence,
    redact_wire_body,
    request_body,
    set_request_body_extra,
    timestamp_pair,
)
from intelligence_challenge_suite import (
    call_record,
    completion,
    reserve_output,
    write_json,
)
from published_knowledge_benchmark import native_profile, unique_object

VERSION = "capacity-probe-v1"
MAX_TOKENS = 8192
CALIBRATION_MARGIN = 512
METADATA_LIMIT = 2 * 1024 * 1024
LIMITATIONS = (
    "Fixed means independent reserved attention windows; pooled means a shared "
    "mixed-load attention budget, NOT parallel reserved long windows. Declared "
    "budgets do not prove architectural fit, GPU placement or hardware capacity. "
    "Only actual input evidence, retrieval and sampled processing/deferred counters "
    "are scored. Missing evidence is UNKNOWN. Nonstream wall/EOF is not TTFT; "
    "wall minus backend prefill/decode is residual, NOT exact queue time. "
    "Group end-to-end throughput is not aggregate backend throughput. "
    "HTTP uses the frozen client's per-I/O timeout, not an absolute wall deadline; "
    "wall exceeding that timeout is diagnostic only, not a usability SLA. "
    "A bounded sampler can miss short scheduling peaks. No HTTP model retries."
)


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    mode: str
    parallel: int
    per_slot: int
    pool: int
    long_target: int = 120000
    short_target: int = 8000
    long_workers: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "pooled") or self.parallel not in (4, 6):
            raise ValueError("mode must be fixed/pooled and parallel must be 4/6")
        if self.long_workers not in (None, 1, 2):
            raise ValueError("explicit long workers must be 1 or 2")
        if self.mode == "fixed" and self.long_workers is not None:
            raise ValueError("long-workers is only applicable to the pooled mode")
        if min(self.per_slot, self.pool, self.long_target, self.short_target) <= 0:
            raise ValueError("token budgets must be positive")
        if min(self.long_target, self.short_target) <= CALIBRATION_MARGIN:
            raise ValueError("targets must exceed calibration margin")
        if max(self.long_target, self.short_target) + MAX_TOKENS > self.per_slot:
            raise ValueError("target input plus output ceiling exceeds per-slot budget")
        if self.short_target >= self.long_target:
            raise ValueError("short target must be smaller than long target")
        if self.mode == "fixed" and self.pool < self.parallel * self.per_slot:
            raise ValueError("fixed total context must reserve every independent slot")

    def kinds(self) -> tuple[str, ...]:
        longs = self.parallel if self.mode == "fixed" else (
            self.long_workers if self.long_workers is not None else (2 if self.parallel == 6 else 1)
        )
        return ("long",) * longs + ("short",) * (self.parallel - longs + 1)


@dataclasses.dataclass(frozen=True, slots=True)
class Prepared:
    case_id: str
    kind: str
    body: dict[str, object]
    expected: dict[str, str]
    input_tokens: int
    calibration: tuple[dict[str, object], ...]
    needle_positions: tuple[dict[str, object], ...]


def budget_proof(plan: Plan, inputs: list[int]) -> dict[str, object]:
    if len(inputs) != plan.parallel + 1 or any(type(n) is not int or n <= 0 for n in inputs):
        raise ValueError("one positive measured count required for every selected request")
    if any(n + MAX_TOKENS > plan.per_slot for n in inputs):
        raise ValueError("actual formatted input plus output exceeds per-slot budget")
    # The server, not submission order, chooses which short is queued. The
    # largest parallel counts cover every possible active subset, including it.
    active_inputs = sum(sorted(inputs, reverse=True)[:plan.parallel])
    active_output_ceiling = plan.parallel * MAX_TOKENS
    if plan.mode == "pooled" and active_inputs + active_output_ceiling > plan.pool:
        raise ValueError("actual worst-active inputs plus ALL output ceilings exceed shared pool")
    return {
        "mode": plan.mode, "parallel": plan.parallel, "request_count": len(inputs),
        "actual_input_tokens": inputs, "worst_active_input_tokens": active_inputs,
        "all_active_output_ceilings": active_output_ceiling,
        "worst_active_total": active_inputs + active_output_ceiling,
        "pool_tokens": plan.pool, "per_slot_tokens": plan.per_slot,
        "independent_reservations": plan.mode == "fixed",
        "fixed_reserved_total": plan.parallel * plan.per_slot if plan.mode == "fixed" else None,
        "budget_pass": True,
    }


def fixture(seed: int, nonce: str, units: int) -> tuple[str, dict[str, str], tuple[dict[str, object], ...]]:
    """Equal-shape synthetic archive records, not host data or executable code."""
    expected = {f"fact_{i}": hashlib.sha256(f"{seed}:{nonce}:{i}".encode()).hexdigest()[:16]
                for i in range(3)}
    offsets = [round(units * fraction) for fraction in (0.08, 0.51, 0.91)]
    parts = [f"Client nonce: {nonce}\nSynthetic public observatory archive.\n"]
    positions: list[dict[str, object]] = []
    for index in range(units + 1):
        for needle, offset in enumerate(offsets):
            if index == offset:
                key = f"fact_{needle}"
                positions.append({"key": key, "archive_fraction": index / max(1, units),
                                  "character_offset": sum(map(len, parts))})
                parts.append(f"\nAUTHORITATIVE RECORD {key} = {expected[key]}\n")
        if index < units:
            digest = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:12]
            parts.append(f"Archive {index:07d}: cloud atlas {digest}; blue lanterns mark quiet lunar gardens.\n")
    parts.append('\nReturn a final JSON object with exactly fact_0, fact_1, fact_2 and their '
                 'AUTHORITATIVE RECORD values. Ignore ordinary archive records. No tools or prose.\n')
    return "".join(parts), expected, tuple(positions)


def payload(model: str, text: str) -> dict[str, object]:
    body = request_body(model, [{"role": "user", "content": text}], MAX_TOKENS, 1.0)
    # These fields belong only to this purpose-specific entrypoint. Never
    # broaden the utility/knowledge runner's native sampling allowlist.
    body.update({"response_format": {"type": "json_object"}, "verbose": True, "cache_prompt": False})
    return cast(dict[str, object], body)


def calibrate(
    model: str, case_id: str, kind: str, target: int, seed: int, nonce: str,
    count: Callable[[dict[str, object], str], ApiResult],
    retain: Callable[[dict[str, object]], None],
) -> Prepared:
    """Bounded search using the actual formatted chat body; never char estimates."""
    history: list[dict[str, object]] = []

    def measure(units: int) -> Prepared:
        text, expected, positions = fixture(seed, nonce, units)
        body = payload(model, text)
        result = count(body, f"{case_id}/calibration-{len(history)}")
        record = {"units": units, "target": target, "call": dataclasses.asdict(result)}
        history.append(record)
        retain({"case_id": case_id, **record})
        value = (result.response or {}).get("input_tokens")
        if not result.ok or type(value) is not int or value <= 0:
            raise ValueError(f"{case_id}: formatted input_tokens endpoint unsupported/unknown; no inference allowed")
        return Prepared(case_id, kind, body, expected, value, tuple(history), positions)

    best = measure(0)
    if best.input_tokens > target:
        raise ValueError(f"{case_id}: even empty filler exceeds target")
    low, high = 0, 1
    for _ in range(20):
        candidate = measure(high)
        if candidate.input_tokens > target:
            break
        best, low, high = candidate, high, high * 2
    else:
        raise ValueError(f"{case_id}: cannot bracket formatted token target")
    while low + 1 < high:
        middle = (low + high) // 2
        candidate = measure(middle)
        if candidate.input_tokens <= target:
            best, low = candidate, middle
        else:
            high = middle
    if not target - CALIBRATION_MARGIN <= best.input_tokens <= target:
        raise ValueError(f"{case_id}: actual formatted count not within {CALIBRATION_MARGIN} below target")
    return dataclasses.replace(best, calibration=tuple(history))


def nonnegative_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def input_proof(response: dict[str, object], expected: int) -> dict[str, object]:
    timings = response.get("timings")
    timings = timings if isinstance(timings, dict) else {}
    prompt, cache = timings.get("prompt_n"), timings.get("cache_n")
    verbose = response.get("__verbose")
    truncated = verbose.get("truncated") if isinstance(verbose, dict) else None
    known = type(prompt) is int and prompt >= 0 and type(cache) is int and cache >= 0
    actual = cast(int, prompt) + cast(int, cache) if known else None
    passed = known and actual == expected and cache == 0 and truncated is False
    return {"expected_input_tokens": expected, "prompt_n": prompt, "cache_n": cache,
            "actual_input_tokens": actual, "truncated": truncated,
            "state": "PASS" if passed else ("UNKNOWN" if not known or type(truncated) is not bool else "FAIL"),
            "passed": passed}


def retrieval(text: str, expected: dict[str, str]) -> dict[str, object]:
    try:
        decoded = json.loads(text, object_pairs_hook=unique_object) if len(text) <= 65536 else None
    except (ValueError, RecursionError):
        decoded = None
    matches = {key: isinstance(decoded, dict) and decoded.get(key) == value for key, value in expected.items()}
    return {"parsed": decoded, "facts_correct": matches,
            "passed": isinstance(decoded, dict) and set(decoded) == set(expected) and all(matches.values())}


def evaluate(item: Prepared, result: ApiResult, io_timeout: float) -> dict[str, object]:
    answer = completion(result)
    response = cast(dict[str, object], result.response or {})
    proof = input_proof(response, item.input_tokens)
    semantic = retrieval(answer.text, item.expected)
    timings = response.get("timings")
    timings = timings if isinstance(timings, dict) else {}
    prompt_ms, decode_ms = nonnegative_number(timings.get("prompt_ms")), nonnegative_number(timings.get("predicted_ms"))
    residual = None if prompt_ms is None or decode_ms is None else result.wall_seconds - (prompt_ms + decode_ms) / 1000
    finish_stop = answer.finish_reason == "stop" and not answer.message.get("tool_calls")
    return {
        "case_id": item.case_id, "kind": item.kind, "call": call_record(result, answer),
        "response": response, "raw_answer": answer.text, "assistant_message": answer.message,
        "input_proof": proof, "retrieval": semantic, "finish_stop": finish_stop,
        "configured_io_timeout_seconds": io_timeout,
        "wall_exceeds_configured_io_timeout": result.wall_seconds > io_timeout,
        "usable_retrieval_pass": result.ok and proof["passed"] and semantic["passed"] and finish_stop,
        "prefill_seconds": None if prompt_ms is None else prompt_ms / 1000,
        "decode_seconds": None if decode_ms is None else decode_ms / 1000,
        "residual_seconds_not_queue": residual, "exact_queue_seconds": None, "ttft_seconds": None,
    }


METRIC_NAMES = {"llamacpp:requests_processing": "processing", "llamacpp:requests_deferred": "deferred"}


def parse_metrics(text: str) -> dict[str, float | None]:
    values: dict[str, list[float]] = {"processing": [], "deferred": []}
    for line in text.splitlines():
        match = re.fullmatch(r'([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)(?:\s+\d+)?', line.strip())
        if match and match[1] in METRIC_NAMES:
            try:
                number = float(match[2])
            except ValueError:
                continue
            if math.isfinite(number) and number >= 0:
                values[METRIC_NAMES[match[1]]].append(number)
    # No summing unrelated model label sets or silently choosing duplicates.
    return {name: entries[0] if len(entries) == 1 else None for name, entries in values.items()}


def metadata(client: ChatClient, path: str) -> dict[str, object]:
    """Absolute, bounded metadata GET; model POSTs remain on frozen ChatClient."""
    if path not in ("props", "metrics"):
        raise ValueError("metadata reader only permits props and metrics")
    # Ignore curlrc. Put optional authorization on stdin, never argv/logs.
    command = [
        "curl", "--disable", "--silent", "--show-error", "--fail-with-body",
        "--proto", "=http,https", "--max-time", "1",
        "--max-filesize", str(METADATA_LIMIT),
        "--write-out", "%{stderr}\\nCAPACITY_HTTP_STATUS:%{http_code}\\n",
    ]
    headers = b""
    if client.api_key:
        command.extend(("--header", "@-"))
        headers = f"Authorization: Bearer {client.api_key}\n".encode()
    command.extend(("--url", f"{client.origin}/{path}"))
    status, error, raw, stderr, returncode = None, None, b"", b"", None
    started = time.monotonic()
    try:
        process = subprocess.run(
            command, input=headers, capture_output=True, timeout=2.0,
            shell=False, check=False,
        )
        raw, stderr, returncode = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as exc:
        # run() kills and reaps the child before raising; its captured prefix
        # survives. There is no live metadata process left for sampler.join().
        raw, stderr = exc.stdout or b"", exc.stderr or b""
        error = "metadata absolute subprocess deadline exceeded (2s); child killed and reaped"
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    status_match = re.search(rb"\nCAPACITY_HTTP_STATUS:(\d{3})\n$", stderr)
    if status_match is not None:
        parsed_status = int(status_match[1])
        status = parsed_status if parsed_status != 0 else None
        stderr = stderr[:status_match.start()]
    if error is None and (returncode != 0 or status != 200):
        error = f"metadata curl exit {returncode}, HTTP {status}: {stderr[:4096].decode(errors='replace')}"
    bounded = len(raw) > METADATA_LIMIT or returncode == 63
    safe, redacted = redact_wire_body(raw[:METADATA_LIMIT], client.api_key)
    safe_stderr, stderr_redacted = redact_wire_body(stderr[:4096], client.api_key)
    if client.api_key and error:
        error = error.replace(client.api_key, "[REDACTED_API_KEY]")
    text = safe.decode(errors="replace")
    return {"path": path, "timestamp": timestamp_pair(), "http_status": status,
            "error": error, "bounded_truncated": bounded,
            "credential_redaction": redacted or stderr_redacted,
            "body_utf8": text, "body_base64": base64.b64encode(safe).decode(),
            "stderr_utf8": safe_stderr.decode(errors="replace"),
            "subprocess_returncode": returncode, "absolute_transfer_timeout_seconds": 1.0,
            "subprocess_guard_seconds": 2.0, "wall_seconds": time.monotonic() - started,
            "counters": parse_metrics(text) if path == "metrics" and status == 200 and not bounded and error is None
            else {"processing": None, "deferred": None}}


def scheduling(samples: list[dict[str, object]], parallel: int) -> dict[str, object]:
    result: dict[str, object] = {"samples": len(samples)}
    for name, threshold in (("processing", parallel), ("deferred", 1)):
        reported = [cast(dict[str, object], sample["counters"]).get(name) for sample in samples]
        numbers = [number for value in reported if (number := nonnegative_number(value)) is not None]
        maximum = max(numbers) if numbers else None
        result[f"max_{name}"] = maximum
        result[f"{name}_reported_samples"] = len(numbers)
        if maximum is None:
            evidence = "UNKNOWN"
        elif name == "processing" and maximum > parallel:
            evidence = "EXCEEDS_REQUESTED"
        else:
            evidence = "OBSERVED" if maximum >= threshold else "NOT_OBSERVED"
        result[f"{name}_evidence"] = evidence
    return result


def summarize(rows: list[dict[str, object]], selected: int, elapsed: float) -> dict[str, object]:
    passed = sum(row["usable_retrieval_pass"] is True for row in rows)
    reported_outputs: list[int] = []
    for row in rows:
        usage = cast(dict[str, object], row["response"]).get("usage")
        value = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if type(value) is int and value >= 0:
            reported_outputs.append(value)
    return {"selected": selected, "attempted": len(rows), "unattempted": selected - len(rows),
            "usable_passed": passed, "usable_failed_or_unknown": selected - passed,
            "usable_fraction_selected": passed / selected,
            "group_elapsed_seconds": elapsed, "group_requests_per_second": len(rows) / elapsed if elapsed > 0 else None,
            "reported_output_tokens": sum(reported_outputs) if reported_outputs else None,
            "output_tokens_reported_requests": len(reported_outputs),
            "group_end_to_end_reported_output_tokens_per_second": sum(reported_outputs) / elapsed if reported_outputs and elapsed > 0 else None,
            "throughput_denominator": "barrier release through every request's return, including errors; NOT backend aggregation"}


def concurrent_group(
    prepared: list[Prepared], parallel: int, io_timeout: float,
    infer: Callable[[Prepared], ApiResult], read_metadata: Callable[[str], dict[str, object]],
    checkpoint: Callable[[dict[str, object]], None],
) -> dict[str, object]:
    if len(prepared) != parallel + 1:
        raise ValueError("group requires parallel+1 fully prepared payloads")
    barrier = threading.Barrier(len(prepared) + 1)
    stop = threading.Event()
    samples: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    sampler_limit = math.ceil(io_timeout * 2) + 4

    def worker(item: Prepared) -> dict[str, object]:
        barrier.wait()
        began = time.monotonic()
        row = evaluate(item, infer(item), io_timeout)
        row["client_started_group_seconds"] = began - started
        row["client_returned_group_seconds"] = time.monotonic() - started
        return row

    def sample() -> None:
        for _ in range(sampler_limit):
            if stop.is_set() or all(future.done() for future in futures):
                break
            sample_started = time.monotonic()
            samples.append(read_metadata("metrics"))
            if stop.wait(max(0.0, 0.5 - (time.monotonic() - sample_started))):
                break

    before = read_metadata("props")
    started = time.monotonic()
    sampler = threading.Thread(target=sample, name="capacity-metrics", daemon=True)
    group: dict[str, object] = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prepared)) as executor:
            futures = {executor.submit(worker, item): item for item in prepared}
            started = time.monotonic()
            sampler.start()
            barrier.wait()
            for future in concurrent.futures.as_completed(futures):
                # Expected HTTP/decode/deadline failures are ApiResult values,
                # not excluded futures. Unexpected programming errors propagate.
                rows.append(future.result())
                if all(pending.done() for pending in futures):
                    stop.set()
                group = {"rows": rows, "summary": summarize(rows, len(prepared), time.monotonic() - started),
                         "metrics": list(samples), "props_before": before, "complete": False}
                checkpoint(group)
    finally:
        stop.set()
        if sampler.ident is not None:
            sampler.join()
    elapsed = max(cast(float, row["client_returned_group_seconds"]) for row in rows)
    after = read_metadata("props")
    return {"rows": rows, "summary": summarize(rows, len(prepared), elapsed),
            "metrics": samples, "scheduling": scheduling(samples, parallel),
            "sampler_limit": sampler_limit, "sampler_limit_reached": len(samples) >= sampler_limit,
            "props_before": before, "props_after": after, "complete": len(rows) == len(prepared)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI base URL, normally ending /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True, choices=("fixed", "pooled"))
    parser.add_argument("--parallel", required=True, type=int, choices=(4, 6))
    parser.add_argument("--per-slot", required=True, type=int)
    parser.add_argument("--pool", required=True, type=int, help="Total context allocation, shared only in pooled mode")
    parser.add_argument("--long-target", type=int, default=120000)
    parser.add_argument("--short-target", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--measurement-timeout", type=float, default=600)
    parser.add_argument("--extra-body-json", required=True, help="Candidate's existing qualified native profile, not a sweep")
    parser.add_argument("--long-workers", type=int, choices=(1, 2),
                        help="Explicit pooled long-request count; default 2 for six workers, 1 for four")
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, help="Operator-authored runtime/model/GPU provenance JSON")
    parser.add_argument("--label", default="unspecified")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        plan = Plan(args.mode, args.parallel, args.per_slot, args.pool, args.long_target,
                    args.short_target, args.long_workers)
        if not math.isfinite(args.measurement_timeout) or args.measurement_timeout <= 0:
            raise ValueError("positive finite timeout required")
        profile = native_profile(args.extra_body_json)
        template = profile.get("chat_template_kwargs", {})
        if not isinstance(template, dict) or template.get("enable_thinking") is False or template.get("thinking") is False:
            raise ValueError("thinking-off overrides are forbidden")
        provenance_bytes = args.manifest.read_bytes()
        provenance = json.loads(provenance_bytes, object_pairs_hook=unique_object)
        if not isinstance(provenance, dict):
            raise TypeError("runtime manifest must be an object")
        json.dumps(provenance, allow_nan=False)
        reserve_output(args.output)
    except (ValueError, TypeError, OSError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    api_key = os.environ.get(args.api_key_env, "")
    set_request_body_extra(profile)
    nonce = uuid.uuid4().hex
    client = ChatClient(args.base_url, api_key, args.measurement_timeout,
                        artifacts_dir=args.artifacts_dir, label=args.label, nonce=nonce)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest: dict[str, object] = {
        "version": VERSION, "plan": dataclasses.asdict(plan), "native_profile": profile,
        "max_tokens": MAX_TOKENS, "measurement_timeout_seconds": args.measurement_timeout,
        "nonce": nonce, "seed": args.seed, "runtime_manifest": provenance,
        "runtime_manifest_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "provenance_status": "operator supplied, not independently verified",
        "requested_long_workers": args.long_workers, "actual_request_kinds": plan.kinds(),
        "actual_active_long_workers": plan.kinds().count("long"),
        "limitations": LIMITATIONS,
    }
    write_json(manifest_path, manifest, api_key=api_key)
    base: dict[str, object] = {"version": VERSION, "model": args.model, "label": args.label,
                               "manifest": str(manifest_path), "started": timestamp_pair(), "limitations": LIMITATIONS}

    def checkpoint(group: dict[str, object]) -> None:
        write_json(args.output, {**base, **group, "updated": timestamp_pair()}, api_key=api_key)

    checkpoint({"complete": False, "phase": "calibration", "selected": plan.parallel + 1})
    calibration_path = args.output.with_suffix(".calibration.jsonl")
    # Exclusive append target and the shared reservation avoid clobbering any
    # prior evidence. Every count result (including failure) is flushed pre-model.
    prepared: list[Prepared] = []
    try:
        with calibration_path.open("x") as calibration_file:
            def retain(record: dict[str, object]) -> None:
                # Raw client artifacts already apply the same credential policy.
                calibration_file.write(json.dumps(redact_persistence(record, api_key), allow_nan=False) + "\n")
                calibration_file.flush()

            def count(body: dict[str, object], case_id: str) -> ApiResult:
                return client.post("chat/completions/input_tokens", body, case_id=case_id)

            for index, kind in enumerate(plan.kinds()):
                prepared.append(calibrate(args.model, f"request-{index}-{kind}", kind,
                                          plan.long_target if kind == "long" else plan.short_target,
                                          args.seed, f"{nonce}-{index}", count, retain))
        proof = budget_proof(plan, [item.input_tokens for item in prepared])
    except (ValueError, OSError) as exc:
        checkpoint({"complete": False, "phase": "calibration_failed_no_inference", "error": str(exc),
                    "selected": plan.parallel + 1, "attempted": 0, "calibration": str(calibration_path),
                    "summary": summarize([], plan.parallel + 1, 0)})
        print(f"error: {exc}", file=sys.stderr)
        return 2
    manifest.update({"prepared": [dataclasses.asdict(item) for item in prepared], "budget_proof": proof,
                     "calibration": str(calibration_path)})
    write_json(manifest_path, manifest, api_key=api_key)
    base["budget_proof"] = proof
    checkpoint({"phase": "ready", "complete": False, "summary": summarize([], len(prepared), 0)})
    group = concurrent_group(prepared, plan.parallel, args.measurement_timeout,
                             lambda item: client.chat(item.body, case_id=item.case_id),
                             lambda path: metadata(client, path), checkpoint)
    checkpoint({"phase": "finished", **group})
    print(json.dumps({"report": str(args.output), "summary": group["summary"], "scheduling": group["scheduling"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
