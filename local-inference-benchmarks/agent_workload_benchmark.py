#!/usr/bin/env python3
"""Realistic long-context and tool-loop benchmark for local chat servers.

The earlier chat_benchmark.py is useful for decode throughput, but its input
prompts are tiny.  This harness targets the workloads that select a practical
agent model:

* a deterministic code-review corpus calibrated near 32K input tokens;
* cold and warm-prefix reviews with 512/2K/4K output budgets;
* a growing, structured multi-turn tool loop;
* host-specific full-concurrency runs; and
* an optional near-contract context probe.

It uses only the Python standard library and OpenAI-compatible endpoints.  For
llama.cpp, the response's ``timings`` object gives prompt/decode measurements
and cache reuse directly.  Results intentionally omit generated text and API
keys; only hashes, sizes, timing, usage, tool validity, and finish reasons are
written to disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MEASUREMENT_TIMEOUT = 900.0
DEFAULT_USABILITY_CUTOFF = 60.0
REQUEST_BODY_EXTRA: dict[str, Any] = {}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_file",
            "description": "Read one repository file with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search repository text and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run a bounded test target and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_diff",
            "description": "Return the current diff for a repository path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_SEQUENCE = (
    "inspect_file",
    "search_code",
    "run_tests",
    "get_diff",
    "inspect_file",
    "search_code",
    "run_tests",
    "inspect_file",
    "get_diff",
    "search_code",
    "inspect_file",
    "run_tests",
)


REVIEW_SIGNAL_PATTERNS = {
    "dictionary_mutation_during_iteration": (
        r"dictionary changed size",
        r"mutat(?:e|ing|ion).{0,40}(?:dict|cache).{0,40}iterat",
        r"delet(?:e|ing).{0,40}(?:dict|cache).{0,40}iterat",
    ),
    "tenant_prefix_collision": (
        r"prefix collision",
        r"tenant-?1.{0,50}tenant-?10",
        r"startswith.{0,80}(?:delimiter|boundary|tenant)",
    ),
    "retry_timeout_or_status_gap": (
        r"retry.{0,80}(?:timeout|unbounded|status)",
        r"second (?:request|response).{0,80}(?:timeout|status|check)",
        r"retri(?:ed|es).{0,80}(?:without|no).{0,40}timeout",
    ),
    "revision_ordering_bug": (
        r"revision.{0,80}(?:sort|order|latest)",
        r"merge.{0,80}(?:input|list).{0,40}order",
        r"latest.{0,80}(?:last|order|sort)",
    ),
}


@dataclasses.dataclass
class ApiResult:
    ok: bool
    wall_seconds: float
    status: int | None = None
    response: dict[str, Any] | None = None
    error: str | None = None


class ChatClient:
    def __init__(self, base_url: str, api_key: str, measurement_timeout: float):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.measurement_timeout = measurement_timeout
        parsed = urllib.parse.urlsplit(self.base_url)
        self.origin = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def post(self, path: str, body: dict[str, Any]) -> ApiResult:
        url = f"{self.base_url}/{path.lstrip('/')}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body, separators=(",", ":")).encode(),
            headers=self._headers(),
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.measurement_timeout) as response:
                payload = json.loads(response.read())
                return ApiResult(
                    ok=True,
                    status=response.status,
                    wall_seconds=time.monotonic() - started,
                    response=payload,
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode(errors="replace")
            return ApiResult(
                ok=False,
                status=exc.code,
                wall_seconds=time.monotonic() - started,
                error=f"HTTP {exc.code}: {detail}",
            )
        except Exception as exc:  # preserve a result for every attempted cell
            return ApiResult(
                ok=False,
                wall_seconds=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    def chat(self, body: dict[str, Any]) -> ApiResult:
        return self.post("chat/completions", body)

    def tokenize(self, content: str) -> int | None:
        req = urllib.request.Request(
            f"{self.origin}/tokenize",
            data=json.dumps({"content": content}, separators=(",", ":")).encode(),
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = json.loads(response.read())
            tokens = payload.get("tokens")
            return len(tokens) if isinstance(tokens, list) else None
        except Exception:
            return None


def code_module(index: int, nonce: str) -> str:
    """Return deterministic, varied code with reviewable cross-file behavior."""
    timeout = 20 + (index % 11)
    shard = index % 17
    return f'''\n=== services/shard_{index:03d}.py ===
"""Worker shard {index}; benchmark fixture {nonce}."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

CACHE: dict[str, "Record{index}"] = {{}}

@dataclass
class Record{index}:
    tenant_id: str
    revision: int
    payload: dict[str, Any]

    def cache_key(self) -> str:
        return f"{{self.tenant_id}}:{shard}:{{self.revision}}"

async def fetch_{index}(client: Any, tenant_id: str, revision: int) -> Record{index}:
    key = f"{{tenant_id}}:{shard}:{{revision}}"
    if key in CACHE:
        return CACHE[key]
    response = await asyncio.wait_for(
        client.get("/v1/tenants/%s/records/%s" % (tenant_id, revision)),
        timeout={timeout},
    )
    if response.status_code >= 500:
        await asyncio.sleep(0.01 * ({index} % 5))
        response = await client.get("/v1/tenants/%s/records/%s" % (tenant_id, revision))
    data = response.json()
    record = Record{index}(tenant_id=tenant_id, revision=revision, payload=data)
    CACHE[key] = record
    return record

async def merge_{index}(client: Any, tenant_id: str, revisions: list[int]) -> dict[str, Any]:
    records = await asyncio.gather(*(fetch_{index}(client, tenant_id, rev) for rev in revisions))
    merged: dict[str, Any] = {{}}
    for record in records:
        merged.update(record.payload)
    return merged

def invalidate_{index}(tenant_id: str) -> int:
    removed = 0
    for key in CACHE:
        if key.startswith(tenant_id):
            del CACHE[key]
            removed += 1
    return removed

=== tests/test_shard_{index:03d}.py ===
import pytest

@pytest.mark.asyncio
async def test_latest_revision_wins_{index}(fake_client):
    result = await merge_{index}(fake_client, "tenant-{index % 9}", [3, 1, 2])
    assert result["revision"] == 3

def test_invalidation_is_tenant_scoped_{index}():
    CACHE["tenant-{index % 9}:{shard}:1"] = object()
    CACHE["tenant-{(index + 1) % 9}:{shard}:1"] = object()
    assert invalidate_{index}("tenant-{index % 9}") == 1
'''


def build_review_prompt(modules: int, nonce: str) -> str:
    header = f'''You are reviewing a production Python change for correctness, security,
concurrency, resource lifecycle, tenant isolation, and test gaps. The repository
snapshot below is complete for this review. Identify concrete defects, cite file
and line-level evidence, explain triggering conditions and impact, and prioritize
only actionable findings. Do not praise the code or summarize unaffected files.
Benchmark fixture nonce: {nonce}
'''
    corpus = "".join(code_module(i, nonce) for i in range(modules))
    footer = '''\n=== REVIEW REQUEST ===
Return a concise code review with the highest-risk findings first. For every
finding give the affected file, the faulty behavior, a realistic trigger, and a
specific repair. Check whether repeated patterns create cross-tenant or
concurrency failures rather than listing every duplicate instance.
'''
    return header + corpus + footer


def calibrate_prompt(client: ChatClient, target_tokens: int, nonce: str) -> tuple[str, int | None, int]:
    """Use llama.cpp /tokenize when available; otherwise use a stable estimate."""
    low, high = 1, max(16, target_tokens // 100)
    high_prompt = build_review_prompt(high, nonce)
    high_count = client.tokenize(high_prompt)
    if high_count is None:
        modules = max(1, round(target_tokens / 275))
        return build_review_prompt(modules, nonce), None, modules
    while high_count < target_tokens and high < 2048:
        low = high
        high *= 2
        high_prompt = build_review_prompt(high, nonce)
        high_count = client.tokenize(high_prompt)
        if high_count is None:
            break
    best_modules, best_count = high, high_count
    while low <= high:
        mid = (low + high) // 2
        prompt = build_review_prompt(mid, nonce)
        count = client.tokenize(prompt)
        if count is None:
            break
        if best_count is None or abs(count - target_tokens) < abs(best_count - target_tokens):
            best_modules, best_count = mid, count
        if count < target_tokens:
            low = mid + 1
        elif count > target_tokens:
            high = mid - 1
        else:
            best_modules, best_count = mid, count
            break
    return build_review_prompt(best_modules, nonce), best_count, best_modules


def request_body(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] = "auto",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
        if temperature == 0:
            body["top_k"] = 1
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    body.update(REQUEST_BODY_EXTRA)
    return body


def set_request_body_extra(extra: dict[str, Any]) -> None:
    global REQUEST_BODY_EXTRA
    REQUEST_BODY_EXTRA = dict(extra)


def summarize_result(
    label: str,
    result: ApiResult,
    usability_cutoff: float,
    prompt_sha256: str,
    prompt_chars: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "label": label,
        "ok": result.ok,
        "http_status": result.status,
        "wall_seconds": round(result.wall_seconds, 6),
        "within_usability_cutoff": result.ok and result.wall_seconds <= usability_cutoff,
        "usability_cutoff_seconds": usability_cutoff,
        "prompt_sha256": prompt_sha256,
        "prompt_chars": prompt_chars,
    }
    if not result.ok or result.response is None:
        summary["error"] = result.error
        return summary
    response = result.response
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    review_text = reasoning + "\n" + content
    review_signals = {
        name: any(re.search(pattern, review_text, re.IGNORECASE | re.DOTALL) for pattern in patterns)
        for name, patterns in REVIEW_SIGNAL_PATTERNS.items()
    }
    summary.update(
        {
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "timings": timings,
            "cache_tokens": timings.get("cache_n")
            if "cache_n" in timings
            else (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
            "tool_call_count": len(tool_calls),
            "tool_names": [((call.get("function") or {}).get("name")) for call in tool_calls],
            "output_sha256": hashlib.sha256((reasoning + "\n" + content).encode()).hexdigest(),
            "review_signals": review_signals,
            "review_signal_count": sum(review_signals.values()),
        }
    )
    return summary


def fake_tool_output(name: str, step: int) -> str:
    if name == "inspect_file":
        return (
            f"services/shard_{step:03d}.py:31-37\n"
            "31 def invalidate(tenant_id):\n"
            "32     for key in CACHE:\n"
            "33         if key.startswith(tenant_id):\n"
            "34             del CACHE[key]\n"
            "Runtime note: mutation during dictionary iteration raises RuntimeError.\n"
        )
    if name == "search_code":
        return "\n".join(
            f"services/shard_{i:03d}.py:33: if key.startswith(tenant_id):"
            for i in range(step, step + 24)
        )
    if name == "run_tests":
        return (
            "pytest -q tests/test_tenant_cache.py\n"
            "17 passed, 3 failed in 4.81s\n"
            "FAILED test_prefix_collision[tenant-1-tenant-10]\n"
            "FAILED test_invalidate_while_iterating\n"
            "FAILED test_retry_timeout_is_bounded\n"
        )
    if name == "get_diff":
        return (
            "@@ -18,8 +18,14 @@ async def fetch(...):\n"
            "+ if response.status_code >= 500:\n"
            "+     await asyncio.sleep(0.02)\n"
            "+     response = await client.get(path)\n"
            "  data = response.json()\n"
        )
    return "tool returned no data"


def run_reviews(
    client: ChatClient,
    model: str,
    prompt: str,
    output_tokens: list[int],
    temperature: float | None,
    cutoff: float,
) -> list[dict[str, Any]]:
    messages = [{"role": "user", "content": prompt}]
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    rows: list[dict[str, Any]] = []
    for max_tokens in output_tokens:
        result = client.chat(request_body(model, messages, max_tokens, temperature))
        rows.append(
            summarize_result(
                f"review_{max_tokens}", result, cutoff, digest, len(prompt)
            )
        )
    warm = client.chat(request_body(model, messages, 128, temperature))
    rows.append(summarize_result("review_warm_128", warm, cutoff, digest, len(prompt)))
    return rows


def run_tool_loop(
    client: ChatClient,
    model: str,
    prompt: str,
    steps: int,
    tool_max_tokens: int,
    final_max_tokens: int,
    final_prompt: str,
    temperature: float | None,
    cutoff: float,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a production code-review agent. Use the supplied tools exactly when "
                "requested. Do not invent tool results. Preserve prior findings across turns."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()
    turns: list[dict[str, Any]] = []
    valid_calls = 0
    sequence = TOOL_SEQUENCE[:steps]
    for index, expected in enumerate(sequence):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Investigation step {index + 1}: call {expected} now with a concrete argument "
                    "for the repository above. Return only the structured tool call."
                ),
            }
        )
        result = client.chat(
            request_body(model, messages, tool_max_tokens, temperature, TOOLS)
        )
        row = summarize_result(
            f"tool_turn_{index + 1}_{expected}", result, cutoff, prompt_digest, len(prompt)
        )
        row["expected_tool"] = expected
        calls: list[dict[str, Any]] = []
        if result.ok and result.response:
            choice = (result.response.get("choices") or [{}])[0]
            assistant = choice.get("message") or {}
            calls = assistant.get("tool_calls") or []
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant.get("content") or "",
                    "tool_calls": calls,
                }
            )
        valid = False
        if calls:
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                    args_valid = isinstance(arguments, dict)
                except json.JSONDecodeError:
                    args_valid = False
                if name == expected and args_valid:
                    valid = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or f"call_{index}_{name}",
                        "name": name,
                        "content": fake_tool_output(name, index),
                    }
                )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"The expected {expected} call was missing; continue to the next step.",
                }
            )
        row["valid_expected_tool_call"] = valid
        valid_calls += int(valid)
        turns.append(row)
    messages.append(
        {
            "role": "user",
            "content": final_prompt,
        }
    )
    # Preserve the agent's tool schema on the final turn. Some runtimes omit
    # tools from the rendered prompt when tool_choice is "none", which breaks
    # an otherwise reusable long prefix and turns the final review cold.
    final = client.chat(
        request_body(
            model,
            messages,
            final_max_tokens,
            temperature,
            TOOLS,
            tool_choice="auto",
        )
    )
    final_row = summarize_result("tool_loop_final", final, cutoff, prompt_digest, len(prompt))
    return {
        "requested_turns": len(sequence),
        "valid_tool_calls": valid_calls,
        "all_tool_calls_valid": valid_calls == len(sequence),
        "all_turns_within_cutoff": all(row["within_usability_cutoff"] for row in turns)
        and final_row["within_usability_cutoff"],
        "turns": turns,
        "final": final_row,
    }


def run_concurrency(
    client: ChatClient,
    model: str,
    target_prompt_tokens: int,
    workers: int,
    max_tokens: int,
    temperature: float | None,
    cutoff: float,
) -> dict[str, Any]:
    prompts: list[str] = []
    counts: list[int | None] = []
    for index in range(workers):
        prompt, count, _ = calibrate_prompt(
            client, target_prompt_tokens, f"concurrent-{index}-{time.time_ns()}"
        )
        prompts.append(prompt)
        counts.append(count)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                client.chat,
                request_body(model, [{"role": "user", "content": prompt}], max_tokens, temperature),
            )
            for prompt in prompts
        ]
        results = [future.result() for future in futures]
    wall = time.monotonic() - started
    rows = [
        summarize_result(
            f"concurrent_{index + 1}",
            result,
            cutoff,
            hashlib.sha256(prompts[index].encode()).hexdigest(),
            len(prompts[index]),
        )
        for index, result in enumerate(results)
    ]
    completion_tokens = sum((row.get("usage") or {}).get("completion_tokens", 0) for row in rows)
    return {
        "workers": workers,
        "target_prompt_tokens": target_prompt_tokens,
        "raw_prompt_token_counts": counts,
        "wall_seconds": round(wall, 6),
        "aggregate_completion_tokens_per_second": round(completion_tokens / wall, 6)
        if wall
        else None,
        "all_within_cutoff": all(row["within_usability_cutoff"] for row in rows),
        "requests": rows,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def add_context_needles(prompt: str, seed: str) -> tuple[str, list[str]]:
    values = [hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:20] for index in range(3)]
    insertions = [
        (0.08, f"\n# AUDIT_NEEDLE_ALPHA={values[0]}\n"),
        (0.51, f"\n# AUDIT_NEEDLE_BRAVO={values[1]}\n"),
        (0.91, f"\n# AUDIT_NEEDLE_CHARLIE={values[2]}\n"),
    ]
    offset = 0
    result = prompt
    for fraction, marker in insertions:
        index = int(len(prompt) * fraction) + offset
        result = result[:index] + marker + result[index:]
        offset += len(marker)
    result += (
        "\nReturn exactly the three audit needle values in ALPHA, BRAVO, CHARLIE order, "
        "separated by one space, with no other text.\n"
    )
    return result, values


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = list(report.get("reviews") or [])
    tool_loop = report.get("tool_loop") or {}
    rows.extend(tool_loop.get("turns") or [])
    if tool_loop.get("final"):
        rows.append(tool_loop["final"])
    concurrent = report.get("concurrency") or {}
    rows.extend(concurrent.get("requests") or [])
    context_probe = report.get("context_probe")
    if context_probe:
        rows.append(context_probe)
    walls = [float(row["wall_seconds"]) for row in rows if row.get("ok")]
    prompt_tps = [
        float(row["timings"]["prompt_per_second"])
        for row in rows
        if (row.get("timings") or {}).get("prompt_per_second") is not None
    ]
    decode_tps = [
        float(row["timings"]["predicted_per_second"])
        for row in rows
        if (row.get("timings") or {}).get("predicted_per_second") is not None
    ]
    return {
        "request_count": len(rows),
        "successful_requests": sum(int(row.get("ok", False)) for row in rows),
        "within_cutoff": sum(int(row.get("within_usability_cutoff", False)) for row in rows),
        "wall_p50_seconds": statistics.median(walls) if walls else None,
        "wall_p95_seconds": percentile(walls, 0.95),
        "prompt_tps_p50": statistics.median(prompt_tps) if prompt_tps else None,
        "decode_tps_p50": statistics.median(decode_tps) if decode_tps else None,
        "tool_calls_valid": tool_loop.get("valid_tool_calls"),
        "tool_calls_requested": tool_loop.get("requested_turns"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--api-key-env",
        default="LOCAL_LLM_API_KEY",
        help="Environment variable containing the API key; the key is never recorded",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("all", "review", "tool-loop", "concurrency", "context"),
        default="all",
    )
    parser.add_argument("--target-prompt-tokens", type=int, default=32768)
    parser.add_argument(
        "--fixture-modules",
        type=int,
        help=(
            "Bypass endpoint token calibration and render exactly this many fixture "
            "modules. Use for runtimes without a tokenize endpoint after calibrating "
            "the fixture against response usage.prompt_tokens."
        ),
    )
    parser.add_argument(
        "--fixture-nonce",
        help=(
            "Use a stable fixture nonce for same-prompt runtime A/B tests. "
            "The default includes the current time to force a cold prefix."
        ),
    )
    parser.add_argument("--context-probe-tokens", type=int, default=120000)
    parser.add_argument("--context-output-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", default="512,2048,4096")
    parser.add_argument("--tool-turns", type=int, default=10)
    parser.add_argument("--tool-max-tokens", type=int, default=1024)
    parser.add_argument("--tool-final-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--tool-final-prompt",
        default=(
            "Now provide the final review. Include only supported findings discovered "
            "from the repository and tool results, ordered by severity."
        ),
        help="Final user instruction for the tool-loop synthesis request",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--concurrency-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help="JSON object merged into every chat request (for example reasoning controls)",
    )
    parser.add_argument("--measurement-timeout", type=float, default=DEFAULT_MEASUREMENT_TIMEOUT)
    parser.add_argument("--usability-cutoff", type=float, default=DEFAULT_USABILITY_CUTOFF)
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
    nonce = args.fixture_nonce or f"{args.label}-{time.time_ns()}"
    if args.fixture_modules is not None:
        if args.fixture_modules < 1:
            print("error: --fixture-modules must be positive", file=sys.stderr)
            return 2
        modules = args.fixture_modules
        prompt = build_review_prompt(modules, nonce)
        raw_tokens = client.tokenize(prompt)
        calibration_method = "fixture_modules_override"
    else:
        prompt, raw_tokens, modules = calibrate_prompt(
            client, args.target_prompt_tokens, nonce
        )
        calibration_method = (
            "endpoint_tokenize" if raw_tokens is not None else "character_estimate"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": args.label,
        "endpoint": args.base_url,
        "model": args.model,
        "suite": args.suite,
        "target_prompt_tokens": args.target_prompt_tokens,
        "raw_prompt_tokens": raw_tokens,
        "prompt_calibration_method": calibration_method,
        "fixture_nonce": args.fixture_nonce,
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "fixture_modules": modules,
        "tool_final_prompt": args.tool_final_prompt,
        "temperature": args.temperature,
        "extra_body": extra_body,
        "measurement_timeout_seconds": args.measurement_timeout,
        "usability_cutoff_seconds": args.usability_cutoff,
    }
    output_tokens = [int(value) for value in args.output_tokens.split(",") if value]
    if args.suite in ("all", "review"):
        report["reviews"] = run_reviews(
            client,
            args.model,
            prompt,
            output_tokens,
            args.temperature,
            args.usability_cutoff,
        )
    if args.suite in ("all", "tool-loop"):
        report["tool_loop"] = run_tool_loop(
            client,
            args.model,
            prompt,
            min(args.tool_turns, len(TOOL_SEQUENCE)),
            args.tool_max_tokens,
            args.tool_final_max_tokens,
            args.tool_final_prompt,
            args.temperature,
            args.usability_cutoff,
        )
    if args.suite in ("all", "concurrency"):
        report["concurrency"] = run_concurrency(
            client,
            args.model,
            args.target_prompt_tokens,
            args.concurrency,
            args.concurrency_output_tokens,
            args.temperature,
            args.usability_cutoff,
        )
    if args.suite in ("all", "context"):
        context_prompt, context_tokens, _ = calibrate_prompt(
            client, args.context_probe_tokens, f"context-{time.time_ns()}"
        )
        context_prompt, expected_needles = add_context_needles(context_prompt, nonce)
        context_result = client.chat(
            request_body(
                args.model,
                [{"role": "user", "content": context_prompt}],
                args.context_output_tokens,
                0,
            )
        )
        context_summary = summarize_result(
            "context_probe",
            context_result,
            args.usability_cutoff,
            hashlib.sha256(context_prompt.encode()).hexdigest(),
            len(context_prompt),
        )
        context_summary["raw_prompt_tokens"] = context_tokens
        context_output = ""
        if context_result.ok and context_result.response:
            context_message = (
                ((context_result.response.get("choices") or [{}])[0].get("message")) or {}
            )
            context_output = (context_message.get("reasoning_content") or "") + "\n" + (
                context_message.get("content") or ""
            )
        context_summary["needle_recall"] = [needle in context_output for needle in expected_needles]
        context_summary["all_needles_recalled"] = all(context_summary["needle_recall"])
        report["context_probe"] = context_summary
    report["summary"] = compact_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
