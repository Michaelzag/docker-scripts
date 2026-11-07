#!/usr/bin/env python3
"""Score one fixed, local 32-question published-gold slice, not full MMLU-Pro.

Use case: native-byte fixtures + endpoint/profile -> retained attempts and honest
scores; gold never enters messages, every selected case stays in the denominator.
Side effects are local evidence writes and one HTTP call per selected prompt;
only endpoint, native sampling profile, budgets and answer-blind subset vary.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TypedDict, cast

from agent_workload_benchmark import ChatClient, request_body, set_request_body_extra, timestamp_pair
from intelligence_challenge_suite import call_record, completion, reserve_output, write_json

VERSION = "published-knowledge-mmlu-pro-32-v1"
CATEGORIES = ("math", "physics", "chemistry", "biology", "computer science", "engineering", "history", "philosophy")
FILES = {
    "gold": ("mmlu-pro-32-gold.json", "d619f2b3c81cc9795b9ea0f8ee6b22179eb4384b7247315c1d384d7298f51053"),
    "prompts": ("mmlu-pro-32-prompts.json", "eb096d0411983db0a4c807ac02f03536fdde109816b14306e81364f6305cdebc"),
    "metadata": ("mmlu-pro-32-metadata.json", "6a931821cad7ece3fb9b942c4f98d30cad5ea53d1b5168fa344c1360cc5fc225"),
}
PROMPT_FIELDS = {"case_id", "question_id", "category", "question", "options"}
GOLD_FIELDS = PROMPT_FIELDS | {"row_idx", "answer", "answer_index", "cot_content", "src"}
# Same allowlist as the frozen intelligence_challenge_suite.main validation.
PROFILE_KEYS = {
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
    "repeat_penalty", "repetition_penalty", "seed", "chat_template_kwargs",
}
INSTRUCTION = (
    "Solve the multiple-choice question. Use your native reasoning process. "
    "In your final answer content, return only the single letter of the selected option "
    "(A through J), with no explanation. Do not call tools."
)
PARSER_GRAMMAR = (
    "v1: Strip outer whitespace. Optionally unwrap exactly one full outer fence with "
    "opening ``` or ```json (case-insensitive), newline, body, newline, closing ```; "
    "no nested fences. The entire body must be either: (1) one A-J letter, optionally "
    "enclosed in matching parentheses with internal whitespace, optionally followed by "
    "one period; (2) case-insensitive exact prefix Answer: / The correct answer is / "
    "Option followed by that same letter form (whitespace required after is/Option); "
    "or (3) a JSON object with exactly one nonduplicate key answer whose value is "
    "exactly one A-J letter, case-insensitive. JSON whitespace is allowed. No other "
    "prose, alternatives, suffixes, keys, nested objects or extraction. At most 65536 "
    "content characters; JSON decode/recursion failures are invalid. Only final "
    "message.content from a stop finish without tool calls is eligible; never reasoning "
    "or length-truncated content. A parsed letter outside the question's options fails."
)
LIMITATIONS = (
    "Small published-gold slice, NOT full MMLU-Pro, IQ or proof of unseen generalization. "
    "32 answer-blind selected public test questions, four per category: total score "
    "granularity 3.125 percentage points, category granularity 25 points; wide "
    "uncertainty, small margins are not reliable rankings. Public-test training "
    "contamination is possible. Upstream used GPT-4 option augmentation and expert "
    "review; this campaign did not generate gold with an LLM and uses no LLM judge. "
    "Rows API was unversioned; contemporaneous HF main SHA is not a page revision "
    "binding. Retained exact-byte subset hashes are authoritative. Nonstreaming "
    "elapsed is not TTFT; missing usage is unknown. No cache flush, no retries or voting."
)


@dataclasses.dataclass(frozen=True, slots=True)
class Question:
    case_id: str
    question_id: int
    category: str
    question: str
    options: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset:
    questions: tuple[Question, ...]
    gold: dict[str, dict[str, object]]
    metadata: dict[str, object]
    hashes: dict[str, str]


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedChoice:
    choice: str | None
    reason: str


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_choice(text: str) -> ParsedChoice:
    """Apply only the predeclared finite grammar, never search for a letter."""
    if len(text) > 65536:
        return ParsedChoice(None, "content_too_long")
    body = text.strip()
    if body.startswith("```"):
        fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", body, re.IGNORECASE | re.DOTALL)
        if fence is None or "```" in fence[1]:
            return ParsedChoice(None, "invalid_fence")
        body = fence[1].strip()
    match = re.fullmatch(
        r"(?:(?:Answer:\s*|The correct answer is\s+|Option\s+))?"
        r"(?:([A-J])|\(\s*([A-J])\s*\))\.?", body, re.IGNORECASE,
    )
    if match is not None:
        return ParsedChoice((match[1] or match[2]).upper(), "letter_grammar")
    if body.startswith("{"):
        try:
            value = json.loads(body, object_pairs_hook=unique_object)
        except (ValueError, RecursionError):
            return ParsedChoice(None, "invalid_json")
        if isinstance(value, dict) and set(value) == {"answer"}:
            answer = value["answer"]
            if isinstance(answer, str) and re.fullmatch(r"[A-J]", answer, re.IGNORECASE):
                return ParsedChoice(answer.upper(), "json_answer")
        return ParsedChoice(None, "invalid_json_answer")
    return ParsedChoice(None, "not_in_final_answer_grammar")


class CallRecord(TypedDict):
    ok: bool
    wall_seconds: float
    http_status: int | None
    error: str | None
    failure: str | None
    finish_reason: str | None
    usage: object
    artifact: str | None


class CaseRow(TypedDict):
    case_id: str
    question_id: int
    category: str
    question: str
    options: tuple[str, ...]
    source: object
    source_row_index: object
    passed: bool
    parsed_choice: str | None
    parse_reason: str
    raw_answer: str
    assistant_message: dict[str, object]
    request: dict[str, object]
    calls: list[CallRecord]
    inference_seconds: float
    elapsed_seconds: float
    ttft_seconds: None


def question_from_record(row: dict[str, object]) -> Question:
    if set(row) != PROMPT_FIELDS:
        raise ValueError("prompt record must contain exactly the whitelist fields")
    case_id, question_id, category = row["case_id"], row["question_id"], row["category"]
    question, options = row["question"], row["options"]
    if not isinstance(case_id, str) or re.fullmatch(r"mmlu-pro-\d{3}", case_id) is None:
        raise ValueError("invalid case_id")
    if type(question_id) is not int or question_id < 0 or category not in CATEGORIES:
        raise ValueError(f"invalid question identity/category: {case_id}")
    if not isinstance(category, str) or not isinstance(question, str) or not question.strip():
        raise ValueError(f"invalid question text: {case_id}")
    if not isinstance(options, list) or not 2 <= len(options) <= 10:
        raise ValueError(f"invalid option count: {case_id}")
    if not all(isinstance(option, str) and option.strip() for option in options):
        raise ValueError(f"invalid options: {case_id}")
    return Question(case_id, question_id, category, question, tuple(options))


def validate_records(prompts: object, gold_rows: object, metadata: object, hashes: dict[str, str]) -> Dataset:
    if not isinstance(prompts, list) or not isinstance(gold_rows, list) or not isinstance(metadata, dict):
        raise ValueError("fixtures must be prompt/gold arrays and metadata object")
    if len(prompts) != 32 or len(gold_rows) != 32:
        raise ValueError("the fixed dataset must contain exactly 32 prompt and gold records")
    questions = []
    for row in prompts:
        if not isinstance(row, dict):
            raise ValueError("prompt record must be an object")
        questions.append(question_from_record(row))
    expected_ids = {f"mmlu-pro-{index:03d}" for index in range(1, 33)}
    if {q.case_id for q in questions} != expected_ids:
        raise ValueError("case IDs must be unique and exactly mmlu-pro-001..032")
    if len({q.question_id for q in questions}) != 32:
        raise ValueError("question IDs must be unique")
    if Counter(q.category for q in questions) != dict.fromkeys(CATEGORIES, 4):
        raise ValueError("the fixed dataset must contain four questions per category")
    by_id: dict[str, dict[str, object]] = {}
    for row in gold_rows:
        if not isinstance(row, dict) or set(row) != GOLD_FIELDS:
            raise ValueError("gold record schema mismatch")
        prompt = question_from_record({key: row[key] for key in PROMPT_FIELDS})
        if prompt.case_id in by_id:
            raise ValueError("duplicate gold case_id")
        index, answer = row["answer_index"], row["answer"]
        if type(index) is not int or not 0 <= index < len(prompt.options) or answer != chr(65 + index):
            raise ValueError(f"gold answer/index mismatch: {prompt.case_id}")
        if type(row["row_idx"]) is not int or row["row_idx"] < 0:
            raise ValueError("invalid source row index")
        if not isinstance(row["src"], str) or not row["src"].strip() or not isinstance(row["cot_content"], str):
            raise ValueError("invalid source/rationale schema")
        by_id[prompt.case_id] = row
    if set(by_id) != expected_ids:
        raise ValueError("prompt/gold case ID join mismatch")
    for question in questions:
        row = by_id[question.case_id]
        if question != question_from_record({key: row[key] for key in PROMPT_FIELDS}):
            raise ValueError(f"prompt/gold question mismatch: {question.case_id}")
    declared = metadata.get("files")
    if not isinstance(declared, dict) or declared.get("gold_sha256") != hashes["gold"] or declared.get("prompts_sha256") != hashes["prompts"]:
        raise ValueError("metadata-declared SHA256 mismatch")
    if declared.get("prompts_whitelist_fields") != ["case_id", "question_id", "category", "question", "options"]:
        raise ValueError("metadata prompt whitelist mismatch")
    return Dataset(tuple(questions), by_id, metadata, hashes)


def load_dataset(directory: Path) -> Dataset:
    values: dict[str, object] = {}
    hashes: dict[str, str] = {}
    for kind, (name, expected) in FILES.items():
        raw = (directory / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise ValueError(f"{name}: native-byte SHA256 does not match the fixed slice")
        hashes[kind] = digest
        values[kind] = json.loads(raw, object_pairs_hook=unique_object)
    return validate_records(values["prompts"], values["gold"], values["metadata"], hashes)


def build_messages(question: Question) -> list[dict[str, str]]:
    # This function cannot access gold, rationale, source metadata or raw rows.
    options = "\n".join(f"{chr(65 + index)}. {text}" for index, text in enumerate(question.options))
    return [{"role": "user", "content": f"{INSTRUCTION}\n\nQuestion:\n{question.question}\n\nOptions:\n{options}"}]


def native_profile(text: str) -> dict[str, object]:
    extra = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(extra, dict):
        raise ValueError("extra body must be a native sampling/profile JSON object")
    unsupported = set(extra) - PROFILE_KEYS
    if unsupported:
        raise ValueError("unsupported native profile keys (request-shape/budget overrides are forbidden): " + ", ".join(sorted(unsupported)))
    # Match the frozen CLI's accepted native keys, while refusing nonfinite JSON.
    json.dumps(extra, allow_nan=False)
    return extra


def select_questions(questions: tuple[Question, ...], cases: str, limit: int | None) -> tuple[Question, ...]:
    names = {name.strip() for name in cases.split(",") if name.strip()}
    if names - {question.case_id for question in questions}:
        raise ValueError("unknown case ID filter")
    if limit is not None and not 1 <= limit <= 32:
        raise ValueError("limit must be between 1 and 32")
    selected = tuple(question for question in questions if not names or question.case_id in names)
    return selected if limit is None else selected[:limit]


def run_case(question: Question, gold: dict[str, object], client: ChatClient, model: str, max_tokens: int) -> CaseRow:
    started = time.monotonic()
    payload = request_body(model, build_messages(question), max_tokens, 0)
    result = client.chat(payload, case_id=question.case_id)
    answer = completion(result)
    if answer.failure is not None:
        parsed = ParsedChoice(None, answer.failure)
    elif answer.finish_reason != "stop" or answer.message.get("tool_calls"):
        parsed = ParsedChoice(None, "unexpected_tools_or_finish")
    else:
        parsed = parse_choice(answer.text)
    if parsed.choice is not None and ord(parsed.choice) - 65 >= len(question.options):
        parsed = ParsedChoice(parsed.choice, "choice_outside_options")
    passed = parsed.choice == gold["answer"] and parsed.reason in ("letter_grammar", "json_answer")
    return {
        "case_id": question.case_id, "question_id": question.question_id,
        "category": question.category, "question": question.question, "options": question.options,
        "source": gold["src"], "source_row_index": gold["row_idx"],
        "passed": passed, "parsed_choice": parsed.choice, "parse_reason": parsed.reason,
        "raw_answer": answer.text, "assistant_message": answer.message,
        "request": payload, "calls": [cast(CallRecord, call_record(result, answer))],
        "inference_seconds": result.wall_seconds, "elapsed_seconds": time.monotonic() - started,
        "ttft_seconds": None,
    }


def summarize(rows: list[CaseRow], selected: tuple[Question, ...]) -> dict[str, object]:
    """Denominators are selected questions, never only parseable/successful answers."""
    groups: dict[str, object] = {}
    for category in ("all", *CATEGORIES):
        total = sum(category == "all" or question.category == category for question in selected)
        if not total:
            continue
        attempted = [row for row in rows if category == "all" or row["category"] == category]
        calls = [call for row in attempted for call in row["calls"]]
        passed = sum(row["passed"] is True for row in attempted)
        latencies = sorted(row["inference_seconds"] for row in attempted)
        usage = dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens"), 0)
        usage_counts = dict.fromkeys(usage, 0)
        for call in calls:
            reported = call["usage"]
            if isinstance(reported, dict):
                for key in usage:
                    value = reported.get(key)
                    if type(value) is int:
                        usage[key] += value
                        usage_counts[key] += 1
        groups[category] = {
            "passed": passed, "total": total, "attempted": len(attempted),
            "failed": len(attempted) - passed, "unattempted": total - len(attempted),
            "accuracy_selected": passed / total,
            "request_errors": sum(not call["ok"] for call in calls),
            "timeout_requests": sum(not call["ok"] and any(marker in str(call["error"]).casefold() for marker in ("timeout", "timed out")) for call in calls),
            "length_finishes": sum(call["finish_reason"] == "length" for call in calls),
            "invalid_answers": sum(row["parsed_choice"] is None for row in attempted),
            "inference_seconds_total": sum(latencies) if latencies else None,
            "inference_seconds_median": statistics.median(latencies) if latencies else None,
            "inference_seconds_p95_nearest_rank": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
            "elapsed_seconds_total": sum(row["elapsed_seconds"] for row in attempted) if attempted else None,
            "usage_reported_sum": usage,
            "requests_with_usage": sum(isinstance(call["usage"], dict) for call in calls),
            "usage_fields_reported_counts": usage_counts,
            "ttft_seconds": None,
        }
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Local immutable three-file slice; never downloaded")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Operator runtime/model/hash/host/GPU provenance JSON")
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--cases", default="", help="Comma-separated exact case IDs, in fixed fixture order")
    parser.add_argument("--limit", type=int, help="First N selected prompts, independent of answers/results")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--measurement-timeout", type=float, default=600)
    parser.add_argument("--extra-body-json", default="{}", help="Native sampling/thinking profile only; no tools or request/budget overrides")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.max_tokens <= 0 or not math.isfinite(args.measurement_timeout) or args.measurement_timeout <= 0:
            raise ValueError("positive finite inference budgets required")
        dataset = load_dataset(args.dataset_dir)
        selected = select_questions(dataset.questions, args.cases, args.limit)
        extra = native_profile(args.extra_body_json)
        provenance = json.loads(args.manifest.read_bytes(), object_pairs_hook=unique_object) if args.manifest else None
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("manifest must be a JSON object")
        json.dumps(provenance, allow_nan=False)
    except (ValueError, OSError, RecursionError) as exc:
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
    set_request_body_extra(extra)
    nonce = args.output.stem
    client = ChatClient(args.base_url, api_key, args.measurement_timeout, artifacts_dir=args.artifacts_dir, label=args.label, nonce=nonce)
    manifest_path = args.output.with_suffix(".manifest.json")
    write_json(manifest_path, {
        "suite_version": VERSION, "dataset_sha256": dataset.hashes,
        "dataset_metadata": dataset.metadata,
        "capture_note": "Full 12032-row source capture retained at /mnt/models/inference-evals/vane-model-selection-20261107/source-v2/mmlu-pro-captured-rows.jsonl; SHA256 808a00891b968caeec82d52bb2a4156f1a6ec33df918f7d678aa97187759e67a. This durable path supersedes metadata scratch-path notes; not re-read by this runner.",
        "selected_case_ids": [question.case_id for question in selected],
        "gold_for_scoring_only": [dataset.gold[question.case_id] for question in selected],
        "prompt_instruction": INSTRUCTION, "final_answer_parser": PARSER_GRAMMAR,
        "max_tokens": args.max_tokens, "measurement_timeout_seconds": args.measurement_timeout,
        "native_profile": extra, "provenance": provenance,
        "provenance_status": "operator supplied, not independently verified" if provenance is not None else "not supplied; runtime identities unknown",
        "limitations": LIMITATIONS,
    }, api_key=api_key)
    started = timestamp_pair()
    rows: list[CaseRow] = []

    def checkpoint() -> None:
        write_json(args.output, {
            "suite_version": VERSION, "label": args.label, "model": args.model, "nonce": nonce,
            "started": started, "updated": timestamp_pair(), "dataset_sha256": dataset.hashes,
            "scoring_manifest": str(manifest_path), "native_profile": extra,
            "max_tokens": args.max_tokens, "measurement_timeout_seconds": args.measurement_timeout,
            "selected_case_ids": [question.case_id for question in selected],
            "completed": len(rows), "selected_total": len(selected), "complete": len(rows) == len(selected),
            "categories": summarize(rows, selected), "cases": rows, "limitations": LIMITATIONS,
        }, api_key=api_key)

    checkpoint()
    for question in selected:
        rows.append(run_case(question, dataset.gold[question.case_id], client, args.model, args.max_tokens))
        checkpoint()
    print(json.dumps({"categories": summarize(rows, selected), "report": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
