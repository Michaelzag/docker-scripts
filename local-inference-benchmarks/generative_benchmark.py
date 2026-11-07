#!/usr/bin/env python3
"""Evaluate exact ZagDB generative fixtures with one deterministic call per row.

Inputs are immutable fixtures and templates; outputs are retained HTTP evidence and
rule-by-rule scores. No answer repair, retries, or model-based judging is used.
Hub summary prompts are proposals for evaluation, not production prompts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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

from agent_workload_benchmark import ChatClient, redact_persistence
from intelligence_challenge_suite import reserve_output, write_json
from published_knowledge_benchmark import unique_object


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def load_rows(fixtures: Path, templates_path: Path) -> list[dict]:
    templates = json.loads(templates_path.read_text(), object_pairs_hook=unique_object)
    rows = []
    for family in ("ontology_naming", "hub_summary"):
        template = templates[family]
        if digest(template["system"]) != template["system_sha256"]:
            raise ValueError(f"{family} system template hash mismatch")
        for line in (fixtures / f"{family}.jsonl").read_text().splitlines():
            row = json.loads(line, object_pairs_hook=unique_object)
            inputs = row["input"]
            user = template["template"]
            for variable in template["variables"]:
                value = inputs[variable]
                rendered = value if family == "ontology_naming" and variable == "kind" else compact(value)
                user = user.replace("{{ " + variable + " }}", rendered)
            messages = [{"role": "system", "content": template["system"]}, {"role": "user", "content": user}]
            if family == "ontology_naming":
                messages.append({"role": "user", "content": compact(inputs["source_parts"])})
            hashes = row["checks"]["render_sha256"]
            for key, message in zip(("system", "user_turn_1", "user_turn_2"), messages):
                if digest(message["content"]) != hashes[key]:
                    raise ValueError(f"{row['id']} {key} render hash mismatch")
            rows.append({"id": row["id"], "family": family, "rules": row["checks"], "messages": messages})
    return rows


def matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def groups_match(groups: list, text: str) -> bool:
    return all(any(matches(pattern, text) for pattern in group) for group in groups)


def normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower()).removesuffix("s")


def score_naming(answer: dict, rules: dict) -> dict[str, bool]:
    name, definition = answer["name"], answer["definition"]
    nc, dc = rules["required_name_constraints"], rules["definition_constraints"]
    checks = {
        "name_max_chars": len(name) <= nc["max_chars"],
        "name_max_words": len(name.split()) <= nc["max_words"],
        "name_required_pattern": not nc["must_match_any"] or any(matches(p, name) for p in nc["must_match_any"]),
        "name_forbidden_patterns": not any(matches(p, name) for p in nc["must_not_match"]),
        "definition_max_chars": len(definition) <= dc["max_chars"],
    }
    for reference in rules["must_distinguish_from"]:
        existing = rules["existing_reference"][reference]
        checks[f"distinguish:{reference}"] = all(normalize(name) != normalize(x) for x in [existing["name"], *existing["aliases"]])
    for i, fact in enumerate(rules["required_definition_facts"]):
        checks[f"definition_fact:{i}:{fact['fact']}"] = any(matches(p, definition) for p in fact["any_of"])
    for i, rule in enumerate(rules["forbidden"]):
        checks[f"forbidden:{i}:{rule['claim']}"] = not matches(rule["pattern"], name + "\n" + definition)
    return checks


def score_summary(answer: dict, rules: dict) -> dict[str, bool]:
    sections = answer["sections"]
    members = set(rules["member_nodes"])
    cited_members = {x for section in sections for x in section["members"]}
    cited_assessments = {x for section in sections for x in section["assessments"]}
    allowed = rules["citeable_assessments"]
    checks = {
        "member_coverage": cited_members == members,
        "citation_order": all(s["members"] == sorted(set(s["members"])) and s["assessments"] == sorted(set(s["assessments"])) for s in sections),
        "citation_grounding": all(set(s["members"]) <= members and set(s["assessments"]) <= {a for m in s["members"] for a in allowed.get(str(m), [])} for s in sections),
        "forbidden_member_citations": not bool(cited_members & set(rules["forbidden_citations"]["members"])),
        "forbidden_assessment_citations": not bool(cited_assessments & set(rules["forbidden_citations"]["assessments"])),
    }
    for revision in rules["required_facts"]:
        checks[f"required_fact:{revision}"] = any(revision in s["assessments"] and groups_match(rules["required_fact_patterns"][str(revision)], s["text"]) for s in sections)
    for revision in rules["uncertainty_must_survive"]:
        citing = [s for s in sections if revision in s["assessments"]]
        patterns = rules["uncertainty_patterns"][str(revision)]
        checks[f"uncertainty:{revision}"] = bool(citing) and (all(matches(rules["hedge_pattern"], s["text"]) for s in citing) or any(groups_match(patterns, text) for text in answer["uncertainty"] + answer["conflicts"]))
    for a, b in rules["contested_pairs"]:
        checks[f"contested_pair:{a}:{b}"] = any(groups_match(rules["uncertainty_patterns"][str(a)], text) and groups_match(rules["uncertainty_patterns"][str(b)], text) for text in answer["conflicts"])
    text = "\n".join([s["text"] for s in sections] + answer["uncertainty"] + answer["conflicts"])
    for i, rule in enumerate(rules["forbidden_claims"]):
        checks[f"forbidden:{i}:{rule['claim']}"] = not matches(rule["pattern"], text)
    return checks


def valid_shape(answer: object, family: str) -> bool:
    if not isinstance(answer, dict):
        return False
    if family == "ontology_naming":
        return set(answer) == {"name", "definition"} and all(isinstance(answer[k], str) and answer[k].strip() for k in answer)
    if set(answer) != {"sections", "uncertainty", "conflicts"}:
        return False
    if not all(isinstance(answer[k], list) for k in answer) or not answer["sections"]:
        return False
    if not all(isinstance(x, str) and x.strip() for k in ("uncertainty", "conflicts") for x in answer[k]):
        return False
    for section in answer["sections"]:
        if not isinstance(section, dict) or set(section) != {"text", "members", "assessments"}:
            return False
        if not isinstance(section["text"], str) or not section["text"].strip():
            return False
        if not all(isinstance(section[k], list) and all(type(x) is int and x >= 0 for x in section[k]) for k in ("members", "assessments")) or not section["members"]:
            return False
    return True


def run_row(row: dict, client: ChatClient, model: str, extra: dict, max_tokens: int) -> dict:
    body = {"model": model, "messages": row["messages"], "temperature": 0, "seed": 0, "max_tokens": max_tokens, **extra}
    result = client.chat(body, case_id=row["id"])
    record = {"id": row["id"], "family": row["family"], "status": result.status, "error": result.error, "artifact": result.artifact, "latency_seconds": result.wall_seconds, "shape_failure": False, "length": False, "checks": {}}
    choices = (result.response or {}).get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
    record["length"] = choice.get("finish_reason") == "length"
    if row["id"] == "hs-13":
        error = {}
        if result.error and result.error.startswith(f"HTTP {result.status}: "):
            try:
                error = json.loads(result.error.split(": ", 1)[1]).get("error", {})
            except (ValueError, AttributeError):
                pass
        record["error_type"] = error.get("type") if isinstance(error, dict) else None
        record["error_message"] = error.get("message") if isinstance(error, dict) else None
        record["checks"] = {"typed_oversize_refusal": result.status is not None and result.status >= 400 and isinstance(error, dict) and (error.get("type") == "context_length_exceeded" or error.get("code") == "context_length_exceeded")}
    elif result.ok:
        record["length"] = choice.get("finish_reason") == "length"
        message = choice.get("message")
        if not isinstance(message, dict):
            message = {}
        try:
            if record["length"] or choice.get("finish_reason") != "stop" or message.get("tool_calls") or not isinstance(message.get("content"), str):
                raise ValueError("invalid completion")
            answer = json.loads(message["content"], object_pairs_hook=unique_object)
            if not valid_shape(answer, row["family"]):
                raise ValueError("invalid response shape")
        except (ValueError, RecursionError):
            record["shape_failure"] = not record["length"]
            record["checks"] = {"shape": False}
        else:
            record["checks"] = {"shape": True, **(score_naming(answer, row["rules"]) if row["family"] == "ontology_naming" else score_summary(answer, row["rules"]))}
    else:
        record["checks"] = {"request": False}
    record["passed"] = all(record["checks"].values())
    return record


def summarize(rows: list[dict], wall: float) -> dict:
    result = {}
    for family in ("ontology_naming", "hub_summary"):
        selected = [r for r in rows if r["family"] == family]
        latencies = sorted(r["latency_seconds"] for r in selected)
        counts: Counter = Counter()
        for row in selected:
            for key, passed in row["checks"].items():
                counts[key] += int(passed)
        result[family] = {"rows": len(selected), "passed": sum(r["passed"] for r in selected), "pass_rate": sum(r["passed"] for r in selected) / len(selected) if selected else None, "per_check_pass_counts": dict(counts), "shape_failures": sum(r["shape_failure"] for r in selected), "length_count": sum(r["length"] for r in selected), "latency_p50": statistics.median(latencies) if latencies else None, "latency_p95": latencies[math.ceil(len(latencies) * .95) - 1] if latencies else None, "wall_seconds": wall}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--extra-body", default="{}")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--measurement-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    args = parser.parse_args()
    try:
        rows = load_rows(args.fixtures, args.templates)
        extra = json.loads(args.extra_body, object_pairs_hook=unique_object)
        if not isinstance(extra, dict) or set(extra) - {"chat_template_kwargs"} or ("chat_template_kwargs" in extra and not isinstance(extra["chat_template_kwargs"], dict)):
            raise ValueError("extra-body permits only chat_template_kwargs")
        if args.max_tokens <= 0 or args.concurrency <= 0 or not math.isfinite(args.measurement_timeout) or args.measurement_timeout <= 0:
            raise ValueError("budgets must be positive and finite")
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise ValueError(f"{args.api_key_env} is unset")
        reserve_output(args.output)
        client = ChatClient(args.base_url, api_key, args.measurement_timeout, artifacts_dir=args.artifacts_dir, label=args.label, nonce=args.output.stem)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(lambda row: run_row(row, client, args.model, extra, args.max_tokens), rows))
        wall = time.monotonic() - started
        report = {"version": "zagdb-generative-v1", "label": args.label, "model": args.model, "hub_summary_prompt_status": "UNVERIFIED_PROPOSED_EVAL_ONLY", "summary": summarize(results, wall), "rows": results}
        write_json(args.output, redact_persistence(report, api_key), api_key=api_key)
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
