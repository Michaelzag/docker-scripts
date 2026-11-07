#!/usr/bin/env python3
"""Native System One fixtures + model profiles -> retained option distributions.

Use case: state/questions/gold enter, per-variant scores and evidence leave; every
selected item remains in the denominator, gold never enters a prompt. Side effects
are HTTP and local files. Actual variation: templates, native profiles, option
order, endpoint, concurrency and budgets. This is a local adapter benchmark, NOT
an implementation of /v1/systemone or a measurement of Jev's retry policy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from string import Template
from typing import cast

from agent_workload_benchmark import ApiResult, ChatClient, timestamp_pair
from intelligence_challenge_suite import completion, reserve_output, write_json
from published_knowledge_benchmark import native_profile, unique_object

VERSION = "system-one-option-distribution-v1"
DEFAULT_TEMPLATE = Path(__file__).with_name("system_one_prompt.txt")
LIMITATIONS = (
    "Local model adapter, not the production Jev endpoint. Synthetic fixtures are "
    "smoke tests, NOT ZagDB gold or evidence of model quality. All selected "
    "item-variants are accuracy/validity/coverage denominators, including failures. "
    "Distribution statistics require full coverage; their sample counts are explicit. "
    "Thinking scores final-key accuracy only, never invented distributions. "
    "Pre-constraint forced-token logprobs, not post-sampling probabilities. "
    "Sequence scores are exact for the server tokenizer's canonical key encoding. "
    "No retries, no model judge, no streaming. HTTP timeout is per I/O, not an "
    "absolute deadline. Nonstream latency is not TTFT. >10s is diagnostic, not "
    "a simulated provider timeout; no 1MiB production response cap is simulated."
    " Native score scalar = expected zero-based original legend index; ZagDB "
    "does not prescribe expectation versus argmax. Probability and scalar metrics "
    "are reported separately; thresholds are diagnostics, not fixture gold. "
    "Native response is null if actual usage is unavailable; no fake zero usage."
)


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected JSON object with string keys")
    return cast(dict[str, object], value)


def array_value(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("expected JSON array")
    return cast(list[object], value)


def text_value(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected nonempty string")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class Option:
    id: str
    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class Item:
    id: str
    family: str
    difficulty: str
    state: object
    question: dict[str, object]
    options: tuple[Option, ...]
    gold: frozenset[str]
    defer_acceptable: bool
    special: dict[str, str]


@dataclasses.dataclass(frozen=True, slots=True)
class Variant:
    id: str
    key_to_option: dict[str, str]
    messages: list[dict[str, str]]


@dataclasses.dataclass(frozen=True, slots=True)
class Settings:
    model: str
    direct_template_kwargs: dict[str, object]
    direct_prefill: str
    thinking_profile: dict[str, object]
    max_tokens: int


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    item_id: str
    family: str
    variant_id: str
    mode: str
    key_to_option: dict[str, str]
    option_id: str | None
    correct: bool
    full_coverage: bool
    probabilities: dict[str, float]
    sequence_logprobs: dict[str, float]
    first_token_logprobs: dict[str, float]
    leaked_mass: float | None
    entropy: float | None
    margin: float | None
    normalized_pmax: float | None
    native_answer: dict[str, object] | None
    native_response: dict[str, object] | None
    scalar_metrics: dict[str, object]
    elapsed_seconds: float
    error: str | None
    calls: tuple[ApiResult, ...]


def parse_item(value: object) -> Item:
    row = object_value(value)
    question = object_value(row.get("question"))
    if question.get("type") not in ("noul", "choice", "score"):
        raise ValueError("question.type must be noul, choice or score")
    options_list = []
    for entry in array_value(question.get("options")):
        record = object_value(entry)
        options_list.append(Option(text_value(record.get("id")), text_value(record.get("text"))))
    options = tuple(options_list)
    option_ids = {option.id for option in options}
    if not 1 <= len(options) <= 26 or len(option_ids) != len(options):
        raise ValueError("require 1..26 unique options")
    if question["type"] == "noul" and option_ids != {"true", "false"}:
        raise ValueError("noul requires exactly true/false option ids")
    text_value(question.get("id"))
    text_value(question.get("text"))
    state = row.get("state")
    if not isinstance(state, (str, dict, list)):
        raise TypeError("state must be Structured (string, object or array)")
    state_bytes = state.encode() if isinstance(state, str) else json.dumps(state, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    if len(state_bytes) > 32768:
        raise ValueError("state must be at most 32768 UTF-8 bytes")
    gold_row = object_value(row.get("gold"))
    gold = frozenset(text_value(key) for key in array_value(gold_row.get("option_ids")))
    defer = gold_row.get("defer_acceptable")
    special = {key: text_value(value) for key, value in object_value(row.get("special", {})).items() if value is not None}
    if set(special) - {"new", "unknown", "defer"} or not set(special.values()) <= option_ids:
        raise ValueError("invalid special option markers")
    if not gold <= option_ids or type(defer) is not bool or (not gold and not defer):
        raise ValueError("gold must name available options or permit defer")
    if not (gold - {special.get("defer")}) and not (defer and "defer" in special):
        raise ValueError("gold must contain an effective correct option")
    return Item(text_value(row.get("id")), text_value(row.get("family")),
                text_value(row.get("difficulty")), state, question, options, gold, defer, special)


def native_request(item: Item, model: str) -> dict[str, object]:
    kind = item.question["type"]
    criteria: object = {option.id: option.text for option in item.options}
    if kind == "score":
        criteria = [option.text for option in item.options]
    return {"model": model, "state": item.state, "questions": {
        text_value(item.question["id"]): {"type": kind, "instructions": item.question["text"], "criteria": criteria},
    }}


def native_answer(item: Item, probabilities: dict[str, float], confidence: float) -> dict[str, object]:
    kind = item.question["type"]
    if kind == "noul":
        return {"type": kind, "noul": probabilities["true"]}
    if kind == "score":
        indexed = {str(index): probabilities[option.id] for index, option in enumerate(item.options)}
        return {"type": kind, "score": sum(int(index) * p for index, p in indexed.items()),
                "legend": {str(index): option.text for index, option in enumerate(item.options)},
                "probabilities": indexed, "confidence": confidence}
    return {"type": kind, "choice": max(probabilities, key=lambda key: probabilities[key]),
            "probabilities": probabilities, "confidence": confidence}


def scalar_metrics(item: Item, probabilities: dict[str, float]) -> dict[str, object]:
    if not probabilities:
        return {}
    if item.question["type"] == "noul":
        probability = probabilities["true"]
        thresholds = {}
        for minimum in (0.6, 0.7, 0.8):
            if 1 - minimum < probability < minimum:
                outcome = "ambiguous"
            else:
                chosen = "true" if probability >= minimum else "false"
                outcome = "correct" if is_correct(item, chosen) else "wrong"
            thresholds[str(minimum)] = outcome
        return {"noul": probability, "threshold_outcomes": thresholds}
    if item.question["type"] != "score":
        return {}
    expected = sum(index * probabilities[option.id] for index, option in enumerate(item.options))
    argmax = max(range(len(item.options)), key=lambda index: probabilities[item.options[index].id])
    gold_indices = [index for index, option in enumerate(item.options) if is_correct(item, option.id)]
    return {"expected_ordinal": expected, "argmax_ordinal": argmax, "gold_indices": gold_indices,
            "expected_absolute_error": min(abs(expected - index) for index in gold_indices),
            "argmax_absolute_error": min(abs(argmax - index) for index in gold_indices),
            "argmax_exact_level_match": argmax in gold_indices}


def summarize_scalars(rows: list[Outcome]) -> dict[str, object]:
    score = [row.scalar_metrics for row in rows if "expected_ordinal" in row.scalar_metrics]
    noul = [row.scalar_metrics for row in rows if "noul" in row.scalar_metrics]
    return {
        "score_convention": "expected original zero-based legend ordinal: sum(i*p_i)",
        "score_n": len(score),
        "score_expected_absolute_error_mean": mean([cast(float, value["expected_absolute_error"]) for value in score]),
        "score_argmax_absolute_error_mean": mean([float(cast(int, value["argmax_absolute_error"])) for value in score]),
        "score_argmax_exact_level_matches": sum(value["argmax_exact_level_match"] is True for value in score),
        "noul_n": len(noul), "noul_probability_mean": mean([cast(float, value["noul"]) for value in noul]),
        "noul_thresholds": {str(minimum): {
            outcome: sum(object_value(value["threshold_outcomes"])[str(minimum)] == outcome for value in noul)
            for outcome in ("correct", "wrong", "ambiguous")
        } for minimum in (0.6, 0.7, 0.8)},
    }


def native_response(item: Item, model: str, probabilities: dict[str, float], confidence: float,
                    calls: list[ApiResult]) -> dict[str, object] | None:
    input_tokens = output_tokens = 0
    for call in calls:
        response = call.response or {}
        if "completion_probabilities" not in response:
            continue
        raw_usage, raw_timings = response.get("usage", {}), response.get("timings", {})
        if not isinstance(raw_usage, dict) or not isinstance(raw_timings, dict):
            return None
        usage, timings = object_value(raw_usage), object_value(raw_timings)
        inputs = usage.get("prompt_tokens", response.get("tokens_evaluated", timings.get("prompt_n")))
        outputs = usage.get("completion_tokens", response.get("tokens_predicted", timings.get("predicted_n")))
        if type(inputs) is not int or type(outputs) is not int or min(inputs, outputs) < 0:
            return None
        input_tokens += inputs
        output_tokens += outputs
    return {"model": model, "answers": {text_value(item.question["id"]): native_answer(item, probabilities, confidence)},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


def load_fixtures(paths: list[Path], model: str) -> tuple[tuple[Item, ...], dict[str, str]]:
    items = []
    hashes = {}
    for path in paths:
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode().splitlines():
            if line.strip():
                item = parse_item(json.loads(line, object_pairs_hook=unique_object))
                request = native_request(item, model)
                # Match compact native wire bytes, not character count.
                if len(json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode()) > 61440:
                    raise ValueError(f"{item.id}: native request exceeds 61440 bytes")
                items.append(item)
    if not items or len({item.id for item in items}) != len(items):
        raise ValueError("fixtures must contain nonempty, globally unique item ids")
    return tuple(items), hashes


def make_variants(item: Item, template: str, count: int, seed: int, system_rule: str) -> tuple[Variant, ...]:
    if count < 2 or count > max(2, math.factorial(len(item.options))):
        raise ValueError(f"{item.id}: require 2..number of distinct orders (two identical orders for K=1)")
    orders = [item.options]
    rng = random.Random(f"{seed}:{item.id}")
    # Finite seeded traversal avoids rejection loops near permutation exhaustion.
    shuffled = list(item.options)
    rng.shuffle(shuffled)
    for order in itertools.permutations(shuffled):
        if order != item.options:
            orders.append(order)
        if len(orders) == count:
            break
    if len(item.options) == 1:
        orders = [item.options] * count
    variants = []
    for index, order in enumerate(orders):
        mapping = {chr(65 + i): option.id for i, option in enumerate(order)}
        prompt = Template(template).substitute(
            state=item.state if isinstance(item.state, str) else json.dumps(item.state, separators=(",", ":"), ensure_ascii=False),
            question=item.question["text"],
            options="\n".join(f"{chr(65 + i)}. {option.text}" for i, option in enumerate(order)),
        )
        messages = ([{"role": "system", "content": system_rule}] if system_rule else [])
        messages.append({"role": "user", "content": prompt})
        variants.append(Variant(f"order-{index}", mapping, messages))
    return tuple(variants)


def is_correct(item: Item, option_id: str | None) -> bool:
    if option_id is None:
        return False
    if option_id == item.special.get("defer"):
        return item.defer_acceptable
    return option_id in item.gold


def parse_final_key(content: str, keys: dict[str, str]) -> str | None:
    stripped = content.strip()
    return stripped if stripped in keys else None


def logprob(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value > 0:
        raise ValueError("missing/nonfinite/positive pre-constraint token logprob")
    return float(value)


def forced_score(response: dict[str, object], key: str, token_count: int) -> tuple[float, str, float]:
    """Own token/logprob is required, even if the key is absent from top_logprobs."""
    if response.get("content") != key:
        raise ValueError("forced generation did not return exactly the requested key")
    records = [object_value(entry) for entry in array_value(response.get("completion_probabilities"))]
    if len(records) != token_count:
        raise ValueError("forced token probability coverage is incomplete")
    tokens = [text_value(entry.get("token")) for entry in records]
    if "".join(tokens) != key:
        raise ValueError("forced tokens do not reconstruct key (no EOS or extra tokens allowed)")
    scores = [logprob(entry.get("logprob")) for entry in records]
    for entry, token, score in zip(records, tokens, scores):
        for candidate in array_value(entry.get("top_logprobs", [])):
            top = object_value(candidate)
            if top.get("token") == token and not math.isclose(logprob(top.get("logprob")), score, abs_tol=1e-5):
                raise ValueError("own token logprob disagrees with pre-constraint top_logprobs")
    return sum(scores), tokens[0], scores[0]


def normalize(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {key: math.exp(value - maximum) for key, value in scores.items()}
    total = sum(weights.values())
    return {key: weight / total for key, weight in weights.items()}


def run_variant(item: Item, variant: Variant, mode: str, client: ChatClient, settings: Settings) -> Outcome:
    started = time.monotonic()
    calls: list[ApiResult] = []
    scores: dict[str, float] = {}
    first: dict[str, float] = {}
    probabilities: dict[str, float] = {}
    option_id = None
    leaked = entropy = margin = normalized = None
    error = None
    full_coverage = False
    case_id = f"{item.id}/{variant.id}/{mode}"

    def post(path: str, body: dict[str, object]) -> dict[str, object]:
        result = client.post(path, body, origin=True, case_id=case_id)
        calls.append(result)
        if not result.ok or result.response is None:
            raise ValueError(result.error or "HTTP request failed")
        return object_value(result.response)

    try:
        if mode == "thinking":
            body: dict[str, object] = {"model": settings.model, "messages": variant.messages,
                                      "max_tokens": settings.max_tokens, "temperature": 0,
                                      **settings.thinking_profile}
            result = client.post("v1/chat/completions", body, origin=True, case_id=case_id)
            calls.append(result)
            answer = completion(result)
            if answer.failure or answer.finish_reason != "stop" or answer.message.get("tool_calls"):
                raise ValueError(answer.failure or "unexpected tools or finish reason")
            key = parse_final_key(answer.text, variant.key_to_option)
            if key is None:
                raise ValueError("final content must be exactly one key, without explanation or alternatives")
            option_id = variant.key_to_option[key]
        else:
            rendered = post("apply-template", {"model": settings.model, "messages": variant.messages,
                                              "chat_template_kwargs": settings.direct_template_kwargs})
            prompt = text_value(rendered.get("prompt")) + settings.direct_prefill
            first_by_token: dict[str, float] = {}
            for key in variant.key_to_option:
                tokenized = post("tokenize", {"content": key, "add_special": False, "parse_special": False})
                tokens = array_value(tokenized.get("tokens"))
                if not tokens or any(type(token) is not int for token in tokens):
                    raise ValueError("invalid key tokenizer response")
                response = post("completion", {
                    "model": settings.model, "prompt": prompt, "n_predict": len(tokens),
                    "temperature": 0, "n_probs": max(20, len(variant.key_to_option)),
                    "grammar": "root ::= " + json.dumps(key), "cache_prompt": True,
                    "post_sampling_probs": False,
                })
                score, first_token, first_score = forced_score(response, key, len(tokens))
                scores[key] = score
                first[key] = first_score
                if first_token in first_by_token and not math.isclose(first_by_token[first_token], first_score, abs_tol=1e-5):
                    raise ValueError("inconsistent first-token scores across forced keys")
                first_by_token[first_token] = first_score
            raw_mass = sum(math.exp(value) for value in first_by_token.values())
            if raw_mass > 1 + 1e-5:
                raise ValueError("pre-constraint first-token key mass exceeds one")
            leaked = max(0.0, 1 - raw_mass)
            by_key = normalize(scores)
            probabilities = {variant.key_to_option[key]: value for key, value in by_key.items()}
            option_id = max(probabilities, key=lambda option: probabilities[option])
            ordered = sorted(probabilities.values(), reverse=True)
            entropy = -sum(p * math.log(p) for p in ordered if p)
            margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
            normalized = (len(ordered) * ordered[0] - 1) / (len(ordered) - 1) if len(ordered) > 1 else 1.0
            full_coverage = True
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        error = str(exc)
    return Outcome(item.id, item.family, variant.id, mode, variant.key_to_option,
                   option_id, is_correct(item, option_id), full_coverage, probabilities,
                   scores, first, leaked, entropy, margin, normalized,
                   native_answer(item, probabilities, normalized) if full_coverage and normalized is not None else None,
                   native_response(item, settings.model, probabilities, normalized, calls) if full_coverage and normalized is not None else None,
                   scalar_metrics(item, probabilities),
                   time.monotonic() - started, error, tuple(calls))


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def js_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = left.keys() | right.keys()
    midpoint = {key: (left.get(key, 0) + right.get(key, 0)) / 2 for key in keys}
    divergence = sum(p * math.log2(p / midpoint[key]) / 2
                     for distribution in (left, right) for key, p in distribution.items() if p)
    return math.sqrt(max(0.0, divergence))


def auroc(rows: list[Outcome]) -> float | None:
    correct = [row.margin for row in rows if row.correct and row.margin is not None]
    wrong = [row.margin for row in rows if not row.correct and row.margin is not None]
    if not correct or not wrong:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in correct for b in wrong) / (len(correct) * len(wrong))


def summarize(rows: list[Outcome], selected: tuple[Item, ...], variants: int, mode: str, wall: float) -> dict[str, object]:
    groups: dict[str, object] = {}
    for family in ("all", *sorted({item.family for item in selected})):
        items = [item for item in selected if family == "all" or item.family == family]
        group = [row for row in rows if row.mode == mode and (family == "all" or row.family == family)]
        n = len(items) * variants
        distributions = [row for row in group if row.full_coverage]
        confidences = [(max(row.probabilities.values()), row.correct) for row in distributions]
        latencies = sorted(row.elapsed_seconds for row in group)
        leaks = [row.leaked_mass for row in distributions if row.leaked_mass is not None]
        confusion = {}
        for special in ("new", "unknown", "defer"):
            counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for item in items:
                marker = item.special.get(special)
                if marker is None:
                    continue
                expected = item.defer_acceptable if special == "defer" else marker in item.gold
                item_rows = {row.variant_id: row for row in group if row.item_id == item.id}
                for index in range(variants):
                    row = item_rows.get(f"order-{index}")
                    predicted = row is not None and row.option_id == marker
                    counts[("t" if predicted == expected else "f") + ("p" if predicted else "n")] += 1
            confusion[special] = counts
        agreements = 0
        distances = []
        pairs = len(items) * math.comb(variants, 2)
        for item in items:
            item_rows = [row for row in group if row.item_id == item.id]
            for left, right in itertools.combinations(item_rows, 2):
                agreements += left.option_id is not None and left.option_id == right.option_id
                if left.full_coverage and right.full_coverage:
                    distances.append(js_distance(left.probabilities, right.probabilities))
        calls = [call for row in group for call in row.calls]
        groups[family] = {
            "n": n, "n_items": len(items), "attempted": len(group), "unattempted": n - len(group),
            "valid_key_rate": sum(row.option_id is not None for row in group) / n,
            "accuracy": sum(row.correct for row in group) / n,
            "full_coverage_rate": len(distributions) / n if mode == "direct" else None,
            "distribution_n": len(distributions), "special_confusion": confusion,
            "native_response_n": sum(row.native_response is not None for row in group),
            "scalars": summarize_scalars(group),
            "entropy_correct": mean([row.entropy for row in distributions if row.correct and row.entropy is not None]),
            "entropy_wrong": mean([row.entropy for row in distributions if not row.correct and row.entropy is not None]),
            "margin_correct": mean([row.margin for row in distributions if row.correct and row.margin is not None]),
            "margin_wrong": mean([row.margin for row in distributions if not row.correct and row.margin is not None]),
            "margin_correctness_auroc": auroc(distributions),
            "argmax_confidence_brier": mean([(p - int(correct)) ** 2 for p, correct in confidences]),
            "argmax_confidence_log_loss": mean([-math.log(max(1e-15, p if correct else 1 - p)) for p, correct in confidences]),
            "mean_leaked_mass": mean(leaks), "max_leaked_mass": max(leaks) if leaks else None,
            "order_stability": {"pairs": pairs, "argmax_agreement": agreements / pairs if pairs else None,
                                "js_distance_mean": mean(distances), "distribution_pairs": len(distances)},
            "latency_p50_seconds": statistics.median(latencies) if latencies else None,
            "latency_p95_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
            "over_10s_item_variants": sum(value > 10 for value in latencies),
            "over_10s_requests": sum(call.wall_seconds > 10 for call in calls),
            "http_errors": sum(not call.ok for call in calls),
            "timeout_requests": sum(not call.ok and any(word in (call.error or "").lower() for word in ("timeout", "timed out")) for call in calls),
            "wall_seconds": wall, "item_variants_per_second": len(group) / wall if wall else None,
            "items_per_second": len(group) / variants / wall if wall else None,
        }
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, nargs="+", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("direct", "thinking", "both"), default="both")
    parser.add_argument("--direct-template-kwargs", default="{}")
    parser.add_argument("--direct-prefill", default="")
    parser.add_argument("--thinking-extra-body", default="{}")
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--system-rule", default="")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--measurement-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.concurrency <= 0 or args.max_tokens <= 0 or not math.isfinite(args.measurement_timeout) or args.measurement_timeout <= 0:
            raise ValueError("positive finite concurrency and inference budgets required")
        items, hashes = load_fixtures(args.fixtures, args.model)
        template = args.prompt_template.read_text()
        if set(Template(template).get_identifiers()) != {"state", "question", "options"}:
            raise ValueError("prompt template must contain exactly $state, $question and $options placeholders")
        template_kwargs = object_value(json.loads(args.direct_template_kwargs, object_pairs_hook=unique_object))
        profile = native_profile(args.thinking_extra_body)
        provenance = object_value(json.loads(args.manifest.read_bytes(), object_pairs_hook=unique_object)) if args.manifest else None
        json.dumps([template_kwargs, provenance], allow_nan=False)
        settings = Settings(args.model, template_kwargs, args.direct_prefill, profile, args.max_tokens)
        prepared = [(item, variant) for item in items for variant in make_variants(item, template, args.variants, args.seed, args.system_rule)]
    except (ValueError, TypeError, OSError, KeyError, RecursionError) as exc:
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
    client = ChatClient(args.base_url, api_key, args.measurement_timeout,
                        artifacts_dir=args.artifacts_dir, label=args.label, nonce=args.output.stem)
    manifest_path = args.output.with_suffix(".manifest.json")
    write_json(manifest_path, {
        "suite_version": VERSION, "dataset_sha256": hashes, "fixtures_for_scoring_only": [dataclasses.asdict(item) | {"gold": sorted(item.gold)} for item in items],
        "prompt_template": template, "prompt_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "system_rule": args.system_rule, "seed": args.seed, "variants": args.variants,
        "settings": dataclasses.asdict(settings), "prepared": [dataclasses.asdict(variant) | {"item_id": item.id} for item, variant in prepared],
        "native_requests": [native_request(item, args.model) for item in items],
        "provenance": provenance, "provenance_status": "operator supplied, not independently verified" if provenance else "not supplied; runtime identities unknown",
        "limitations": LIMITATIONS,
    }, api_key=api_key)
    modes = ("direct", "thinking") if args.mode == "both" else (args.mode,)
    rows: list[Outcome] = []
    walls: dict[str, float] = {}
    started_at = timestamp_pair()

    def checkpoint() -> None:
        write_json(args.output, {
            "suite_version": VERSION, "label": args.label, "model": args.model,
            "started": started_at, "updated": timestamp_pair(), "complete": len(rows) == len(prepared) * len(modes),
            "selected_items": len(items), "selected_item_variants_per_mode": len(prepared),
            "concurrency": args.concurrency, "measurement_timeout_seconds": args.measurement_timeout,
            "scoring_manifest": str(manifest_path), "dataset_sha256": hashes,
            "modes": {mode: summarize(rows, items, args.variants, mode, walls.get(mode, 0)) for mode in modes},
            "cases": [dataclasses.asdict(row) for row in rows], "limitations": LIMITATIONS,
        }, api_key=api_key)

    checkpoint()
    for mode in modes:
        began = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(run_variant, item, variant, mode, client, settings) for item, variant in prepared]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                walls[mode] = time.monotonic() - began
                checkpoint()
        walls[mode] = time.monotonic() - began
        checkpoint()
    print(json.dumps({"report": str(args.output), "modes": {mode: summarize(rows, items, args.variants, mode, walls[mode]) for mode in modes}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
