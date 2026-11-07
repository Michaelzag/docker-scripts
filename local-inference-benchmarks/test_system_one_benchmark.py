"""Offline System One contract tests; only a loopback fake HTTP server is used."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import io
import itertools
import json
import math
import tempfile
import threading
import time
import unittest
from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from unittest.mock import patch

import system_one_benchmark as bench
from agent_workload_benchmark import ChatClient

DATASET = Path(__file__).parent / "datasets/system-one-synthetic-smoke/smoke.jsonl"
Reply = Callable[[str, dict[str, object]], tuple[int, dict[str, object]]]


def fixture(kind: str = "choice") -> bench.Item:
    ids = ("true", "false") if kind == "noul" else ("first", "second")
    return bench.parse_item({
        "id": "synthetic-test", "family": "synthetic", "difficulty": "smoke", "state": "SYNTHETIC STATE",
        "question": {"id": "q1", "type": kind, "text": "Pick the first semantic option.",
                     "options": [{"id": key, "text": key} for key in ids]},
        "gold": {"option_ids": [ids[0]], "defer_acceptable": False},
        "special": {"new": None, "unknown": None, "defer": None},
        "rationale": "GOLD_RATIONALE_NEVER_SEND",
    })


class FakeModel:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.omit_logprob = False
        self.mismatch_top = False
        self.fail_path = ""
        self.delay_path = ""
        self.thinking_text = " A \n"
        self.thinking_finish = "stop"
        self.multi = False
        self.missing_primary_key: str | None = None
        self.first_probability = 0.6
        self.other_probability = 0.2

    def reply(self, path: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
        self.requests.append((path, body))
        if path == self.delay_path:
            time.sleep(0.1)
        if path == self.fail_path:
            return 503, {"error": "synthetic unavailable"}
        if path == "/v1/systemone":
            question_id, raw_question = next(iter(bench.object_value(body["questions"]).items()))
            question = bench.object_value(raw_question)
            kind = question["type"]
            if kind == "noul":
                answer: dict[str, object] = {"type": kind, "noul": 0.75}
            else:
                criteria = question["criteria"]
                weights = ({str(index): 3.0 if text == "first" else 1.0
                            for index, text in enumerate(bench.array_value(criteria))}
                           if kind == "score" else {key: 3.0 if key == "first" else 1.0
                                                  for key in bench.object_value(criteria)})
                total = sum(weights.values())
                probabilities = {key: weight / total for key, weight in weights.items()}
                answer = {"type": kind, "probabilities": probabilities, "confidence": 0.5}
                if kind == "score":
                    answer |= {"score": sum(int(key) * value for key, value in probabilities.items()),
                               "legend": {str(index): text for index, text in enumerate(bench.array_value(criteria))}}
                else:
                    answer["choice"] = max(probabilities, key=lambda key: probabilities[key])
            return 200, {"model": body["model"], "answers": {question_id: answer},
                         "usage": {"input_tokens": 12, "output_tokens": 3}, "request_id": "native-request"}
        if path == "/apply-template":
            return 200, {"prompt": json.dumps(body["messages"])}
        if path == "/tokenize":
            key = str(body["content"])
            return 200, {"tokens": list(range(len(key))) if self.multi else [100 + ord(key)]}
        if path == "/completion":
            keys = [json.loads(term) for term in str(body["grammar"]).removeprefix("root ::= ").split(" | ")]
            key = keys[0]
            if self.multi:
                tokens = list(key)
                probabilities = [0.6, 0.5 if key[-1] == "X" else 0.25]
            else:
                tokens = [key]
                # Stable by semantic option rather than positional letter.
                prompt = str(body["prompt"])
                probability = self.first_probability if f"{key}. first" in prompt or f"{key}. true" in prompt else self.other_probability
                probabilities = [probability]
            records = []
            for token, probability in zip(tokens, probabilities):
                score = math.log(probability)
                record: dict[str, object] = {
                    "token": token, "logprob": score,
                    "top_logprobs": [{"token": "Which", "logprob": math.log(0.1)}],
                }
                # A forced key may be outside top-k: the entry's own score is indispensable.
                if token == "A":
                    cast(list[dict[str, object]], record["top_logprobs"]).append({
                        "token": token, "logprob": score + (0.2 if self.mismatch_top else 0),
                    })
                if len(keys) > 1:
                    record["top_logprobs"] = [
                        {"token": candidate, "logprob": math.log(
                            self.first_probability if f"{candidate}. first" in str(body["prompt"]) or f"{candidate}. true" in str(body["prompt"])
                            else self.other_probability
                        ) + (0.2 if self.mismatch_top and candidate == token else 0)}
                        for candidate in keys if candidate != self.missing_primary_key
                    ]
                if self.omit_logprob and key == "B":
                    del record["logprob"]
                records.append(record)
            return 200, {"content": key, "completion_probabilities": records,
                         "tokens_evaluated": 11, "tokens_predicted": len(tokens)}
        if path == "/v1/chat/completions":
            return 200, {"choices": [{"finish_reason": self.thinking_finish,
                                      "message": {"content": self.thinking_text, "reasoning_content": "B"}}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        return 404, {"error": "unexpected path"}


@contextlib.contextmanager
def server(reply: Reply, authorization: list[str] | None = None) -> Generator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            if authorization is not None:
                authorization.append(self.headers.get("Authorization", ""))
            size = int(self.headers["Content-Length"])
            body = bench.object_value(json.loads(self.rfile.read(size)))
            status, response = reply(self.path, body)
            raw = json.dumps(response).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                # Expected only when the timeout test's client has already returned.
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


class AdapterTests(unittest.TestCase):
    def settings(self) -> bench.Settings:
        return bench.Settings("candidate-pin", {"enable_thinking": False}, "<|message|>",
                              {"temperature": 0.7, "chat_template_kwargs": {"enable_thinking": True}}, 8192)

    def variants(self, item: bench.Item) -> tuple[bench.Variant, ...]:
        return bench.make_variants(item, bench.DEFAULT_TEMPLATE.read_text(), 2, 7, "SYSTEM RULE")

    def test_preconstraint_own_logprobs_full_coverage_renormalization_and_leak(self) -> None:
        model = FakeModel()
        item = fixture()
        with tempfile.TemporaryDirectory() as directory, server(model.reply) as url:
            client = ChatClient(url, "synthetic-api-key", 2, artifacts_dir=Path(directory))
            result = bench.run_variant(item, self.variants(item)[0], "direct", client, self.settings())
            self.assertTrue(result.full_coverage, result.error)
            self.assertTrue(result.correct)
            self.assertAlmostEqual(result.probabilities["first"], 0.75)
            self.assertAlmostEqual(result.probabilities["second"], 0.25)
            self.assertAlmostEqual(cast(float, result.leaked_mass), 0.2)
            self.assertAlmostEqual(cast(float, result.margin), 0.5)
            self.assertAlmostEqual(cast(float, result.normalized_pmax), 0.5)
            self.assertEqual(result.native_response, {
                "model": "candidate-pin", "answers": {"q1": {"type": "choice", "choice": "first",
                "probabilities": result.probabilities, "confidence": result.normalized_pmax}},
                "usage": {"input_tokens": 11, "output_tokens": 1},
            })
            self.assertEqual(len(result.calls), 4)
            for call in result.calls:
                self.assertIsNotNone(call.artifact)
                record = json.loads(Path(cast(str, call.artifact)).read_bytes())
                self.assertTrue(record["request_body_utf8"])
                self.assertTrue(record["response_body_base64"])
                self.assertEqual(record["response"], call.response)
            completions = [body for path, body in model.requests if path == "/completion"]
            self.assertEqual(len(completions), 1)
            for body in completions:
                self.assertIs(body["post_sampling_probs"], False)
                self.assertIs(body["cache_prompt"], True)
                self.assertEqual(body["temperature"], 0)
                self.assertGreaterEqual(cast(int, body["n_probs"]), 50)
                self.assertTrue(str(body["prompt"]).endswith("<|message|>"))
            encoded = json.dumps(model.requests)
            self.assertNotIn("GOLD_RATIONALE_NEVER_SEND", encoded)
            self.assertNotIn("option_ids", encoded)
            rendered = model.requests[0][1]
            self.assertEqual(rendered["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(bench.array_value(rendered["messages"])[0], {"role": "system", "content": "SYSTEM RULE"})

    def test_multi_token_exact_scores_shared_first_token_leak(self) -> None:
        model = FakeModel()
        model.multi = True
        item = fixture()
        variant = bench.Variant("order-0", {"AX": "first", "AY": "second"}, [{"role": "user", "content": "multi"}])
        with server(model.reply) as url:
            result = bench.run_variant(item, variant, "direct", ChatClient(url, "", 2), self.settings())
        self.assertTrue(result.full_coverage, result.error)
        self.assertAlmostEqual(result.sequence_logprobs["AX"], math.log(0.3))
        self.assertAlmostEqual(result.sequence_logprobs["AY"], math.log(0.15))
        self.assertAlmostEqual(result.probabilities["first"], 2 / 3)
        self.assertAlmostEqual(cast(float, result.leaked_mass), 0.4)
        self.assertEqual(len(result.first_token_logprobs), 2)

    def test_single_forward_pass_scores_all_keys_without_forced_fallback(self) -> None:
        model = FakeModel()
        item = fixture()
        with server(model.reply) as url:
            row = bench.run_variant(item, self.variants(item)[0], "direct", ChatClient(url, "", 2), self.settings())
        completions = [body for path, body in model.requests if path == "/completion"]
        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0]["grammar"], 'root ::= "A" | "B"')
        self.assertGreaterEqual(cast(int, completions[0]["n_probs"]), 50)
        self.assertTrue(row.full_coverage, row.error)
        self.assertEqual(row.scoring_source, "single")
        self.assertFalse(row.mixed_sources)
        self.assertAlmostEqual(cast(float, row.raw_key_mass), 0.8)
        self.assertAlmostEqual(row.probabilities["first"], 0.75)

    def test_only_missing_key_uses_forced_fallback_and_marks_mixed_sources(self) -> None:
        model = FakeModel()
        model.missing_primary_key = "B"
        item = fixture()
        with server(model.reply) as url:
            row = bench.run_variant(item, self.variants(item)[0], "direct", ChatClient(url, "", 2), self.settings())
        completions = [body for path, body in model.requests if path == "/completion"]
        self.assertEqual([body["grammar"] for body in completions], ['root ::= "A" | "B"', 'root ::= "B"'])
        self.assertTrue(row.full_coverage, row.error)
        self.assertEqual(row.scoring_source, "mixed")
        self.assertTrue(row.mixed_sources)
        self.assertAlmostEqual(row.probabilities["first"], 0.75)

    def test_overshoot_is_rejected_only_for_single_source(self) -> None:
        item = fixture()
        for missing in (None, "B"):
            model = FakeModel()
            model.first_probability = 0.7
            model.other_probability = 0.35
            model.missing_primary_key = missing
            with self.subTest(missing=missing), server(model.reply) as url:
                row = bench.run_variant(item, self.variants(item)[0], "direct", ChatClient(url, "", 2), self.settings())
                self.assertEqual(row.full_coverage, missing is not None)
                self.assertAlmostEqual(cast(float, row.raw_key_mass), 1.05)
                if missing:
                    self.assertAlmostEqual(row.probabilities["first"], 2 / 3)
                    self.assertEqual(row.leaked_mass, 0)
                    self.assertTrue(row.mixed_sources)
                else:
                    self.assertIn("mass exceeds one", row.error or "")

    def test_repeat_draw_drift_preserves_first_draw_in_direct_and_native_modes(self) -> None:
        item = fixture()
        for mode in ("direct", "native"):
            model = FakeModel()
            draw_count = 0
            distributions = ((0.75, 0.25), (0.74, 0.26), (0.2, 0.8))

            def drifting_reply(
                path: str, body: dict[str, object], *, model: FakeModel = model,
                mode: str = mode, distributions: tuple[tuple[float, float], ...] = distributions,
            ) -> tuple[int, dict[str, object]]:
                nonlocal draw_count
                scoring = path == ("/completion" if mode == "direct" else "/v1/systemone")
                if scoring:
                    first, second = distributions[draw_count]
                    draw_count += 1
                    model.first_probability, model.other_probability = first, second
                status, response = model.reply(path, body)
                if scoring and mode == "native":
                    answer = bench.object_value(bench.object_value(response["answers"])["q1"])
                    answer["probabilities"] = {"first": model.first_probability, "second": model.other_probability}
                    answer["choice"] = "first" if model.first_probability > model.other_probability else "second"
                return status, response

            with self.subTest(mode=mode), server(drifting_reply) as url:
                row = bench.run_variant(item, self.variants(item)[0], mode, ChatClient(url, "", 2),
                                        dataclasses.replace(self.settings(), repeat_draws=3))
            self.assertEqual(draw_count, 3)
            self.assertTrue(row.correct)
            self.assertEqual(row.option_id, "first")
            self.assertAlmostEqual(row.probabilities["first"], 0.75)
            self.assertAlmostEqual(cast(float, row.normalized_pmax), 0.5)
            self.assertEqual(len(row.draws), 3)
            self.assertEqual(row.draws[2].option_id, "second")
            self.assertAlmostEqual(cast(float, row.drift["normalized_pmax_mean_absolute_difference"]), 0.08)
            self.assertAlmostEqual(cast(float, row.drift["normalized_pmax_max_absolute_difference"]), 0.12)
            self.assertAlmostEqual(cast(float, row.drift["per_option_probability_mean_absolute_difference"]), 1.1 / 3)
            self.assertAlmostEqual(cast(float, row.drift["per_option_probability_max_absolute_difference"]), 0.55)
            self.assertAlmostEqual(cast(float, row.drift["argmax_flip_rate"]), 2 / 3)
            crossings = bench.object_value(row.drift["normalized_pmax_floor_crossings"])
            self.assertEqual(crossings, {"0.5": True, "0.6": True, "0.7": False, "0.8": False, "0.9": False})
            summary = bench.object_value(bench.summarize([row], (item,), 2, mode, 1)["all"])
            drift = bench.object_value(summary["drift"])
            self.assertEqual(drift["comparable_cases"], 1)
            self.assertEqual(bench.object_value(drift["normalized_pmax_floor_crossing_cases"])["0.5"], 1)
            self.assertAlmostEqual(cast(float, drift["argmax_flip_rate"]), 2 / 3)
            self.assertEqual(bench.object_value(row.native_response)["usage"],
                             {"input_tokens": 11, "output_tokens": 1} if mode == "direct"
                             else {"input_tokens": 12, "output_tokens": 3})
            scoring_bodies = [body for path, body in model.requests
                              if path == ("/completion" if mode == "direct" else "/v1/systemone")]
            self.assertEqual(scoring_bodies, [scoring_bodies[0]] * 3)

    def test_repeat_failures_neither_replace_nor_invalidate_primary_draw(self) -> None:
        item = fixture()
        for mode, fail_first in itertools.product(("direct", "native"), (False, True)):
            model = FakeModel()
            draw_count = 0

            def one_failed_draw(
                path: str, body: dict[str, object], *, model: FakeModel = model,
                mode: str = mode, fail_first: bool = fail_first,
            ) -> tuple[int, dict[str, object]]:
                nonlocal draw_count
                status, response = model.reply(path, body)
                if path == ("/completion" if mode == "direct" else "/v1/systemone"):
                    draw_count += 1
                    if draw_count == (1 if fail_first else 2):
                        return 503, {"error": "synthetic analysis failure"}
                return status, response

            with self.subTest(mode=mode, fail_first=fail_first), server(one_failed_draw) as url:
                row = bench.run_variant(item, self.variants(item)[0], mode, ChatClient(url, "", 2),
                                        dataclasses.replace(self.settings(), repeat_draws=2))
            self.assertEqual(draw_count, 2)
            self.assertEqual(row.full_coverage, not fail_first)
            self.assertEqual(row.correct, not fail_first)
            self.assertEqual(row.drift["failed_draws"], 1)
            self.assertEqual(row.drift["valid_draws"], 1)
            self.assertIsNone(row.drift["argmax_flip_rate"])

    def test_missing_own_probability_and_disagreeing_top_are_coverage_failures(self) -> None:
        for mismatch in (False, True):
            model = FakeModel()
            model.omit_logprob = not mismatch
            model.mismatch_top = mismatch
            model.missing_primary_key = "B" if not mismatch else None
            item = fixture()
            with self.subTest(mismatch=mismatch), server(model.reply) as url:
                result = bench.run_variant(item, self.variants(item)[0], "direct", ChatClient(url, "", 2), self.settings())
                self.assertFalse(result.full_coverage)
                self.assertIsNone(result.option_id)
                self.assertEqual(result.probabilities, {})
                self.assertIsNotNone(result.error)
                summary = bench.object_value(bench.summarize([result], (item,), 2, "direct", 1)["all"])
                self.assertEqual(summary["n"], 2)
                self.assertEqual(summary["full_coverage_rate"], 0)
                self.assertEqual(summary["accuracy"], 0)

    def test_generated_order_mapping_and_stability_use_option_identity(self) -> None:
        model = FakeModel()
        item = fixture()
        variants = self.variants(item)
        self.assertEqual(variants, self.variants(item))
        self.assertNotEqual(variants[0].key_to_option, variants[1].key_to_option)
        with server(model.reply) as url:
            client = ChatClient(url, "", 2)
            rows = [bench.run_variant(item, variant, "direct", client, self.settings()) for variant in variants]
        self.assertEqual([row.option_id for row in rows], ["first", "first"])
        summary = bench.object_value(bench.summarize(rows, (item,), 2, "direct", 2)["all"])
        self.assertEqual(summary["accuracy"], 1)
        self.assertAlmostEqual(cast(float, summary["argmax_confidence_brier"]), 0.0625)
        self.assertAlmostEqual(cast(float, summary["argmax_confidence_log_loss"]), -math.log(0.75))
        stability = bench.object_value(summary["order_stability"])
        self.assertEqual(stability["argmax_agreement"], 1)
        self.assertAlmostEqual(cast(float, stability["js_distance_mean"]), 0)
        self.assertAlmostEqual(bench.js_distance({"a": 1, "b": 0}, {"a": 0, "b": 1}), 1)
        wrong = dataclasses.replace(rows[0], correct=False, margin=0.1)
        self.assertEqual(bench.auroc([rows[0], wrong]), 1)
        self.assertIsNone(bench.auroc(rows))
        self.assertEqual(bench.auroc([dataclasses.replace(wrong, margin=rows[0].margin), rows[0]]), 0.5)

    def test_defer_rules_and_special_confusion(self) -> None:
        item = dataclasses.replace(fixture(), gold=frozenset({"second"}),
                                   special={"defer": frozenset({"first"}), "unknown": frozenset({"second"})})
        self.assertFalse(bench.is_correct(item, "first"))  # Non-gold deferral needs permission.
        self.assertTrue(bench.is_correct(dataclasses.replace(item, defer_acceptable=True), "first"))
        self.assertFalse(bench.is_correct(item, None))
        model = FakeModel()
        with server(model.reply) as url:
            rows = [bench.run_variant(item, variant, "direct", ChatClient(url, "", 2), self.settings()) for variant in self.variants(item)]
        summary = bench.object_value(bench.summarize(rows, (item,), 2, "direct", 1)["all"])
        confusion = bench.object_value(summary["special_confusion"])
        self.assertEqual(bench.object_value(confusion["defer"])["fp"], 2)
        self.assertEqual(summary["accuracy"], 0)

    def test_list_markers_load_multiple_ids_empty_lists_and_legacy_values(self) -> None:
        row = json.loads(DATASET.read_text().splitlines()[1])
        row["special"] = {"new": ["red", "new-label"], "unknown": ["blue"], "defer": []}
        item = bench.parse_item(row)
        self.assertEqual(item.special, {"new": frozenset({"red", "new-label"}), "unknown": frozenset({"blue"})})
        for legacy in ("new-label", ["new-label"]):
            row["special"] = {"new": legacy, "unknown": None, "defer": []}
            self.assertEqual(bench.parse_item(row).special, {"new": frozenset({"new-label"})})
        for invalid in (["missing-option"], [17], [["red"]]):
            row["special"] = {"new": invalid}
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                bench.parse_item(row)

    def test_set_marker_defer_rules_apply_to_every_marked_option(self) -> None:
        item = dataclasses.replace(fixture(), special={"defer": frozenset({"first", "second"})})
        self.assertTrue(bench.is_correct(item, "first"))  # Explicit gold remains correct.
        self.assertFalse(bench.is_correct(item, "second"))
        allowed = dataclasses.replace(item, gold=frozenset(), defer_acceptable=True)
        self.assertTrue(bench.is_correct(allowed, "first"))
        self.assertTrue(bench.is_correct(allowed, "second"))
        row = json.loads(DATASET.read_text().splitlines()[1])
        row["special"] = {"defer": ["red", "new-label"]}
        row["gold"] = {"option_ids": ["new-label"], "defer_acceptable": False}
        explicit_gold = bench.parse_item(row)
        self.assertTrue(bench.is_correct(explicit_gold, "new-label"))
        self.assertFalse(bench.is_correct(explicit_gold, "red"))
        row["gold"] = {"option_ids": [], "defer_acceptable": True}
        parsed = bench.parse_item(row)
        self.assertTrue(bench.is_correct(parsed, "red"))
        self.assertTrue(bench.is_correct(parsed, "new-label"))
        row["special"] = {"defer": []}
        with self.assertRaisesRegex(ValueError, "gold must name available options or permit defer"):
            bench.parse_item(row)

    def test_set_marker_confusion_uses_gold_intersection_and_prediction_membership(self) -> None:
        item = dataclasses.replace(fixture(), special={"new": frozenset({"first", "second"}),
                                                     "unknown": frozenset({"second"})})
        model = FakeModel()
        with server(model.reply) as url:
            rows = [bench.run_variant(item, variant, "direct", ChatClient(url, "", 2), self.settings())
                    for variant in self.variants(item)]
        summary = bench.object_value(bench.summarize(rows, (item,), 2, "direct", 1)["all"])
        confusion = bench.object_value(summary["special_confusion"])
        self.assertEqual(bench.object_value(confusion["new"]), {"tp": 2, "fp": 0, "fn": 0, "tn": 0})
        self.assertEqual(bench.object_value(confusion["unknown"]), {"tp": 0, "fp": 0, "fn": 0, "tn": 2})
        self.assertEqual(bench.object_value(confusion["defer"]), {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    def test_explicit_defer_mapped_gold_loads_without_extra_permission(self) -> None:
        for case_id in ("rule_usefulness-006", "rule_usefulness-012"):
            row = {
                "id": case_id, "family": "rule_usefulness", "difficulty": "medium",
                "state": "SYNTHETIC regression: a matching rule is contested.",
                "question": {"id": "q1", "type": "choice", "text": "Choose the rule effect.",
                             "options": [{"id": key, "text": key} for key in
                                         ("propose", "penalty", "contested", "not_applicable", "demoted")]},
                "special": {"defer": ["contested"], "unknown": ["not_applicable"], "new": []},
                "gold": {"option_ids": ["contested"], "defer_acceptable": False},
            }
            with self.subTest(case_id=case_id):
                item = bench.parse_item(row)
                self.assertTrue(bench.is_correct(item, "contested"))
                self.assertFalse(bench.is_correct(item, "not_applicable"))

    def test_gold_membership_precedes_defer_permission(self) -> None:
        item = dataclasses.replace(fixture(), special={"defer": frozenset({"first", "second"})})
        self.assertTrue(bench.is_correct(item, "first"))
        self.assertFalse(bench.is_correct(item, "second"))
        self.assertTrue(bench.is_correct(dataclasses.replace(item, defer_acceptable=True), "second"))
        self.assertFalse(bench.is_correct(item, None))

    def test_explicit_defer_gold_is_true_positive_in_confusion(self) -> None:
        item = dataclasses.replace(fixture(), special={"defer": frozenset({"first"})})
        model = FakeModel()
        with server(model.reply) as url:
            rows = [bench.run_variant(item, variant, "direct", ChatClient(url, "", 2), self.settings())
                    for variant in self.variants(item)]
        summary = bench.object_value(bench.summarize(rows, (item,), 2, "direct", 1)["all"])
        confusion = bench.object_value(summary["special_confusion"])
        self.assertEqual(bench.object_value(confusion["defer"]), {"tp": 2, "fp": 0, "fn": 0, "tn": 0})
        self.assertEqual(summary["accuracy"], 1)

    def test_thinking_exact_final_parser_native_profile_and_truncation(self) -> None:
        model = FakeModel()
        item = fixture()
        with server(model.reply) as url:
            client = ChatClient(url, "", 2)
            valid = bench.run_variant(item, self.variants(item)[0], "thinking", client, self.settings())
            self.assertTrue(valid.correct)
            self.assertFalse(valid.full_coverage)
            self.assertEqual(valid.probabilities, {})
            for text in ("A or B", "A\nB", "Answer: A", "", "a", "<think>A</think>A"):
                model.thinking_text = text
                result = bench.run_variant(item, self.variants(item)[0], "thinking", client, self.settings())
                self.assertIsNone(result.option_id, text)
            model.thinking_text = "A"
            model.thinking_finish = "length"
            self.assertIsNone(bench.run_variant(item, self.variants(item)[0], "thinking", client, self.settings()).option_id)
        for path, body in model.requests:
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(body["temperature"], 0.7)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(body["max_tokens"], 8192)
        with self.assertRaises(ValueError):
            bench.native_profile('{"messages": []}')

    def test_http_and_timeout_failures_remain_in_denominator(self) -> None:
        item = fixture()
        rows = []
        for timeout in (False, True):
            model = FakeModel()
            if timeout:
                model.delay_path = "/apply-template"
            else:
                model.fail_path = "/apply-template"
            with server(model.reply) as url:
                rows.append(bench.run_variant(item, self.variants(item)[int(timeout)], "direct",
                            ChatClient(url, "", 0.02 if timeout else 2), self.settings()))
        summary = bench.object_value(bench.summarize(rows, (item,), 2, "direct", 1)["all"])
        self.assertEqual((summary["n"], summary["attempted"], summary["accuracy"]), (2, 2, 0))
        self.assertEqual(summary["http_errors"], 2)
        self.assertEqual(summary["timeout_requests"], 1)
        self.assertEqual(summary["full_coverage_rate"], 0)
        self.assertEqual(bench.object_value(summary["order_stability"])["argmax_agreement"], 0)
        slow = dataclasses.replace(rows[0], elapsed_seconds=12,
                                   calls=(dataclasses.replace(rows[0].calls[0], wall_seconds=11),))
        summary = bench.object_value(bench.summarize([slow], (item,), 2, "direct", 12)["all"])
        self.assertEqual(summary["over_10s_requests"], 1)
        self.assertEqual(summary["over_10s_item_variants"], 1)

    def test_native_shapes_score_scalars_noul_thresholds_and_usage_unknown(self) -> None:
        score = fixture("score")
        probabilities = {"first": 0.25, "second": 0.75}
        answer = bench.native_answer(score, probabilities, 0.5)
        self.assertEqual(answer["legend"], {"0": "first", "1": "second"})
        self.assertEqual(answer["probabilities"], {"0": 0.25, "1": 0.75})
        self.assertEqual(answer["score"], 0.75)
        metrics = bench.scalar_metrics(score, probabilities)
        self.assertEqual(metrics["expected_absolute_error"], 0.75)
        self.assertEqual(metrics["argmax_absolute_error"], 1)
        self.assertFalse(metrics["argmax_exact_level_match"])
        noul = fixture("noul")
        self.assertEqual(bench.native_answer(noul, {"true": 0.75, "false": 0.25}, 0.5), {"type": "noul", "noul": 0.75})
        metrics = bench.scalar_metrics(noul, {"true": 0.75, "false": 0.25})
        self.assertEqual(metrics["threshold_outcomes"], {"0.6": "correct", "0.7": "correct", "0.8": "ambiguous"})
        false_gold = dataclasses.replace(noul, gold=frozenset({"false"}))
        self.assertEqual(bench.scalar_metrics(false_gold, {"true": 0.25, "false": 0.75})["threshold_outcomes"], metrics["threshold_outcomes"])
        request = bench.native_request(noul, "pin")
        self.assertEqual(request, {"model": "pin", "state": "SYNTHETIC STATE", "questions": {
            "q1": {"type": "noul", "instructions": "Pick the first semantic option.", "criteria": {"true": "true", "false": "false"}},
        }})
        call = bench.ApiResult(True, 1, response={"completion_probabilities": []})
        self.assertIsNone(bench.native_response(score, "pin", probabilities, 0.5, [call]))

    def test_cli_both_arms_synthetic_fixture_manifest_retention_and_collision(self) -> None:
        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory, server(model.reply) as url:
            output = Path(directory) / "smoke.json"
            provenance = Path(directory) / "provenance.json"
            provenance.write_text('{"runtime":"fake local HTTP server"}')
            fixtures = Path(directory) / "list-markers.jsonl"
            fixture_rows = [json.loads(line) for line in DATASET.read_text().splitlines()]
            for row in fixture_rows:
                row["special"] = {key: [value] if value is not None else [] for key, value in row["special"].items()}
            fixtures.write_text("".join(json.dumps(row) + "\n" for row in fixture_rows))
            argv = ["system_one_benchmark.py", "--fixtures", str(fixtures), "--base-url", url,
                    "--model", "pin", "--mode", "both", "--concurrency", "2", "--repeat-draws", "2", "--output", str(output),
                    "--artifacts-dir", str(Path(directory) / "raw"), "--manifest", str(provenance),
                    "--direct-template-kwargs", '{"enable_thinking":false}', "--system-rule", "RULE"]
            with patch("sys.argv", argv), patch.dict("os.environ", {"LOCAL_LLM_API_KEY": "synthetic-api-key"}), patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(bench.main(), 0)
                request_count = len(model.requests)
                self.assertEqual(bench.main(), 2)
                self.assertEqual(len(model.requests), request_count)
            report = json.loads(output.read_bytes())
            self.assertTrue(report["complete"])
            self.assertEqual(report["selected_items"], 4)
            self.assertEqual(len(report["cases"]), 16)
            for mode in ("direct", "thinking"):
                self.assertEqual(report["modes"][mode]["all"]["n"], 8)
            self.assertEqual(report["modes"]["direct"]["all"]["full_coverage_rate"], 1)
            self.assertEqual(report["modes"]["direct"]["all"]["drift"]["comparable_cases"], 8)
            self.assertEqual(report["modes"]["direct"]["all"]["scoring_source_counts"], {"single": 8, "mixed": 0})
            manifest = json.loads(output.with_suffix(".manifest.json").read_bytes())
            self.assertEqual(len(manifest["dataset_sha256"]), 1)
            self.assertEqual(len(manifest["native_requests"]), 4)
            self.assertEqual(manifest["provenance"]["runtime"], "fake local HTTP server")
            self.assertEqual(manifest["settings"]["repeat_draws"], 2)
            self.assertEqual(manifest["fixtures_for_scoring_only"][1]["special"], {"new": ["new-label"]})
            self.assertTrue(output.with_suffix(".run-owner.json").exists())
            self.assertEqual(len(list((Path(directory) / "raw").glob("*.json"))), request_count)
            for row in report["cases"]:
                for call in row["calls"]:
                    self.assertTrue(Path(call["artifact"]).exists())

    def test_native_choice_order_metrics_and_provider_metadata(self) -> None:
        item = dataclasses.replace(fixture(), special={"new": frozenset({"first"}), "unknown": frozenset({"second"})})
        model = FakeModel()
        variants = self.variants(item)
        with server(model.reply) as url:
            rows = [bench.run_variant(item, variant, "native", ChatClient(url, "", 2), self.settings()) for variant in variants]
        for row in rows:
            self.assertTrue(row.full_coverage, row.error)
            self.assertTrue(row.correct)
            self.assertEqual(row.probabilities, {"first": 0.75, "second": 0.25})
            self.assertEqual(row.sequence_logprobs, {})
            self.assertIsNone(row.leaked_mass)
            self.assertEqual(row.normalized_pmax, 0.5)
            self.assertEqual(row.margin, 0.5)
            self.assertAlmostEqual(cast(float, row.entropy), -0.75 * math.log(0.75) - 0.25 * math.log(0.25))
            self.assertEqual(bench.object_value(row.native_response)["request_id"], "native-request")
            self.assertEqual(bench.object_value(row.native_response)["usage"], {"input_tokens": 12, "output_tokens": 3})
            self.assertEqual(len(row.calls), 1)
        for (path, body), variant in zip(model.requests, variants):
            self.assertEqual(path, "/v1/systemone")
            self.assertEqual(set(body), {"model", "state", "questions"})
            question = bench.object_value(bench.object_value(body["questions"])["q1"])
            self.assertEqual(list(bench.object_value(question["criteria"])), list(variant.key_to_option.values()))
            self.assertNotIn("GOLD_RATIONALE_NEVER_SEND", json.dumps(body))
        summary = bench.object_value(bench.summarize(rows, (item,), 2, "native", 2)["all"])
        self.assertEqual(summary["full_coverage_rate"], 1)
        self.assertEqual(summary["accuracy"], 1)
        self.assertEqual(summary["normalized_pmax_mean"], 0.5)
        self.assertEqual(summary["argmax_confidence_brier"], 0.0625)
        self.assertAlmostEqual(cast(float, summary["argmax_confidence_log_loss"]), -math.log(0.75))
        self.assertEqual(bench.object_value(summary["order_stability"])["argmax_agreement"], 1)
        self.assertEqual(bench.object_value(summary["order_stability"])["js_distance_mean"], 0)
        confusion = bench.object_value(summary["special_confusion"])
        self.assertEqual(bench.object_value(confusion["new"]), {"tp": 2, "fp": 0, "fn": 0, "tn": 0})
        wrong = dataclasses.replace(rows[0], correct=False, margin=0.1)
        self.assertEqual(bench.auroc([rows[0], wrong]), 1)

    def test_native_noul_and_score_restore_original_option_identities(self) -> None:
        for kind in ("noul", "score"):
            model = FakeModel()
            item = fixture(kind)
            with self.subTest(kind=kind), server(model.reply) as url:
                rows = [bench.run_variant(item, variant, "native", ChatClient(url, "", 2), self.settings())
                        for variant in self.variants(item)]
            expected = {"true": 0.75, "false": 0.25} if kind == "noul" else {"first": 0.75, "second": 0.25}
            for row in rows:
                self.assertTrue(row.full_coverage, row.error)
                self.assertTrue(row.correct)
                self.assertEqual(row.probabilities, expected)
                self.assertEqual(row.scalar_metrics, bench.scalar_metrics(item, expected))
            if kind == "score":
                questions = [bench.object_value(bench.object_value(body["questions"])["q1"]) for _, body in model.requests]
                self.assertEqual(questions[0]["criteria"], ["first", "second"])
                self.assertEqual(questions[1]["criteria"], ["second", "first"])
                answer = bench.object_value(bench.object_value(bench.object_value(rows[1].native_response)["answers"])["q1"])
                self.assertEqual(answer["probabilities"], {"0": 0.25, "1": 0.75})
                self.assertEqual(bench.object_value(rows[1].native_answer)["probabilities"], {"0": 0.75, "1": 0.25})

    def test_native_model_drift_and_bad_distributions_are_item_failures(self) -> None:
        invalid: tuple[tuple[str, object], ...] = (
            ("model", "different-pin"), ("model", None),
            ("probabilities", {"first": 1}), ("probabilities", {"first": 0.75, "second": 0.25, "extra": 0}),
            ("probabilities", {"first": -0.1, "second": 1.1}),
            ("probabilities", {"first": True, "second": 0}),
            ("probabilities", {"first": "0.75", "second": 0.25}),
            ("probabilities", {"first": float("nan"), "second": 0.25}),
            ("probabilities", {"first": float("inf"), "second": 0.25}),
            ("probabilities", {"first": 0.4, "second": 0.4}),
            ("type", "score"), ("answers", {}), ("usage", {"input_tokens": -1, "output_tokens": 3}),
        )
        item = fixture()
        model = FakeModel()
        for field, value in invalid:
            def reply(path: str, body: dict[str, object], field: str = field, value: object = value) -> tuple[int, dict[str, object]]:
                status, response = model.reply(path, body)
                if field in {"model", "answers", "usage"}:
                    response[field] = value
                else:
                    bench.object_value(bench.object_value(response["answers"])["q1"])[field] = value
                return status, response

            with self.subTest(field=field, value=value), server(reply) as url:
                row = bench.run_variant(item, self.variants(item)[0], "native", ChatClient(url, "", 2), self.settings())
            self.assertFalse(row.full_coverage)
            self.assertFalse(row.correct)
            self.assertIsNone(row.option_id)
            self.assertEqual(row.probabilities, {})
            self.assertIsNotNone(row.error)
            if field == "model":
                self.assertIn("ModelDrift", cast(str, row.error))
            summary = bench.object_value(bench.summarize([row], (item,), 2, "native", 1)["all"])
            self.assertEqual((summary["n"], summary["accuracy"], summary["full_coverage_rate"]), (2, 0, 0))
        for kind, probabilities in (("noul", 1.1), ("score", {"0": 0.75, "2": 0.25})):
            def invalid_scalar_reply(path: str, body: dict[str, object], kind: str = kind, probabilities: object = probabilities) -> tuple[int, dict[str, object]]:
                status, response = model.reply(path, body)
                answer = bench.object_value(bench.object_value(response["answers"])["q1"])
                answer["noul" if kind == "noul" else "probabilities"] = probabilities
                return status, response

            item = fixture(kind)
            with self.subTest(kind=kind), server(invalid_scalar_reply) as url:
                row = bench.run_variant(item, self.variants(item)[0], "native", ChatClient(url, "", 2), self.settings())
            self.assertFalse(row.full_coverage)
            self.assertEqual(row.probabilities, {})

    def test_native_http_timeouts_and_slow_requests_stay_in_denominator(self) -> None:
        item = fixture()
        rows = []
        for timeout in (False, True):
            model = FakeModel()
            if timeout:
                model.delay_path = "/v1/systemone"
            else:
                model.fail_path = "/v1/systemone"
            with server(model.reply) as url:
                rows.append(bench.run_variant(item, self.variants(item)[int(timeout)], "native",
                                             ChatClient(url, "", 0.02 if timeout else 2), self.settings()))
        rows[0] = dataclasses.replace(rows[0], elapsed_seconds=12,
                                      calls=(dataclasses.replace(rows[0].calls[0], wall_seconds=11),))
        summary = bench.object_value(bench.summarize(rows, (item,), 2, "native", 13)["all"])
        self.assertEqual((summary["n"], summary["accuracy"], summary["full_coverage_rate"]), (2, 0, 0))
        self.assertEqual((summary["http_errors"], summary["timeout_requests"]), (2, 1))
        self.assertEqual((summary["over_10s_requests"], summary["over_10s_item_variants"]), (1, 1))

    def test_native_cli_auth_redaction_state_warning_and_manifest(self) -> None:
        model = FakeModel()
        authorization: list[str] = []
        key = "synthetic-native-key"

        def reply(path: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
            status, response = model.reply(path, body)
            response["request_id"] = key
            return status, response

        with tempfile.TemporaryDirectory() as directory, server(reply, authorization) as url:
            output = Path(directory) / "native.json"
            argv = ["system_one_benchmark.py", "--fixtures", str(DATASET), "--base-url", url,
                    "--model", "pin", "--mode", "native", "--output", str(output),
                    "--artifacts-dir", str(Path(directory) / "raw"), "--api-key-env", "NATIVE_TEST_API_KEY",
                    "--state-token-warning", "1"]
            stdout = io.StringIO()
            with patch("sys.argv", argv), patch.dict("os.environ", {"NATIVE_TEST_API_KEY": key}), patch("sys.stdout", stdout):
                self.assertEqual(bench.main(), 0)
            self.assertEqual(authorization, [f"Bearer {key}"] * 8)
            self.assertNotIn(key, stdout.getvalue())
            report = json.loads(output.read_bytes())
            manifest = json.loads(output.with_suffix(".manifest.json").read_bytes())
            self.assertTrue(report["complete"])
            self.assertEqual(report["modes"]["native"]["all"]["n"], 8)
            self.assertEqual(manifest["settings"]["state_token_warning"], 1)
            self.assertEqual(len(manifest["native_variant_requests"]), 8)
            for row in report["cases"]:
                self.assertGreater(row["state_bytes"], 4)
                self.assertEqual(row["state_estimated_tokens"], row["state_bytes"] / 4)
                self.assertTrue(row["state_token_warning_exceeded"])
                self.assertEqual(row["native_response"]["request_id"], "[REDACTED_API_KEY]")
            self.assertEqual(report["modes"]["native"]["all"]["state_token_warning_item_variants"], 8)
            for path in Path(directory).rglob("*.json"):
                text = path.read_text()
                self.assertNotIn(key, text)
                record = json.loads(text)
                if "response_body_base64" in record:
                    self.assertNotIn(key.encode(), base64.b64decode(record["response_body_base64"]))
                    self.assertTrue(record["credential_redaction"])
            # The heuristic uses UTF-8 bytes, with compact JSON for structured state.
            self.assertEqual(bench.state_byte_size("é"), 2)
            self.assertEqual(bench.state_byte_size({"a": "é"}), len('{"a":"é"}'.encode()))
            for threshold in ("0", "-1", "nan", "inf"):
                with patch("sys.argv", argv + ["--state-token-warning", threshold]), patch("sys.stderr", io.StringIO()):
                    self.assertEqual(bench.main(), 2)

    def test_native_state_warning_threshold_is_strict(self) -> None:
        item = dataclasses.replace(fixture(), state="é" * 4)
        model = FakeModel()
        with server(model.reply) as url:
            rows = [bench.run_variant(item, self.variants(item)[0], "native", ChatClient(url, "", 2),
                                      dataclasses.replace(self.settings(), state_token_warning=threshold))
                    for threshold in (2, 1)]
        self.assertEqual([row.state_bytes for row in rows], [8, 8])
        self.assertEqual([row.state_estimated_tokens for row in rows], [2, 2])
        self.assertEqual([row.state_token_warning_exceeded for row in rows], [False, True])

    def test_single_option_normalized_gate_and_score_are_defined(self) -> None:
        item = fixture("score")
        item = dataclasses.replace(item, options=item.options[:1])
        model = FakeModel()
        with server(model.reply) as url:
            rows = [bench.run_variant(item, variant, "direct", ChatClient(url, "", 2), self.settings())
                    for variant in self.variants(item)]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row.normalized_pmax, 1)
            self.assertEqual(row.probabilities, {"first": 1})
            self.assertEqual(bench.object_value(row.native_answer)["score"], 0)
        scalars = bench.summarize_scalars(rows)
        self.assertEqual(scalars["score_argmax_exact_level_matches"], 2)

    def test_fixture_validation_and_native_byte_budgets(self) -> None:
        items, hashes = bench.load_fixtures([DATASET], "pin")
        self.assertEqual({item.question["type"] for item in items}, {"noul", "choice", "score"})
        self.assertEqual(len(hashes), 1)
        self.assertIn("SYNTHETIC", items[0].id)
        raw = json.loads(DATASET.read_text().splitlines()[0])
        for state in ("x" * 32769, "é" * 16385, 7):
            with self.subTest(state_type=type(state)), self.assertRaises((ValueError, TypeError)):
                bench.parse_item(raw | {"state": state})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            huge = raw | {"question": raw["question"] | {"text": "x" * 62000}}
            path.write_text(json.dumps(huge) + "\n")
            with self.assertRaisesRegex(ValueError, "61440"):
                bench.load_fixtures([path], "pin")
            path.write_text(json.dumps(raw) + "\n" + json.dumps(raw) + "\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                bench.load_fixtures([path], "pin")


if __name__ == "__main__":
    unittest.main()
