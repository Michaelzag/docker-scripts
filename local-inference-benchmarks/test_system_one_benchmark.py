"""Offline System One contract tests; only a loopback fake HTTP server is used."""

from __future__ import annotations

import contextlib
import dataclasses
import io
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

    def reply(self, path: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
        self.requests.append((path, body))
        if path == self.delay_path:
            time.sleep(0.1)
        if path == self.fail_path:
            return 503, {"error": "synthetic unavailable"}
        if path == "/apply-template":
            return 200, {"prompt": json.dumps(body["messages"])}
        if path == "/tokenize":
            key = str(body["content"])
            return 200, {"tokens": list(range(len(key))) if self.multi else [100 + ord(key)]}
        if path == "/completion":
            key = json.loads(str(body["grammar"]).removeprefix("root ::= "))
            if self.multi:
                tokens = list(key)
                probabilities = [0.6, 0.5 if key[-1] == "X" else 0.25]
            else:
                tokens = [key]
                # Stable by semantic option rather than positional letter.
                prompt = str(body["prompt"])
                probability = 0.6 if f"{key}. first" in prompt or f"{key}. true" in prompt else 0.2
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
def server(reply: Reply) -> Generator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
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
                "usage": {"input_tokens": 22, "output_tokens": 2},
            })
            self.assertEqual(len(result.calls), 5)
            for call in result.calls:
                self.assertIsNotNone(call.artifact)
                record = json.loads(Path(cast(str, call.artifact)).read_bytes())
                self.assertTrue(record["request_body_utf8"])
                self.assertTrue(record["response_body_base64"])
                self.assertEqual(record["response"], call.response)
            completions = [body for path, body in model.requests if path == "/completion"]
            self.assertEqual(len(completions), 2)
            for body in completions:
                self.assertIs(body["post_sampling_probs"], False)
                self.assertIs(body["cache_prompt"], True)
                self.assertEqual(body["temperature"], 0)
                self.assertGreaterEqual(cast(int, body["n_probs"]), 20)
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

    def test_missing_own_probability_and_disagreeing_top_are_coverage_failures(self) -> None:
        for mismatch in (False, True):
            model = FakeModel()
            model.omit_logprob = not mismatch
            model.mismatch_top = mismatch
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
        item = dataclasses.replace(fixture(), special={"defer": frozenset({"first"}), "unknown": frozenset({"second"})})
        self.assertFalse(bench.is_correct(item, "first"))  # Even if defer is listed in gold.
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
        self.assertFalse(bench.is_correct(item, "first"))
        self.assertFalse(bench.is_correct(item, "second"))
        allowed = dataclasses.replace(item, gold=frozenset(), defer_acceptable=True)
        self.assertTrue(bench.is_correct(allowed, "first"))
        self.assertTrue(bench.is_correct(allowed, "second"))
        row = json.loads(DATASET.read_text().splitlines()[1])
        row["special"] = {"defer": ["red", "new-label"]}
        row["gold"] = {"option_ids": ["new-label"], "defer_acceptable": False}
        with self.assertRaisesRegex(ValueError, "effective correct"):
            bench.parse_item(row)
        row["gold"] = {"option_ids": [], "defer_acceptable": True}
        parsed = bench.parse_item(row)
        self.assertTrue(bench.is_correct(parsed, "red"))
        self.assertTrue(bench.is_correct(parsed, "new-label"))
        row["special"] = {"defer": []}
        with self.assertRaisesRegex(ValueError, "effective correct"):
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
                    "--model", "pin", "--mode", "both", "--concurrency", "2", "--output", str(output),
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
            manifest = json.loads(output.with_suffix(".manifest.json").read_bytes())
            self.assertEqual(len(manifest["dataset_sha256"]), 1)
            self.assertEqual(len(manifest["native_requests"]), 4)
            self.assertEqual(manifest["provenance"]["runtime"], "fake local HTTP server")
            self.assertEqual(manifest["fixtures_for_scoring_only"][1]["special"], {"new": ["new-label"]})
            self.assertTrue(output.with_suffix(".run-owner.json").exists())
            self.assertEqual(len(list((Path(directory) / "raw").glob("*.json"))), request_count)
            for row in report["cases"]:
                for call in row["calls"]:
                    self.assertTrue(Path(call["artifact"]).exists())

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
