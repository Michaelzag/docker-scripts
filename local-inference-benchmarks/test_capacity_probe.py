"""Offline capacity behavior; only controlled loopback HTTP, never model calls."""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import re
import tempfile
import threading
import time
import unittest
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from unittest.mock import patch

import capacity_probe as probe
from agent_workload_benchmark import ApiResult, ChatClient, set_request_body_extra


def prepared(case_id: str = "request-0", tokens: int = 120000) -> probe.Prepared:
    return probe.Prepared(case_id, "long", {}, {"fact_0": "amber", "fact_1": "birch", "fact_2": "cobalt"}, tokens, (), ())


def result(item: probe.Prepared, *, wall: float = 5.0) -> ApiResult:
    return ApiResult(True, wall, 200, {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(item.expected)}}],
        "timings": {"prompt_n": item.input_tokens, "cache_n": 0, "prompt_ms": 2000, "predicted_ms": 1000},
        "__verbose": {"truncated": False},
        "usage": {"completion_tokens": 12},
    })


class BudgetTests(unittest.TestCase):
    def test_fixed_and_mixed_counts_include_extra_short_for_four_or_six(self):
        for parallel, longs in ((4, 1), (6, 2)):
            fixed = probe.Plan("fixed", parallel, 131072, parallel * 131072)
            pooled = probe.Plan("pooled", parallel, 131072, 393216)
            self.assertEqual(fixed.kinds(), ("long",) * parallel + ("short",))
            self.assertEqual(pooled.kinds().count("long"), longs)
            self.assertEqual(pooled.kinds().count("short"), parallel - longs + 1)
            self.assertEqual(len(pooled.kinds()), parallel + 1)

    def test_actual_per_slot_boundary_not_character_or_nominal_budget(self):
        plan = probe.Plan("fixed", 4, 131072, 524288)
        accepted = probe.budget_proof(plan, [122880] * 4 + [8000])
        self.assertEqual(accepted["worst_active_total"], 524288)
        with self.assertRaisesRegex(ValueError, "actual formatted"):
            probe.budget_proof(plan, [122881] + [120000] * 3 + [8000])
        for invalid in ([120000] * 4, [120000] * 4 + [True], [120000] * 4 + [0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                probe.budget_proof(plan, invalid)

    def test_pool_uses_worst_possible_active_subset_and_all_output_ceilings(self):
        counts = [120000, 120000, 8000, 8000, 8000, 8000, 8001]
        total = 120000 * 2 + 8000 * 3 + 8001 + 6 * 8192
        plan = probe.Plan("pooled", 6, 131072, total)
        proof = probe.budget_proof(plan, counts)
        self.assertEqual(proof["worst_active_total"], total)
        self.assertEqual(proof["all_active_output_ceilings"], 49152)
        self.assertFalse(proof["independent_reservations"])
        with self.assertRaisesRegex(ValueError, "shared pool"):
            probe.budget_proof(dataclasses.replace(plan, pool=total - 1), counts)

    def test_smaller_pool_requires_explicit_long_worker_choice_never_auto_shrinks(self):
        default = probe.Plan("pooled", 6, 131072, 262144)
        self.assertEqual(default.kinds().count("long"), 2)
        with self.assertRaisesRegex(ValueError, "shared pool"):
            probe.budget_proof(default, [120000 if kind == "long" else 8000 for kind in default.kinds()])
        explicit = dataclasses.replace(default, long_workers=1)
        counts = [120000 if kind == "long" else 8000 for kind in explicit.kinds()]
        self.assertEqual(probe.budget_proof(explicit, counts)["worst_active_total"], 209152)
        self.assertEqual(explicit.kinds(), ("long",) + ("short",) * 6)
        with self.assertRaises(ValueError):
            probe.Plan("fixed", 6, 131072, 786432, long_workers=1)


class CalibrationTests(unittest.TestCase):
    def tearDown(self):
        set_request_body_extra({})

    def test_nonlinear_actual_formatted_counts_choose_in_budget_payload_and_keep_all_calls(self):
        set_request_body_extra({"temperature": 1.0, "top_p": 0.95,
                                "chat_template_kwargs": {"enable_thinking": True}})
        counted: list[tuple[dict[str, object], int]] = []
        retained: list[dict[str, object]] = []

        def count(body: dict[str, object], case_id: str) -> ApiResult:
            messages = cast(list[dict[str, object]], body["messages"])
            text = cast(str, messages[0]["content"])
            units = len(re.findall(r"^Archive ", text, re.MULTILINE))
            # A synthetic tokenizer with nonlinear template overhead: the
            # boundary cannot be solved by character counts or length growth.
            tokens = 73 + units * 11 + units // 9
            counted.append((body, tokens))
            return ApiResult(True, 0.01, 200, {"input_tokens": tokens}, artifact=f"raw/{case_id}")

        item = probe.calibrate("model", "long-0", "long", 4000, 9, "unique-client-0", count, retained.append)
        self.assertLessEqual(item.input_tokens, 4000)
        self.assertGreaterEqual(item.input_tokens, 4000 - 512)
        self.assertIn((item.body, item.input_tokens), counted)
        self.assertEqual(len(item.calibration), len(counted))
        self.assertEqual(len(retained), len(counted))
        self.assertTrue(all(entry["call"] for entry in retained))
        self.assertEqual(item.body["response_format"], {"type": "json_object"})
        self.assertEqual(item.body["max_tokens"], 8192)
        self.assertEqual(item.body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertIs(item.body["cache_prompt"], False)
        self.assertIs(item.body["verbose"], True)
        positions = [cast(float, position["archive_fraction"]) for position in item.needle_positions]
        for actual, expected in zip(positions, (0.08, 0.51, 0.91), strict=True):
            self.assertAlmostEqual(actual, expected, delta=0.01)

    def test_unsupported_count_fails_without_alternative_tokenizer(self):
        for count_result in (ApiResult(False, 0.1, 404, error="not supported"),
                             ApiResult(True, 0.1, 200, {}),
                             ApiResult(True, 0.1, 200, {"input_tokens": True}),
                             ApiResult(True, 0.1, 200, {"input_tokens": 0})):
            retained: list[dict[str, object]] = []
            with self.subTest(result=count_result), self.assertRaisesRegex(ValueError, "unsupported/unknown"):
                probe.calibrate("m", "id", "long", 4000, 9, "n",
                                lambda body, case, value=count_result: value, retained.append)
            self.assertEqual(len(retained), 1)

    def test_main_unsupported_count_preserves_raw_attempt_and_never_infers(self):
        # Actual retaining transport sees only an offline 404. This exercises
        # startup -> calibration failure checkpoint, not mere mock forwarding.
        import urllib.error

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "runtime.json"
            manifest.write_text('{"runtime":"offline-test"}')
            args = argparse.Namespace(
                mode="pooled", parallel=4, per_slot=131072, pool=393216,
                long_target=120000, short_target=8000, long_workers=None,
                measurement_timeout=600.0, extra_body_json='{"temperature":1.0}',
                manifest=manifest, output=root / "report.json", api_key_env="CAPACITY_OFFLINE_NO_KEY",
                base_url="http://offline.invalid/v1", model="offline", artifacts_dir=root / "raw",
                label="offline", seed=9,
            )
            error = urllib.error.HTTPError("http://offline.invalid/v1/chat/completions/input_tokens", 404,
                                           "unsupported", Message(), io.BytesIO(b'{"error":"unsupported"}'))
            with patch.object(probe, "parse_args", return_value=args), \
                    patch("urllib.request.urlopen", side_effect=error), \
                    patch.object(ChatClient, "chat", side_effect=AssertionError("inference forbidden")), \
                    patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(probe.main(), 2)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["attempted"], 0)
            self.assertEqual(report["selected"], 5)
            self.assertEqual(report["phase"], "calibration_failed_no_inference")
            raw_records = [json.loads(path.read_text()) for path in args.artifacts_dir.glob("*.json")]
            self.assertEqual(len(raw_records), 1)
            self.assertEqual(raw_records[0]["http_status"], 404)
            self.assertEqual(raw_records[0]["response_body_utf8"], '{"error":"unsupported"}')
            self.assertTrue(args.output.with_suffix(".calibration.jsonl").read_text())


class EvidenceTests(unittest.TestCase):
    def test_exact_verbose_input_proof_cache_and_unknown_fields_fail_closed(self):
        item = prepared()
        base = cast(dict[str, object], result(item).response)
        self.assertTrue(probe.input_proof(base, 120000)["passed"])
        for change in (
            {"timings": {"prompt_n": 119999, "cache_n": 0}},
            {"timings": {"prompt_n": 119999, "cache_n": 1}},
            {"timings": {"prompt_n": 120000}},
            {"timings": {"prompt_n": True, "cache_n": 0}},
            {"__verbose": {"truncated": True}},
            {"__verbose": {}}, {"__verbose": {"truncated": "false"}},
        ):
            with self.subTest(change=change):
                self.assertFalse(probe.input_proof({**base, **change}, 120000)["passed"])
        self.assertEqual(probe.input_proof({}, 120000)["state"], "UNKNOWN")

    def test_wrong_facts_duplicate_json_and_no_answer_repair(self):
        expected = prepared().expected
        self.assertTrue(probe.retrieval(json.dumps(expected), expected)["passed"])
        for text in (
            json.dumps({**expected, "fact_2": "wrong"}),
            json.dumps({"fact_0": "amber", "fact_1": "birch"}),
            json.dumps({**expected, "extra": "cobalt"}),
            '```json\n' + json.dumps(expected) + '\n```',
            '<think>' + json.dumps(expected) + '</think>',
            '{"fact_0":"wrong","fact_0":"amber","fact_1":"birch","fact_2":"cobalt"}',
        ):
            with self.subTest(text=text):
                self.assertFalse(probe.retrieval(text, expected)["passed"])

    def test_semantic_finish_and_provisioning_are_independent(self):
        item = prepared()
        response = result(item)
        assert response.response is not None
        response.response["__verbose"] = {"truncated": True}
        row = probe.evaluate(item, response, 600)
        self.assertTrue(cast(dict[str, object], row["retrieval"])["passed"])
        self.assertFalse(row["usable_retrieval_pass"])
        self.assertTrue(row["finish_stop"])
        response = result(item)
        assert response.response is not None
        response.response["choices"][0]["finish_reason"] = "length"
        row = probe.evaluate(item, response, 600)
        self.assertTrue(cast(dict[str, object], row["input_proof"])["passed"])
        self.assertFalse(row["finish_stop"])
        self.assertFalse(row["usable_retrieval_pass"])

    def test_residual_is_unclamped_difference_not_queue_and_missing_is_unknown(self):
        item = prepared()
        row = probe.evaluate(item, result(item, wall=5), 600)
        self.assertEqual(row["prefill_seconds"], 2)
        self.assertEqual(row["decode_seconds"], 1)
        self.assertEqual(row["residual_seconds_not_queue"], 2)
        self.assertIsNone(row["exact_queue_seconds"])
        self.assertEqual(probe.evaluate(item, result(item, wall=2), 600)["residual_seconds_not_queue"], -1)
        missing = result(item)
        assert missing.response is not None
        missing.response["timings"].pop("predicted_ms")
        self.assertIsNone(probe.evaluate(item, missing, 600)["residual_seconds_not_queue"])

    def test_deadline_http_errors_and_unattempted_stay_in_denominator(self):
        item = prepared()
        rows = [probe.evaluate(item, result(item), 600),
                probe.evaluate(item, result(item, wall=600.01), 600),
                probe.evaluate(item, ApiResult(False, 600, error="TimeoutError: timed out"), 600)]
        self.assertTrue(rows[1]["wall_exceeds_configured_io_timeout"])
        self.assertTrue(rows[1]["usable_retrieval_pass"])
        summary = probe.summarize(rows, 5, 601)
        self.assertEqual(summary["usable_passed"], 2)
        self.assertEqual(summary["usable_failed_or_unknown"], 3)
        self.assertEqual(summary["unattempted"], 2)
        self.assertEqual(summary["usable_fraction_selected"], 0.4)
        self.assertEqual(summary["output_tokens_reported_requests"], 2)
        self.assertEqual(summary["group_end_to_end_reported_output_tokens_per_second"], 24 / 601)
        self.assertIsNone(probe.summarize([], 5, 0)["reported_output_tokens"])


class MetricsAndConcurrencyTests(unittest.TestCase):
    def test_missing_invalid_and_duplicate_metrics_never_become_zero(self):
        for text in ("", "# disabled", "llamacpp:requests_processing NaN\nllamacpp:requests_deferred -1",
                     "llamacpp:requests_processing 2\nllamacpp:requests_processing 3"):
            with self.subTest(text=text):
                self.assertIsNone(probe.parse_metrics(text)["processing"])
                self.assertIsNone(probe.parse_metrics(text)["deferred"])
        self.assertEqual(probe.parse_metrics('llamacpp:requests_processing{model="m"} 6\nllamacpp:requests_deferred 1'),
                         {"processing": 6.0, "deferred": 1.0})
        self.assertEqual(probe.scheduling([], 6)["deferred_evidence"], "UNKNOWN")
        zero: dict[str, object] = {"counters": probe.parse_metrics("llamacpp:requests_processing 0\nllamacpp:requests_deferred 0")}
        self.assertEqual(probe.scheduling([zero], 6)["deferred_evidence"], "NOT_OBSERVED")
        self.assertEqual(probe.scheduling([zero], 6)["max_deferred"], 0)
        excessive: dict[str, object] = {"counters": {"processing": 7, "deferred": 1}}
        self.assertEqual(probe.scheduling([excessive], 6)["processing_evidence"], "EXCEEDS_REQUESTED")

    def test_all_prepared_calls_reach_barrier_metrics_are_observed_and_sampler_stops(self):
        items = [prepared(f"request-{index}", 8000) for index in range(5)]
        all_running = threading.Barrier(5)
        observed = threading.Event()
        ready = threading.Event()
        calls: list[str] = []
        lock = threading.Lock()
        checkpoints: list[dict[str, object]] = []
        metadata_paths: list[str] = []

        def infer(item: probe.Prepared) -> ApiResult:
            with lock:
                calls.append(item.case_id)
                if len(calls) == len(items):
                    ready.set()
            all_running.wait(timeout=2)
            if not observed.wait(2):
                raise AssertionError("metrics never sampled concurrent calls")
            return result(item)

        def read_metadata(path: str) -> dict[str, object]:
            metadata_paths.append(path)
            if path == "metrics":
                if not ready.wait(2):
                    raise AssertionError("not all prepared clients reached inference")
                # Explicit scheduling fixture, not inferred from client count.
                for_thread: dict[str, object] = {"counters": {"processing": 4, "deferred": 1}}
                observed.set()
                return for_thread
            return {"path": "props", "body_utf8": "{}"}

        group = probe.concurrent_group(items, 4, 600, infer, read_metadata, checkpoints.append)
        self.assertEqual(set(calls), {item.case_id for item in items})
        self.assertEqual(len(calls), 5)
        self.assertTrue(group["complete"])
        self.assertEqual(cast(dict[str, object], group["summary"])["attempted"], 5)
        self.assertEqual(cast(dict[str, object], group["scheduling"])["deferred_evidence"], "OBSERVED")
        self.assertEqual(metadata_paths[0], "props")
        self.assertEqual(metadata_paths[-1], "props")
        self.assertEqual(metadata_paths.count("props"), 2)
        self.assertFalse(any(thread.name == "capacity-metrics" for thread in threading.enumerate()))
        self.assertEqual(len(checkpoints), 5)

    def test_unexpected_worker_failure_still_joins_sampler(self):
        items = [prepared(str(index)) for index in range(5)]

        def infer(item: probe.Prepared) -> ApiResult:
            raise RuntimeError("offline programming failure")

        with self.assertRaisesRegex(RuntimeError, "programming failure"):
            probe.concurrent_group(items, 4, 600, infer,
                                   lambda path: {"counters": {"processing": None, "deferred": None}},
                                   lambda group: None)
        self.assertFalse(any(thread.name == "capacity-metrics" for thread in threading.enumerate()))


class MetadataDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.stop = threading.Event()
        self.metrics_started = threading.Event()
        stop, metrics_started = self.stop, self.metrics_started

        class SlowMetadata(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(503 if self.path == "/metrics" else 200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                if self.path == "/metrics":
                    metrics_started.set()
                try:
                    self.wfile.write(b"llamacpp:requests_processing 6\nllamacpp:requests_deferred 1\n")
                    self.wfile.flush()
                    # Frequent bytes defeat a per-read socket timeout. End
                    # after 3s so the original implementation fails finitely.
                    for _ in range(60):
                        if stop.wait(0.05):
                            break
                        self.wfile.write(b"# drip\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, format: str, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SlowMetadata)
        self.thread = threading.Thread(target=self.server.serve_forever, name="offline-metadata-http")
        self.thread.start()
        self.client = ChatClient(f"http://127.0.0.1:{self.server.server_port}/v1", "", 600)

    def tearDown(self):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_real_slow_drip_success_and_http_error_bodies_have_absolute_deadlines(self):
        for path, status in (("props", 200), ("metrics", 503)):
            with self.subTest(path=path):
                started = time.monotonic()
                snapshot = probe.metadata(self.client, path)
                self.assertLess(time.monotonic() - started, 2.5)
                self.assertEqual(snapshot["http_status"], status)
                self.assertIsNotNone(snapshot["error"])
                self.assertEqual(snapshot["subprocess_returncode"], 28)
                self.assertIn("llamacpp:requests_processing", cast(str, snapshot["body_utf8"]))
                self.assertEqual(snapshot["counters"], {"processing": None, "deferred": None})

    def test_real_inflight_drip_cannot_strand_sampler_join_after_model_futures_finish(self):
        items = [prepared(str(index)) for index in range(5)]

        def infer(item: probe.Prepared) -> ApiResult:
            if not self.metrics_started.wait(3):
                raise AssertionError("controlled metadata request did not start")
            return result(item)

        started = time.monotonic()
        group = probe.concurrent_group(
            items, 4, 600, infer, lambda path: probe.metadata(self.client, path),
            lambda checkpoint: None,
        )
        self.assertLess(time.monotonic() - started, 5.5)
        self.assertTrue(group["complete"])
        self.assertTrue(group["metrics"])
        self.assertEqual(cast(dict[str, object], group["scheduling"])["deferred_evidence"], "UNKNOWN")
        self.assertFalse(any(thread.name == "capacity-metrics" for thread in threading.enumerate()))
        for snapshot in cast(list[dict[str, object]], group["metrics"]):
            self.assertIsNotNone(snapshot["error"])
            self.assertIsNotNone(snapshot["subprocess_returncode"])


if __name__ == "__main__":
    unittest.main()
