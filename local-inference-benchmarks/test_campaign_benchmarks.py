"""Stdlib tests; no endpoint, Docker daemon, or generated host-code execution."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import http.client
import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from collections import Counter
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from agent_workload_benchmark import (
    ApiResult,
    ChatClient,
    request_body,
    set_request_body_extra,
)
from challenge_cases import (
    Case,
    build_cases,
    grade_answer,
    merge_intervals,
    optimal_assignment,
    retrieval_gold,
    shortest_distance,
    strict_equal,
)
from code_sandbox import docker_argv, grade_code, sandbox_command, validate_image
from intelligence_challenge_suite import (
    DepotTask,
    completion,
    main as challenge_main,
    reserve_output,
    run_case,
    summarize,
    write_json,
)

IMAGE = "python@sha256:" + "a" * 64
NAME = "challenge-code-" + "b" * 32


def artifact_path(result: ApiResult) -> Path:
    assert result.artifact is not None, "retention must return its evidence path"
    return Path(result.artifact)


class Response(io.BytesIO):
    status = 200


class GoldTests(unittest.TestCase):
    def test_assignment_minimum_and_tie_break(self):
        self.assertEqual(optimal_assignment([[4, 1], [2, 8]]), {"assignment": [1, 0], "cost": 3})
        self.assertEqual(optimal_assignment([[1, 1], [1, 1]]), {"assignment": [0, 1], "cost": 2})

    def test_directed_route_does_not_reverse_edges(self):
        self.assertEqual(shortest_distance([[0, 1, 8], [0, 2, 2], [2, 1, 1], [1, 3, 4], [3, 0, 1]], 0, 3), 7)

    def test_merge_includes_touching_and_negative_intervals(self):
        source = [[3, 4], [-4, -1], [-1, 2], [2, 3], [8, 8]]
        self.assertEqual(merge_intervals(source), [[-4, 4], [8, 8]])
        self.assertEqual(source, [[3, 4], [-4, -1], [-1, 2], [2, 3], [8, 8]])

    def test_retrieval_namespace_numeric_revision_and_abstention(self):
        records = [
            {"tenant": "a", "key": "x", "revision": 9, "active": True, "value": "old"},
            {"tenant": "a", "key": "x", "revision": 12, "active": True, "value": "new"},
            {"tenant": "a", "key": "x", "revision": 99, "active": False, "value": "inactive"},
            {"tenant": "ab", "key": "x", "revision": 100, "active": True, "value": "wrong"},
        ]
        self.assertEqual(retrieval_gold(records, "a", "x"), {"value": "new", "revision": 12})
        self.assertEqual(retrieval_gold(records, "a", "missing"), {"value": None, "revision": None})

    def test_30_cases_are_seeded_and_categories_equal(self):
        first = build_cases(20261107)
        self.assertEqual(first, build_cases(20261107))
        self.assertNotEqual(first, build_cases(20261108))
        self.assertEqual(len({c.name for c in first}), 30)
        self.assertEqual(Counter(c.category for c in first), dict.fromkeys(("reasoning", "code", "tools", "structured", "retrieval"), 6))
        for case in first:
            if case.category != "code":
                text = case.expected if isinstance(case.expected, str) else json.dumps(case.expected)
                self.assertTrue(grade_answer(text, case.expected), case.name)

    def test_json_types_keys_and_output_contract_are_strict(self):
        self.assertFalse(strict_equal(True, 1))
        self.assertFalse(grade_answer('{"n":true}', {"n": 1}))
        self.assertFalse(grade_answer('{"n":0,"n":1}', {"n": 1}))
        self.assertFalse(grade_answer('```json\n{"n":1}\n```', {"n": 1}))
        self.assertFalse(grade_answer('{"n":1,"extra":0}', {"n": 1}))
        self.assertTrue(grade_answer('{ "nested": [false, null, 0] }', {"nested": [False, None, 0]}))


class NativeProfileTests(unittest.TestCase):
    def test_native_sampling_profile_reaches_actual_payload_and_retained_manifest(self):
        self.addCleanup(set_request_body_extra, {})
        set_request_body_extra({})
        self.assertEqual(request_body("model", [], 8192, 0)["top_k"], 1)
        case = next(case for case in build_cases(20261107) if case.name == "stock_ledger")
        minimal = {"temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True}}
        full = {
            **minimal, "top_k": 64, "min_p": 0.05, "presence_penalty": 0.1,
            "frequency_penalty": 0.2, "repeat_penalty": 1.05,
            "repetition_penalty": 1.07, "seed": 123,
        }
        captured: list[dict[str, object]] = []
        for index, profile in enumerate((minimal, full)):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / f"native-{index}.json"
                artifacts = Path(directory) / "raw"
                captured.clear()

                def endpoint(request: urllib.request.Request, *, timeout: float) -> Response:
                    data = request.data
                    assert isinstance(data, bytes)
                    self.assertEqual(timeout, 600)
                    captured.append(json.loads(data))
                    return Response(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(case.expected)}}]}).encode())

                argv = [
                    "intelligence_challenge_suite.py", "--base-url", "http://127.0.0.1/v1",
                    "--model", "native-candidate", "--output", str(output),
                    "--artifacts-dir", str(artifacts), "--cases", case.name,
                    "--extra-body-json", json.dumps(profile),
                ]
                with (
                    patch("sys.argv", argv),
                    patch("intelligence_challenge_suite.os.environ.get", return_value="fixture-private-key"),
                    patch("urllib.request.urlopen", side_effect=endpoint),
                    patch("sys.stdout", new=io.StringIO()),
                ):
                    self.assertEqual(challenge_main(), 0)
                self.assertEqual(len(captured), 1)
                payload = captured[0]
                for key, value in profile.items():
                    self.assertEqual(payload[key], value, key)
                if "top_k" not in profile:
                    self.assertNotIn("top_k", payload)
                self.assertEqual(payload["max_tokens"], 8192)
                self.assertEqual(payload["model"], "native-candidate")
                self.assertEqual(payload["messages"], [{"role": "user", "content": case.prompt}])
                manifest = json.loads(output.with_suffix(".manifest.json").read_text())
                self.assertEqual(manifest["temperature"], profile["temperature"])
                self.assertEqual(manifest["extra_body"], profile)
                records = list(artifacts.glob("*.json"))
                self.assertEqual(len(records), 1)
                self.assertEqual(json.loads(records[0].read_text())["request"], payload)
                self.assertTrue(json.loads(output.read_text())["cases"][0]["passed"])

    def test_native_profile_cannot_override_budgets_tools_or_request_shape(self):
        self.addCleanup(set_request_body_extra, {})
        forbidden = {
            "model": "other", "messages": [], "stream": True,
            "max_tokens": 1, "max_completion_tokens": 1, "max_output_tokens": 1,
            "measurement_timeout": 1, "tools": [], "tool_choice": "none",
            "stop": ["END"], "n": 2, "response_format": {"type": "json_object"},
        }
        with tempfile.TemporaryDirectory() as directory:
            for index, (key, value) in enumerate(forbidden.items()):
                with self.subTest(key=key):
                    output = Path(directory) / f"blocked-{index}.json"
                    errors = io.StringIO()
                    argv = [
                        "intelligence_challenge_suite.py", "--base-url", "http://127.0.0.1/v1",
                        "--model", "candidate", "--output", str(output),
                        "--artifacts-dir", str(Path(directory) / "raw"),
                        "--cases", "stock_ledger", "--extra-body-json", json.dumps({key: value}),
                    ]
                    with (
                        patch("sys.argv", argv),
                        patch("urllib.request.urlopen") as transport,
                        patch("sys.stderr", new=errors),
                    ):
                        self.assertEqual(challenge_main(), 2)
                        transport.assert_not_called()
                    self.assertIn(key, errors.getvalue())
                    self.assertFalse(output.exists())
                    self.assertFalse(output.with_suffix(".manifest.json").exists())


class ScoringTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(model="test", max_tokens=8192, docker_command="docker", sandbox_image=IMAGE, sandbox_timeout=8)

    def test_truncation_cannot_pass_even_with_correct_visible_answer(self):
        case = Case("test", "reasoning", "solve", {"n": 7})
        client = ChatClient("http://127.0.0.1/v1", "test-key", 600)
        response = {"choices": [{"finish_reason": "length", "message": {"content": '{"n":7}'}}]}
        with patch.object(client, "chat", return_value=ApiResult(True, 12, response=response)):
            row = run_case(case, client, self.args())
        self.assertFalse(row["passed"])
        self.assertEqual(row["failure"], "finish_length")
        category = summarize([row])["reasoning"]
        self.assertEqual(category["total"], 1)
        self.assertEqual(category["length_finishes"], 1)

    def test_timeout_and_error_remain_in_denominator_and_latency(self):
        client = ChatClient("http://127.0.0.1/v1", "test-key", 600)
        case = Case("test", "reasoning", "solve", {"n": 7})
        good = ApiResult(True, 2, response={"choices": [{"finish_reason": "stop", "message": {"content": '{"n":7}'}}], "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}})
        bad = ApiResult(False, 600, error="TimeoutError: timed out")
        with patch.object(client, "chat", side_effect=[good, bad]):
            rows = [run_case(case, client, self.args()), run_case(case, client, self.args())]
        category = summarize(rows)["reasoning"]
        self.assertEqual((category["passed"], category["total"], category["failed"]), (1, 2, 1))
        self.assertEqual(category["request_errors"], 1)
        self.assertEqual(category["inference_seconds_median"], 301)
        self.assertEqual(category["inference_seconds_p95_nearest_rank"], 600)
        self.assertEqual(category["usage_reported_sum"]["total_tokens"], 14)
        self.assertEqual(category["requests_with_usage"], 1)

    def test_missing_content_or_finish_never_fabricates_pass(self):
        for response in ({}, {"choices": None}, {"choices": [{"finish_reason": "stop", "message": None}]}, {"choices": [{"message": {"content": "ok"}}]}):
            self.assertIsNotNone(completion(ApiResult(True, 1, response=response)).failure)

    def test_tool_goal_accepts_multiple_legal_plans_not_a_hint_sequence(self):
        fixture = next(c.tool_fixture for c in build_cases(3) if c.category == "tools")
        for south_first in (True, False):
            world = DepotTask(fixture)
            names = sorted(world.depots)
            self.assertEqual(world.apply("list_depots", {})["depots"], names)
            north = next(n for n in names if n.startswith("north"))
            south = next(n for n in names if n.startswith("south"))
            for name in (north, south):
                world.apply("inspect_depot", {"name": name})
            transfers = [(south, 1), (north, world.demand - 1)]
            for source, count in transfers if south_first else reversed(transfers):
                world.apply("transfer", {"source": source, "destination": world.target, "count": count})
            self.assertTrue(world.goal_met())

    def test_twentieth_tool_operation_can_complete_but_twenty_first_is_rejected(self):
        case = next(c for c in build_cases(3) if c.category == "tools")
        fixture = json.loads(case.tool_fixture)
        north = next(name for name in fixture["depots"] if name.startswith("north"))
        south = next(name for name in fixture["depots"] if name.startswith("south"))
        target, demand = fixture["target"], fixture["demand"]

        def tool_call(index: int, name: str, arguments: dict[str, object]) -> dict[str, object]:
            return {"id": f"op-{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

        tools = [tool_call(0, "list_depots", {})]
        tools.extend(tool_call(i, "inspect_depot", {"name": north}) for i in range(1, 18))
        tools.extend([
            tool_call(18, "transfer", {"source": north, "destination": target, "count": demand - 1}),
            tool_call(19, "transfer", {"source": south, "destination": target, "count": 1}),
        ])
        first = ApiResult(True, 1, response={"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": tools}}]})
        final = ApiResult(True, 1, response={"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(case.expected)}}]})
        extra = ApiResult(True, 1, response={"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [tool_call(20, "transfer", {"source": target, "destination": south, "count": 1})]}}]})
        client = ChatClient("http://127.0.0.1/v1", "test-key", 600)
        with patch.object(client, "chat", side_effect=[first, final]):
            passed = run_case(case, client, self.args())
        self.assertTrue(passed["passed"])
        events = passed["tool_events"]
        assert isinstance(events, list)
        self.assertEqual(len(events), 20)
        with patch.object(client, "chat", side_effect=[first, extra]):
            rejected = run_case(case, client, self.args())
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["failure"], "tool_operation_limit")
        events = rejected["tool_events"]
        assert isinstance(events, list)
        self.assertEqual(len(events), 21)
        self.assertIn("error", events[-1]["result"])
        self.assertEqual(events[-1]["before"], events[-1]["after"])

    def test_tool_invalid_mutation_is_atomic_and_goal_not_just_final_claim(self):
        world = DepotTask(next(c.tool_fixture for c in build_cases(3) if c.category == "tools"))
        before = world.snapshot()
        locked = next(n for n in world.depots if n.startswith("quarantine"))
        self.assertIn("error", world.apply("transfer", {"source": locked, "destination": world.target, "count": 1}))
        self.assertEqual(before, world.snapshot())
        self.assertFalse(world.goal_met())
        self.assertIn("error", world.apply("transfer", {"source": locked, "destination": world.target, "count": True}))


class ReportPersistenceTests(unittest.TestCase):
    def test_complete_report_copy_redacts_without_changing_scored_response(self):
        key = "sample-private-key"
        client = ChatClient("http://127.0.0.1/v1", key, 600)
        response = {"choices": [{"finish_reason": "stop", "message": {"content": key}}], "usage": {"echo": key}}
        args = argparse.Namespace(model="test", max_tokens=8192)
        with patch.object(client, "chat", return_value=ApiResult(True, 1, response=response)):
            row = run_case(Case("echo", "structured", "return the fixture", key), client, args)
        self.assertTrue(row["passed"])
        report = {
            "cases": [row], "metadata": {"echo": key},
            "tool_events": [{"arguments": r'{"echo":"sample\u002dprivate\u002dkey"}'}],
        }
        before = json.dumps(report, sort_keys=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json(path, report, api_key=key)
            persisted = json.loads(path.read_text())
        self.assertEqual(json.dumps(report, sort_keys=True), before)
        self.assertEqual(row["answer"], key)
        self.assertTrue(row["passed"])
        self.assertNotIn(key, json.dumps(persisted))
        self.assertEqual(persisted["cases"][0]["answer"], "[REDACTED_API_KEY]")
        self.assertEqual(persisted["cases"][0]["calls"][0]["usage"]["echo"], "[REDACTED_API_KEY]")
        self.assertEqual(json.loads(persisted["tool_events"][0]["arguments"])["echo"], "[REDACTED_API_KEY]")

    def test_two_writers_cannot_claim_or_overwrite_the_same_fresh_run(self):
        barrier = threading.Barrier(2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "campaign.json"

            def attempt(writer: str) -> str:
                barrier.wait(timeout=5)
                try:
                    reserve_output(output)
                except FileExistsError:
                    return "rejected"
                write_json(output.with_suffix(".manifest.json"), {"writer": writer})
                write_json(output, {"writer": writer, "complete": True})
                return writer

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(attempt, ("one", "two")))
            self.assertEqual(outcomes.count("rejected"), 1)
            winner = next(value for value in outcomes if value != "rejected")
            self.assertEqual(json.loads(output.read_text())["writer"], winner)
            self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text())["writer"], winner)
            paths = (output, output.with_suffix(".manifest.json"), output.with_suffix(".run-owner.json"))
            before = [path.read_bytes() for path in paths]
            with self.assertRaises(FileExistsError):
                reserve_output(output)
            self.assertEqual([path.read_bytes() for path in paths], before)

    def test_report_manifest_cross_role_collision_preserves_owner_and_rolls_back_only_loser(self):
        for first_name, second_name in (
            ("a.json", "a.manifest.json"), ("a.manifest.json", "a.json"),
        ):
            with self.subTest(first=first_name), tempfile.TemporaryDirectory() as directory:
                first, second = Path(directory) / first_name, Path(directory) / second_name
                reserve_output(first)
                write_json(first, {"owner": first_name, "complete": True})
                write_json(first.with_suffix(".manifest.json"), {"owner": first_name})
                owned = {first, first.with_suffix(".manifest.json"), first.with_suffix(".run-owner.json")}
                before = {path: path.read_bytes() for path in owned}
                with self.assertRaises(FileExistsError):
                    reserve_output(second)
                self.assertEqual({path: path.read_bytes() for path in owned}, before)
                for path in {second, second.with_suffix(".manifest.json"), second.with_suffix(".run-owner.json")} - owned:
                    self.assertFalse(path.exists(), str(path))


class RetentionTests(unittest.TestCase):
    def test_full_payload_response_reasoning_usage_and_safe_timestamps(self):
        key = "sample-private-key"
        raw = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "answer" * 2000, "reasoning_content": "reasoning"}}], "usage": {"prompt_tokens": 42, "completion_tokens": 9}}).encode()
        body = request_body("model", [{"role": "user", "content": "full prompt"}], 8192, 0)
        with tempfile.TemporaryDirectory() as directory:
            client = ChatClient("http://127.0.0.1/v1", key, 600, artifacts_dir=Path(directory), label="label", nonce="run")
            with patch("urllib.request.urlopen", return_value=Response(raw)):
                result = client.chat(body, case_id="case/turn-1")
            self.assertTrue(result.ok)
            record_text = artifact_path(result).read_text()
            record = json.loads(record_text)
        self.assertEqual(record["request"], body)
        self.assertEqual(json.loads(record["request_body_utf8"]), body)
        self.assertEqual(base64.b64decode(record["response_body_base64"]), raw)
        self.assertEqual(record["response_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(record["finish_reasons"], ["stop"])
        self.assertEqual(record["usage"]["completion_tokens"], 9)
        self.assertEqual((record["case_id"], record["label"], record["nonce"]), ("case/turn-1", "label", "run"))
        self.assertIn("utc", record["started"])
        self.assertIn("et", record["ended"])
        self.assertIsNone(record["ttft_seconds"])
        self.assertNotIn(key, record_text)
        self.assertNotIn("Authorization", record_text)
        self.assertNotIn("environment", record)

    def test_errors_timeout_invalid_json_and_partial_body_are_retained(self):
        raw = b"failure-" * 1000
        failures = [
            urllib.error.HTTPError("http://127.0.0.1/v1/chat/completions", 503, "unavailable", Message(), io.BytesIO(raw)),
            TimeoutError("timed out"),
            http.client.IncompleteRead(b"partial-body", 100),
        ]
        with tempfile.TemporaryDirectory() as directory:
            client = ChatClient("http://127.0.0.1/v1", "sample-private-key", 600, artifacts_dir=Path(directory))
            for failure in failures:
                with patch("urllib.request.urlopen", side_effect=failure):
                    result = client.chat({"model": "m", "messages": []})
                self.assertFalse(result.ok)
                record = json.loads(artifact_path(result).read_text())
                self.assertEqual(record["request"]["model"], "m")
                self.assertTrue(record["error"])
                if result.status == 503:
                    self.assertEqual(base64.b64decode(record["response_body_base64"]), raw)
            with patch("urllib.request.urlopen", return_value=Response(b"not JSON\xff")):
                result = client.chat({"model": "m"})
            self.assertFalse(result.ok)
            self.assertEqual(base64.b64decode(json.loads(artifact_path(result).read_text())["response_body_base64"]), b"not JSON\xff")
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 4)

    def test_real_http_timeout_keeps_returned_body_prefix_for_success_and_error(self):
        prefix = b'{"partial":"' + b"x" * 100
        for status in (200, 503):
            with self.subTest(status=status):
                sent, release = threading.Event(), threading.Event()

                class Handler(BaseHTTPRequestHandler):
                    response_status = status
                    sent_event = sent
                    release_event = release

                    def do_POST(self):
                        self.rfile.read(int(self.headers["Content-Length"]))
                        self.send_response(self.response_status)
                        self.send_header("Content-Length", "10000")
                        self.end_headers()
                        self.wfile.write(prefix)
                        self.wfile.flush()
                        self.sent_event.set()
                        self.release_event.wait(timeout=5)

                    def log_message(self, format: str, *args: object) -> None:
                        pass

                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with tempfile.TemporaryDirectory() as directory:
                        client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "sample-private-key", 0.2, artifacts_dir=Path(directory))
                        result = client.chat({})
                        record = json.loads(artifact_path(result).read_text())
                    self.assertTrue(sent.is_set())
                    self.assertFalse(result.ok)
                    self.assertEqual(result.status, status)
                    self.assertIn("timed out", str(result.error).casefold())
                    self.assertEqual(base64.b64decode(record["response_body_base64"]), prefix)
                    self.assertEqual(record["response_sha256"], hashlib.sha256(prefix).hexdigest())
                    self.assertGreater(result.wall_seconds, 0)
                    self.assertIsNone(record["nonstream_eof_seconds"])
                finally:
                    release.set()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

    def test_real_http_short_declared_body_fails_but_complete_chunked_body_does_not(self):
        raw = b'{"choices":[{"finish_reason":"stop","message":{"content":"correct"}}]}'
        for status, chunked in ((200, False), (503, False), (200, True), (503, True)):
            with self.subTest(status=status, chunked=chunked):
                class Handler(BaseHTTPRequestHandler):
                    protocol_version = "HTTP/1.1"
                    response_status = status
                    use_chunks = chunked

                    def do_POST(self):
                        self.rfile.read(int(self.headers["Content-Length"]))
                        self.send_response(self.response_status)
                        self.send_header("Content-Length", str(len(raw) + 100))
                        if self.use_chunks:
                            self.send_header("Transfer-Encoding", "chunked")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        if self.use_chunks:
                            self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n0\r\n\r\n")
                        else:
                            self.wfile.write(raw)
                        self.wfile.flush()
                        self.close_connection = True

                    def log_message(self, format: str, *args: object) -> None:
                        pass

                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with tempfile.TemporaryDirectory() as directory:
                        client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "sample-private-key", 2, artifacts_dir=Path(directory))
                        result = client.chat({})
                        record = json.loads(artifact_path(result).read_text())
                    self.assertEqual(base64.b64decode(record["response_body_base64"]), raw)
                    self.assertEqual(result.status, status)
                    if not chunked:
                        self.assertFalse(result.ok)
                        self.assertIn("IncompleteRead", str(result.error))
                        self.assertEqual(completion(result).failure, "request_error")
                    else:
                        self.assertEqual(result.ok, status == 200)
                        self.assertNotIn("IncompleteRead", str(result.error))
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

    def test_partial_json_escaped_echo_redacts_without_changing_unrelated_bytes(self):
        key = "sample-private-key"
        escaped = br"sample\u002dprivate\u002dkey"
        partial = b'{"echo":"' + escaped + b'"'
        for suffix in (b"", b"\xff\x80"):
            raw = partial + suffix
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                client = ChatClient("http://127.0.0.1/v1", key, 600, artifacts_dir=Path(directory))
                with patch("urllib.request.urlopen", return_value=Response(raw)):
                    result = client.chat({})
                record = json.loads(artifact_path(result).read_text())
            expected = raw.replace(escaped, b"[REDACTED_API_KEY]")
            self.assertFalse(result.ok)
            self.assertEqual(base64.b64decode(record["response_body_base64"]), expected)
            self.assertEqual(record["response_body_utf8"], expected.decode(errors="replace"))
            self.assertEqual(record["response_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertTrue(record["credential_redaction"])

    def test_unicode_escaped_echo_is_redacted_in_both_wire_views(self):
        key = "sample-private-key"
        raw = br'{"echo":"sample\u002dprivate\u002dkey","choices":[]}'
        with tempfile.TemporaryDirectory() as directory:
            client = ChatClient("http://127.0.0.1/v1", key, 600, artifacts_dir=Path(directory))
            with patch("urllib.request.urlopen", return_value=Response(raw)):
                result = client.chat({})
            record = json.loads(artifact_path(result).read_text())
        safe_bytes = base64.b64decode(record["response_body_base64"])
        self.assertEqual(safe_bytes, record["response_body_utf8"].encode())
        self.assertEqual(json.loads(safe_bytes)["echo"], "[REDACTED_API_KEY]")
        self.assertEqual(record["response"]["echo"], "[REDACTED_API_KEY]")
        self.assertEqual(record["response_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertTrue(record["credential_redaction"])
        assert result.response is not None
        self.assertEqual(result.response["echo"], key)

    def test_concurrent_calls_never_clobber_and_tokenize_has_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ChatClient("http://127.0.0.1/v1", "sample-private-key", 600, artifacts_dir=Path(directory))
            with (
                patch("urllib.request.urlopen", side_effect=lambda *a, **k: Response(b'{"tokens":[1,2,3]}')),
                concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor,
            ):
                self.assertEqual(list(executor.map(client.tokenize, ["fixture"] * 24)), [3] * 24)
            paths = list(Path(directory).glob("*.json"))
            self.assertEqual(len(paths), 24)
            for path in paths:
                record = json.loads(path.read_text())
                self.assertEqual(record["path"], "tokenize")
                self.assertEqual(record["request"], {"content": "fixture"})
                self.assertEqual(record["case_id"], "tokenize_calibration")

    def test_default_has_no_artifact_and_echoed_key_is_redacted(self):
        client = ChatClient("http://127.0.0.1/v1", "sample-private-key", 600)
        with patch("urllib.request.urlopen", return_value=Response(b'{}')):
            self.assertIsNone(client.chat({}).artifact)
        with tempfile.TemporaryDirectory() as directory:
            client.artifacts_dir = Path(directory)
            with patch("urllib.request.urlopen", return_value=Response(b'{"error":"sample-private-key"}')):
                result = client.chat({})
            record = json.loads(artifact_path(result).read_text())
            self.assertNotIn("sample-private-key", json.dumps(record))
            self.assertNotIn(b"sample-private-key", base64.b64decode(record["response_body_base64"]))
            self.assertTrue(record["credential_redaction"])


class SandboxTests(unittest.TestCase):
    def test_only_two_docker_launch_forms_and_immutable_image(self):
        self.assertEqual(docker_argv("docker"), ("docker",))
        self.assertEqual(docker_argv("sudo -n docker"), ("sudo", "-n", "docker"))
        for command in ("sudo docker", "docker --host tcp://example", "docker; true", "sh -c docker", "/usr/bin/docker", "sudo -n docker run"):
            with self.assertRaises(ValueError):
                docker_argv(command)
        for image in ("", "python:3.12-slim", "--privileged", "python@sha256:abc"):
            with self.assertRaises(ValueError):
                validate_image(image)

    def test_timeout_removes_only_its_container_and_keeps_elapsed_failure(self):
        process = Mock()
        process.stdout = io.BytesIO()
        process.poll.return_value = None
        process.wait.return_value = -9
        with (
            patch("code_sandbox.subprocess.Popen", return_value=process) as launch,
            patch("code_sandbox.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stderr=b"")) as cleanup,
            patch("code_sandbox.selectors.DefaultSelector") as selector,
            patch("code_sandbox.time.monotonic", side_effect=[10.0, 11.0, 18.25]),
        ):
            selector.return_value.__enter__.return_value.select.return_value = []
            grade = grade_code("raise RuntimeError('must never execute on host')", (), command=("sudo", "-n", "docker"), image=IMAGE)
        self.assertFalse(grade.passed)
        self.assertEqual(grade.state, "timeout")
        self.assertEqual(grade.elapsed_seconds, 8.25)
        process.kill.assert_called_once()
        argv = launch.call_args.args[0]
        name = argv[argv.index("--name") + 1]
        self.assertRegex(name, r"^challenge-code-[0-9a-f]{32}$")
        self.assertEqual(cleanup.call_args.args[0], ["sudo", "-n", "docker", "rm", "-f", name])

    def test_deep_bounded_stdout_is_invalid_output_not_a_suite_crash(self):
        raw = b"[" * 2000 + b"0" + b"]" * 2000
        read_fd, write_fd = os.pipe()
        with os.fdopen(write_fd, "wb") as writer:
            writer.write(raw)
        process = Mock()
        process.stdout = os.fdopen(read_fd, "rb")
        process.poll.return_value = 0
        process.wait.return_value = 0
        with (
            patch("code_sandbox.subprocess.Popen", return_value=process),
            patch("code_sandbox.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stderr=b"")),
        ):
            grade = grade_code("not executed on the host", (), command=("docker",), image=IMAGE)
        self.assertFalse(grade.passed)
        self.assertEqual(grade.state, "invalid_output")
        self.assertEqual(grade.evidence.encode(), raw)
        self.assertGreater(grade.elapsed_seconds, 0)
        self.assertTrue(process.stdout.closed)

    def test_command_enforces_every_sandbox_constraint(self):
        for command in (("docker",), ("sudo", "-n", "docker")):
            argv = sandbox_command(command, IMAGE, NAME)
            self.assertEqual(argv[:len(command)], list(command))
            required = {"--network": "none", "--user": "65534:65534", "--cap-drop": "ALL", "--security-opt": "no-new-privileges:true", "--memory": "256m", "--memory-swap": "256m", "--cpus": "1", "--pids-limit": "64", "--oom-score-adj": "800", "--pull": "never", "--entrypoint": "python3"}
            for flag, value in required.items():
                self.assertEqual(argv[argv.index(flag) + 1], value)
            self.assertIn("--read-only", argv)
            self.assertIn("--rm", argv)
            self.assertIn("/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777", argv)
            for forbidden in ("--privileged", "--mount", "--volume", "-v", "--pid", "--device", "/var/run/docker.sock", "--gpus"):
                self.assertNotIn(forbidden, argv)
            self.assertEqual(argv[argv.index(IMAGE) + 1:argv.index(IMAGE) + 3], ["-I", "-c"])


if __name__ == "__main__":
    unittest.main()
