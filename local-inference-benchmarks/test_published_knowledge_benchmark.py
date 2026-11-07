"""Offline stdlib regressions; endpoint I/O is replaced at urllib's boundary."""

from __future__ import annotations

import dataclasses
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from collections import Counter
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import published_knowledge_benchmark as knowledge
from agent_workload_benchmark import set_request_body_extra
from intelligence_challenge_suite import reserve_output, write_json

DATASET_DIR = Path(__file__).parent / "datasets" / "mmlu-pro-32-20261107"


class Response(io.BytesIO):
    status = 200


def response(content: str | None, *, finish: str = "stop", reasoning: str = "") -> Response:
    return Response(json.dumps({
        "choices": [{"finish_reason": finish, "message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }).encode())


class ParserTests(unittest.TestCase):
    def test_finite_positive_grammar(self):
        examples = {
            "A": "A", " b \n": "B", "( c )": "C", "J.": "J", "(D).": "D",
            "Answer: E": "E", "answer:f.": "F", "The correct answer is (g).": "G",
            "OPTION h": "H", '{ "answer": "i" }': "I",
            "```\nA\n```": "A", '```json\n{"answer":"B"}\n```': "B",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(knowledge.parse_choice(text).choice, expected)

    def test_ambiguous_alternatives_prose_and_malformed_forms_are_invalid(self):
        invalid = (
            "", "K", "A or B", "Answer: B because it fits", "A\nB", "A/B", "A, B",
            "The correct answer is B. Therefore B", "I think B", "(A", "B)", "B..",
            "Option: B", "Answer B", "B: explanation", "<think>A</think>B", "**B**",
            '{"answer":"B","reason":"x"}', '{"answer":"B","answer":"B"}',
            '{"answer":"B","answer":"C"}', '{"answer":["B"]}', '{"answer":" B "}',
            '["B"]', '"B"', '{"answer":true}', '{"answer":"B or C"}',
            "```python\nB\n```", "```json\nB\n``` extra", "```\n```\nB\n```\n```",
            "B" * 65537,
        )
        for text in invalid:
            with self.subTest(text=text[:100]):
                self.assertIsNone(knowledge.parse_choice(text).choice)

    def test_bounded_deep_untrusted_json_is_invalid_not_a_crash(self):
        nested = '{"answer":' + "[" * 2000 + '"B"' + "]" * 2000 + "}"
        self.assertIsNone(knowledge.parse_choice(nested).choice)
        with patch.object(knowledge.json, "loads", side_effect=RecursionError("nested")):
            self.assertEqual(knowledge.parse_choice('{"answer":"B"}').reason, "invalid_json")


class DatasetTests(unittest.TestCase):
    def test_fixed_dataset_hashes_schema_and_balanced_categories(self):
        dataset = knowledge.load_dataset(DATASET_DIR)
        self.assertEqual(len(dataset.questions), 32)
        self.assertEqual(Counter(q.category for q in dataset.questions), dict.fromkeys(knowledge.CATEGORIES, 4))
        self.assertEqual(dataset.hashes, {kind: declared for kind, (_, declared) in knowledge.FILES.items()})
        for question in dataset.questions:
            gold = dataset.gold[question.case_id]
            index = gold["answer_index"]
            assert type(index) is int
            self.assertEqual(gold["answer"], chr(65 + index))

    def test_native_bytes_not_reserialized_json_are_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name, _ in knowledge.FILES.values():
                (target / name).write_bytes((DATASET_DIR / name).read_bytes())
            prompt_path = target / knowledge.FILES["prompts"][0]
            prompt_path.write_bytes(prompt_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "native-byte SHA256"):
                knowledge.load_dataset(target)

    def test_schema_join_and_answer_index_consistency_fail_closed(self):
        dataset = knowledge.load_dataset(DATASET_DIR)
        prompts = json.loads((DATASET_DIR / knowledge.FILES["prompts"][0]).read_bytes())
        gold = json.loads((DATASET_DIR / knowledge.FILES["gold"][0]).read_bytes())
        mutations = (
            ("prompt extra field", lambda p, g, m: p[0].update(answer="B")),
            ("prompt duplicate", lambda p, g, m: p.__setitem__(1, p[0])),
            ("question duplicate", lambda p, g, m: p[1].update(question_id=p[0]["question_id"])),
            ("gold duplicate", lambda p, g, m: g.__setitem__(1, g[0])),
            ("gold mismatch", lambda p, g, m: g[0].update(question="different")),
            ("gold index", lambda p, g, m: g[0].update(answer_index=0)),
            ("bool index", lambda p, g, m: g[0].update(answer_index=True)),
            ("empty source", lambda p, g, m: g[0].update(src="")),
            ("missing case", lambda p, g, m: g.pop()),
            ("declared hash", lambda p, g, m: m["files"].update(prompts_sha256="wrong")),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                p, g, m = json.loads(json.dumps([prompts, gold, dataset.metadata]))
                mutate(p, g, m)
                with self.assertRaises(ValueError):
                    knowledge.validate_records(p, g, m, dataset.hashes)

    def test_selection_uses_only_case_identity_and_fixed_order(self):
        dataset = knowledge.load_dataset(DATASET_DIR)
        first = dataset.questions[:3]
        self.assertEqual(knowledge.select_questions(dataset.questions, "", 3), first)
        self.assertEqual(knowledge.select_questions(dataset.questions, "mmlu-pro-003,mmlu-pro-001", None), (first[0], first[2]))
        for cases, limit in (("unknown", None), ("", 0), ("", 33)):
            with self.subTest(cases=cases, limit=limit), self.assertRaises(ValueError):
                knowledge.select_questions(dataset.questions, cases, limit)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_request_body_extra, {})
        set_request_body_extra({})
        self.dataset = knowledge.load_dataset(DATASET_DIR)

    def argv(self, output: Path, *, limit: int = 3, extra: dict[str, object] | None = None) -> list[str]:
        return [
            "published_knowledge_benchmark.py", "--dataset-dir", str(DATASET_DIR),
            "--base-url", "http://127.0.0.1/v1", "--model", "candidate",
            "--output", str(output), "--artifacts-dir", str(output.parent / "raw"),
            "--limit", str(limit), "--extra-body-json", json.dumps(extra or {}),
        ]

    def test_gold_teacher_fields_never_reach_actual_serialized_request(self):
        gold = {key: dict(value) for key, value in self.dataset.gold.items()}
        marker = "WATERMARK_TEACHER_RATIONALE_NEVER_SEND_71"
        for row in gold.values():
            row["cot_content"] = marker
            row["src"] = "WATERMARK_SOURCE_NEVER_SEND_92"
        dataset = dataclasses.replace(self.dataset, gold=gold)
        profile = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05, "chat_template_kwargs": {"enable_thinking": True}}
        captured = []

        def endpoint(request: urllib.request.Request, *, timeout: float) -> Response:
            self.assertEqual(timeout, 600)
            assert isinstance(request.data, bytes)
            payload = json.loads(request.data)
            captured.append(payload)
            question = dataset.questions[len(captured) - 1]
            return response(str(gold[question.case_id]["answer"]), reasoning="retained native thinking")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge.json"
            with (
                patch("sys.argv", self.argv(output, extra=profile)),
                patch.object(knowledge, "load_dataset", return_value=dataset),
                patch.object(knowledge.os.environ, "get", return_value="fixture-private-key"),
                patch("urllib.request.urlopen", side_effect=endpoint),
                patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertEqual(knowledge.main(), 0)
            self.assertEqual(len(captured), 3)
            for question, payload in zip(dataset.questions, captured):
                self.assertEqual(payload["messages"], knowledge.build_messages(question))
                encoded = json.dumps(payload)
                for forbidden in (marker, "WATERMARK_SOURCE_NEVER_SEND_92", "answer_index", "cot_content", "row_idx"):
                    self.assertNotIn(forbidden, encoded)
                self.assertNotIn("tools", payload)
                self.assertNotIn("tool_choice", payload)
                self.assertNotIn("stream", payload)
                self.assertEqual(payload["max_tokens"], 8192)
                for key, value in profile.items():
                    self.assertEqual(payload[key], value)
            report = json.loads(output.read_bytes())
            self.assertTrue(report["complete"])
            self.assertEqual(report["categories"]["all"]["passed"], 3)
            manifest = json.loads(output.with_suffix(".manifest.json").read_bytes())
            self.assertEqual(manifest["final_answer_parser"], knowledge.PARSER_GRAMMAR)
            self.assertEqual(manifest["gold_for_scoring_only"][0]["cot_content"], marker)
            self.assertEqual(manifest["native_profile"], profile)
            self.assertTrue(output.with_suffix(".run-owner.json").exists())
            for row in report["cases"]:
                artifact = json.loads(Path(row["calls"][0]["artifact"]).read_bytes())
                self.assertEqual(artifact["request"], row["request"])
                self.assertEqual(artifact["response"]["choices"][0]["message"]["reasoning_content"], "retained native thinking")
                self.assertIsNotNone(artifact["response_body_base64"])
                self.assertEqual(artifact["http_status"], 200)
                self.assertGreater(artifact["wall_seconds"], 0)
                self.assertIsNone(artifact["ttft_seconds"])

    def test_wrong_http_truncated_ambiguous_and_abstention_keep_denominator(self):
        first = self.dataset.questions[0]
        first_gold = self.dataset.gold[first.case_id]["answer"]
        second_gold = self.dataset.gold[self.dataset.questions[1].case_id]["answer"]
        wrong = "A" if second_gold != "A" else "B"
        bodies = [
            response(str(first_gold)), response(wrong),
            urllib.error.HTTPError("http://127.0.0.1/v1", 503, "unavailable", Message(), io.BytesIO(b"backend unavailable")),
            response(str(self.dataset.gold[self.dataset.questions[3].case_id]["answer"]), finish="length"),
            response("A or B"), response("", reasoning=str(self.dataset.gold[self.dataset.questions[5].case_id]["answer"])),
        ]
        checkpoints = []

        def persist(path: Path, value: object, *, api_key: str = "") -> None:
            if path.name == "knowledge.json":
                checkpoints.append(json.loads(json.dumps(value)))
            write_json(path, value, api_key=api_key)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge.json"
            with (
                patch("sys.argv", self.argv(output, limit=6)),
                patch.object(knowledge.os.environ, "get", return_value="fixture-private-key"),
                patch("urllib.request.urlopen", side_effect=bodies),
                patch.object(knowledge, "write_json", side_effect=persist),
                patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertEqual(knowledge.main(), 0)
            report = json.loads(output.read_bytes())
            summary = report["categories"]["all"]
            self.assertEqual((summary["passed"], summary["failed"], summary["total"]), (1, 5, 6))
            self.assertEqual(summary["request_errors"], 1)
            self.assertEqual(summary["length_finishes"], 1)
            self.assertEqual(summary["requests_with_usage"], 5)
            self.assertEqual(report["categories"]["math"]["total"], 4)
            self.assertEqual(len(report["cases"]), 6)
            self.assertEqual(report["cases"][3]["parse_reason"], "finish_length")
            self.assertIsNone(report["cases"][3]["parsed_choice"])
            self.assertIsNone(report["cases"][5]["parsed_choice"])
            self.assertEqual(len(list((output.parent / "raw").glob("*.json"))), 6)
            self.assertEqual(len(checkpoints), 7)
            for index, checkpoint in enumerate(checkpoints):
                self.assertEqual(checkpoint["completed"], index)
                self.assertEqual(checkpoint["complete"], index == 6)
                self.assertEqual(checkpoint["categories"]["all"]["total"], 6)
                self.assertEqual(checkpoint["categories"]["all"]["unattempted"], 6 - index)
            self.assertIsNone(checkpoints[0]["categories"]["all"]["inference_seconds_total"])

    def test_native_whitelist_accepts_qualified_profiles_and_blocks_override_keys(self):
        profile = {
            "temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05,
            "presence_penalty": 0.1, "frequency_penalty": 0.2, "repeat_penalty": 1.05,
            "repetition_penalty": 1.07, "seed": 20261107, "chat_template_kwargs": {"enable_thinking": True},
        }
        self.assertEqual(knowledge.native_profile(json.dumps(profile)), profile)
        for key in ("model", "messages", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens", "n", "stop"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                knowledge.native_profile(json.dumps({key: "forbidden"}))
        with self.assertRaises(ValueError):
            knowledge.native_profile('{"temperature":NaN}')

    def test_shared_namespace_reservation_is_used_and_never_clobbers_owner(self):
        self.assertIs(knowledge.reserve_output, reserve_output)
        self.assertIs(knowledge.write_json, write_json)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge.json"
            collision = output.with_suffix(".manifest.json")
            collision.write_bytes(b"prior evidence\n")
            with (
                patch("sys.argv", self.argv(output)),
                patch.object(knowledge.os.environ, "get", return_value="fixture-private-key"),
                patch("urllib.request.urlopen") as endpoint,
                patch("sys.stderr", new=io.StringIO()),
            ):
                self.assertEqual(knowledge.main(), 2)
            endpoint.assert_not_called()
            self.assertEqual(collision.read_bytes(), b"prior evidence\n")
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".run-owner.json").exists())


if __name__ == "__main__":
    unittest.main()
