import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from agent_workload_benchmark import ChatClient
from generative_benchmark import (
    CANDIDATE_STATUS,
    candidate_rows,
    candidate_variables,
    context_settings,
    digest,
    json_candidate,
    main,
    render_candidate,
    rescore_run,
    run_row,
    summarize,
)


class Handler(BaseHTTPRequestHandler):
    response: ClassVar[dict] = {}
    status = 200
    requests: ClassVar[list] = []
    props: ClassVar[dict] = {}
    props_status = 200
    get_paths: ClassVar[list] = []

    def do_GET(self):
        self.get_paths.append(self.path)
        data = json.dumps(self.props).encode()
        self.send_response(self.props_status)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.requests.append(body)
        data = json.dumps(self.response).encode()
        self.send_response(self.status)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


def reply(content, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': json.dumps(content)}}]}


def naming():
    return {'id': 'on-01', 'family': 'ontology_naming', 'messages': [], 'rules': {
        'required_name_constraints': {'max_chars': 60, 'max_words': 6, 'must_match_any': ['gift card'], 'must_not_match': []},
        'definition_constraints': {'max_chars': 400}, 'must_distinguish_from': [], 'existing_reference': {},
        'required_definition_facts': [{'fact': 'balance', 'any_of': ['balance']}], 'forbidden': []}}


def summary():
    return {'id': 'hs-01', 'family': 'hub_summary', 'messages': [], 'rules': {
        'member_nodes': [1, 2], 'citeable_assessments': {'1': [11], '2': [22]},
        'forbidden_citations': {'members': [], 'assessments': []}, 'required_facts': [],
        'required_fact_patterns': {}, 'uncertainty_must_survive': [], 'contested_pairs': [],
        'forbidden_claims': []}}


class GenerativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Handler.status = 200
        Handler.requests = []
        Handler.props = {'default_generation_settings': {'n_ctx': 262144}}
        Handler.props_status = 200
        Handler.get_paths = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = ChatClient(f'http://127.0.0.1:{self.server.server_port}/v1', 'test-key', 5, artifacts_dir=Path(self.tmp.name))

    def run_case(self, row, context_tokens=131072):
        result = run_row(row, self.client, 'test-model', {}, 100, context_tokens)
        self.assertEqual(len(Handler.requests), 1)
        self.assertEqual(Handler.requests[0]['temperature'], 0)
        self.assertEqual(Handler.requests[0]['seed'], 0)
        return result

    def test_pass_and_extra_key_rejection(self):
        Handler.response = reply({'name': 'Gift Card', 'definition': 'A card with a balance.'})
        self.assertTrue(self.run_case(naming())['passed'])
        Handler.requests = []
        Handler.response = reply({'name': 'Gift Card', 'definition': 'A balance.', 'extra': 'no'})
        result = self.run_case(naming())
        self.assertFalse(result['passed'])
        self.assertTrue(result['shape_failure'])

    def test_native_json_candidate_fences_plain_and_real_failure_modes(self):
        good = json.dumps({'name': 'Gift Card', 'definition': 'A card with a balance.'})
        extra = json.dumps({'name': 'Gift Card', 'definition': 'A balance.', 'extra': 'no'})
        for content, passed in (
            ('```json\n' + good + '\n```', True),
            (good, True),
            ('Here you go: ' + good + ' — done', True),
            ('Here is "the answer": ' + good, False),
            ('```json\n' + extra + '\n```', False),
            ('```\nnot JSON\n```\n```json\n' + good + '\n```', False),
        ):
            with self.subTest(content=content):
                Handler.requests = []
                Handler.response = {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
                result = self.run_case(naming())
                self.assertEqual(result['passed'], passed)
                self.assertEqual(result['shape_failure'], not passed)

    def test_length_is_terminal(self):
        Handler.response = reply({'name': 'Gift Card', 'definition': 'A balance.'}, 'length')
        result = self.run_case(naming())
        self.assertTrue(result['length'])
        self.assertFalse(result['passed'])

    def test_summary_order_and_coverage_fail(self):
        Handler.response = reply({'sections': [{'text': 'Fact', 'members': [2, 1], 'assessments': [22, 11]}], 'uncertainty': [], 'conflicts': []})
        result = self.run_case(summary())
        self.assertFalse(result['checks']['citation_order'])
        Handler.requests = []
        Handler.response = reply({'sections': [{'text': 'Fact', 'members': [1], 'assessments': [11]}], 'uncertainty': [], 'conflicts': []})
        result = self.run_case(summary())
        self.assertFalse(result['checks']['member_coverage'])

    def test_oversize_error_retained(self):
        Handler.status = 400
        Handler.response = {'error': {'type': 'context_length_exceeded', 'message': 'too large'}}
        row = summary()
        row['id'] = 'hs-13'
        result = self.run_case(row)
        self.assertTrue(result['passed'])
        self.assertEqual((result['status'], result['error_type'], result['error_message']), (400, 'context_length_exceeded', 'too large'))
        self.assertTrue(Path(result['artifact']).exists())
        self.assertTrue(result['diagnostics']['zagdb_detector_compatible'])

    def test_llama_cpp_context_refusal_is_pass_with_incompatible_detector(self):
        Handler.status = 400
        Handler.response = {'error': {
            'code': 400, 'type': 'exceed_context_size_error',
            'message': 'request (237064 tokens) exceeds the available context size (131072 tokens)...',
            'n_prompt_tokens': 237064, 'n_ctx': 131072,
        }}
        rows = []
        for variant in ('UNVERIFIED_PROPOSED_EVAL_ONLY', CANDIDATE_STATUS):
            Handler.requests = []
            row = summary()
            row.update(id='hs-13', prompt_variant=variant)
            result = self.run_case(row)
            self.assertTrue(result['checks']['typed_oversize_refusal'])
            self.assertTrue(result['passed'])
            self.assertFalse(result['diagnostics']['zagdb_detector_compatible'])
            artifact = json.loads(Path(result['artifact']).read_text())
            self.assertEqual(json.loads(artifact['response_body_utf8']), Handler.response)
            rows.append(result)
        groups = summarize(rows, 1)
        self.assertEqual(len(groups), 2)
        for group in groups.values():
            self.assertEqual(group['passed'], 1)
            self.assertEqual(group['zagdb_detector_compatible_count'], 0)
            self.assertEqual(group['zagdb_detector_checked_count'], 1)

    def test_oversize_refusal_requires_4xx_known_type_and_no_output(self):
        row = summary()
        row['id'] = 'hs-13'
        for status, payload in (
            (500, {'error': {'type': 'context_length_exceeded'}}),
            (400, {'error': {'type': 'other_error', 'message': 'context too large'}}),
            (400, {'error': {'type': 'exceed_context_size_error'}, **reply('generated')}),
            (400, {'error': {'type': 'exceed_context_size_error'}, 'content': 'generated'}),
        ):
            with self.subTest(status=status, payload=payload):
                Handler.status, Handler.response, Handler.requests = status, payload, []
                result = self.run_case(row)
                self.assertFalse(result['checks']['typed_oversize_refusal'])
                self.assertFalse(result['passed'])

    def test_missing_variables_fail_without_http(self):
        row = summary()
        row.update(prompt_variant=CANDIDATE_STATUS, missing_variables=['level'])
        result = run_row(row, self.client, 'test-model', {}, 100, 131072)
        self.assertEqual(Handler.requests, [])
        self.assertEqual(result['reason'], 'missing_variable')
        self.assertEqual(result['missing_variables'], ['level'])
        self.assertFalse(result['passed'])

    def test_candidate_and_proposal_scores_remain_separate(self):
        Handler.response = reply({'sections': [{'text': 'Fact', 'members': [1, 2], 'assessments': [11, 22]}], 'uncertainty': [], 'conflicts': []})
        proposal = summary()
        proposal['prompt_variant'] = 'UNVERIFIED_PROPOSED_EVAL_ONLY'
        candidate = {**proposal, 'prompt_variant': CANDIDATE_STATUS}
        results = [run_row(row, self.client, 'test-model', {}, 100, 131072) for row in (proposal, candidate)]
        groups = summarize(results, 1)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(group['rows'] == 1 and group['passed'] == 1 for group in groups.values()))

    def test_accepted_fitting_trial_is_excluded_with_content_diagnostics(self):
        Handler.response = reply({'sections': [{'text': 'Incomplete', 'members': [1], 'assessments': [11]}], 'uncertainty': [], 'conflicts': []})
        Handler.response['usage'] = {'prompt_tokens': 237064}
        row = summary()
        row.update(id='hs-13', prompt_variant=CANDIDATE_STATUS)
        accepted = self.run_case(row, 262144)
        self.assertEqual(accepted['outcome'], 'not_oversize_for_config')
        self.assertTrue(accepted['excluded_from_scoring'])
        self.assertIsNone(accepted['passed'])
        self.assertFalse(accepted['diagnostics']['content_checks']['member_coverage'])
        self.assertFalse(accepted['diagnostics']['content_passed'])
        Handler.requests = []
        Handler.response = reply({'sections': [{'text': 'Complete', 'members': [1, 2], 'assessments': [11, 22]}], 'uncertainty': [], 'conflicts': []})
        row['id'] = 'hs-01'
        scored = self.run_case(row, 262144)
        group = next(iter(summarize([accepted, scored], 1).values()))
        self.assertEqual((group['rows'], group['scored_rows'], group['passed'], group['failed']), (2, 1, 1, 0))
        self.assertEqual(group['pass_rate'], 1)
        self.assertEqual(group['not_oversize_for_config_count'], 1)
        excluded_only = next(iter(summarize([accepted], 1).values()))
        self.assertIsNone(excluded_only['pass_rate'])

    def test_accepted_at_context_or_truncated_trial_fails(self):
        row = summary()
        row['id'] = 'hs-13'
        for usage, truncation in (
            ({'prompt_tokens': 131072}, {}),
            ({'prompt_tokens': 237064}, {}),
            ({'prompt_tokens': 100}, {'__verbose': {'truncated': True}}),
            ({'prompt_tokens': 100}, {'truncated': True}),
            ({}, {}),
        ):
            with self.subTest(usage=usage, truncation=truncation):
                Handler.requests = []
                Handler.response = {**reply({'sections': [{'text': 'Fact', 'members': [1, 2], 'assessments': []}], 'uncertainty': [], 'conflicts': []}),
                                    'usage': usage, **truncation}
                result = self.run_case(row)
                self.assertFalse(result['passed'])
                self.assertFalse(result['excluded_from_scoring'])
                self.assertEqual(result['outcome'], 'invalid_oversize_acceptance')

    def test_context_discovery_and_explicit_fallback(self):
        settings = context_settings(self.client, 999)
        self.assertEqual((settings['n_ctx'], settings['source']), (262144, 'props.default_generation_settings.n_ctx'))
        Handler.props = {'n_ctx': 131072}
        settings = context_settings(self.client, None)
        self.assertEqual((settings['n_ctx'], settings['source']), (131072, 'props.n_ctx'))
        Handler.props_status = 404
        settings = context_settings(self.client, 65536)
        self.assertEqual((settings['n_ctx'], settings['source']), (65536, '--context-tokens'))
        with self.assertRaisesRegex(ValueError, 'supply --context-tokens'):
            context_settings(self.client, None)

    def test_cli_65536_budget_reaches_wire_and_props_checked_once(self):
        root = Path(self.tmp.name)
        output = root / 'report.json'
        Handler.response = reply({'name': 'Gift Card', 'definition': 'A card with a balance.'})
        argv = ['generative_benchmark.py', '--fixtures', str(root), '--templates', str(root / 'templates.json'),
                '--base-url', self.client.base_url, '--model', 'test-model', '--max-tokens', '65536',
                '--output', str(output), '--artifacts-dir', str(root / 'artifacts'),
                '--api-key-env', 'GENERATIVE_TEST_API_KEY']
        with patch('sys.argv', argv), patch.dict('os.environ', {'GENERATIVE_TEST_API_KEY': 'test-key'}), \
                patch('generative_benchmark.load_rows', return_value=[naming(), {**naming(), 'id': 'on-02'}]), \
                patch('generative_benchmark.candidate_rows', return_value=[]):
            self.assertEqual(main(), 0)
        self.assertEqual(Handler.get_paths, ['/props'])
        self.assertEqual(len(Handler.requests), 2)
        self.assertTrue(all(body['max_tokens'] == 65536 for body in Handler.requests))
        report = json.loads(output.read_text())
        self.assertEqual(report['context']['n_ctx'], 262144)
        self.assertEqual(report['context']['source'], 'props.default_generation_settings.n_ctx')
        self.assertTrue(all(row['passed'] for row in report['rows']))
        Handler.props_status, Handler.get_paths, Handler.requests = 404, [], []
        output = root / 'fallback.json'
        argv[argv.index('--output') + 1] = str(output)
        argv.extend(['--context-tokens', '131072'])
        with patch('sys.argv', argv), patch.dict('os.environ', {'GENERATIVE_TEST_API_KEY': 'test-key'}), \
                patch('generative_benchmark.load_rows', return_value=[naming()]), \
                patch('generative_benchmark.candidate_rows', return_value=[]):
            self.assertEqual(main(), 0)
        report = json.loads(output.read_text())
        self.assertEqual(Handler.get_paths, ['/props'])
        self.assertEqual(Handler.requests[0]['max_tokens'], 65536)
        self.assertEqual(report['context']['n_ctx'], 131072)
        self.assertEqual(report['context']['source'], '--context-tokens')

    def test_cli_missing_context_fails_before_generation(self):
        root = Path(self.tmp.name)
        Handler.props_status = 404
        output = root / 'unclaimed.json'
        argv = ['generative_benchmark.py', '--fixtures', str(root), '--templates', str(root / 'templates.json'),
                '--base-url', self.client.base_url, '--model', 'test-model',
                '--output', str(output), '--artifacts-dir', str(root / 'artifacts'),
                '--api-key-env', 'GENERATIVE_TEST_API_KEY']
        with patch('sys.argv', argv), patch.dict('os.environ', {'GENERATIVE_TEST_API_KEY': 'test-key'}), \
                patch('generative_benchmark.load_rows', return_value=[naming()]), \
                patch('generative_benchmark.candidate_rows', return_value=[]), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(), 2)
        self.assertEqual(Handler.requests, [])
        self.assertFalse(output.exists())

    def test_cli_rescores_raw_artifact_without_http_and_preserves_old_report(self):
        root = Path(self.tmp.name)
        content = '```json\n' + json.dumps({'name': 'Gift Card', 'definition': 'A card with a balance.'}) + '\n```'
        Handler.response = {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
        row = {**naming(), 'prompt_variant': 'VERIFIED_VERBATIM'}
        previous = self.run_case(row)
        # Simulate the former strict parser's failure, not a new model attempt.
        previous.update(passed=False, shape_failure=True, checks={'shape': False})
        original = {'version': 'zagdb-generative-v3', 'model': 'test-model',
                    'context': {'n_ctx': 131072, 'source': 'props.n_ctx'},
                    'rows': [previous], 'summary': summarize([previous], 2.5)}
        original_path = root / 'report.json'
        original_path.write_text(json.dumps(original))
        original_bytes = original_path.read_bytes()
        artifact_path = Path(previous['artifact'])
        retained = json.loads(artifact_path.read_text())
        retained['response'] = reply({'name': 'Wrong', 'definition': 'Wrong'})
        artifact_path.write_text(json.dumps(retained))
        artifact_bytes = artifact_path.read_bytes()
        argv = ['generative_benchmark.py', '--rescore', str(root), '--fixtures', str(root),
                '--templates', str(root / 'templates.json')]
        with patch('sys.argv', argv), patch('generative_benchmark.load_rows', return_value=[row]), \
                patch('generative_benchmark.candidate_rows', return_value=[]), \
                patch('urllib.request.urlopen', side_effect=AssertionError('HTTP forbidden')) as network, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        network.assert_not_called()
        rescored = json.loads((root / 'report.rescored.json').read_text())
        self.assertTrue(rescored['rows'][0]['passed'])
        self.assertEqual(rescored['rescore_http_calls'], 0)
        self.assertEqual(rescored['context'], original['context'])
        self.assertEqual(next(iter(rescored['summary'].values()))['wall_seconds'], 2.5)
        self.assertEqual(original_path.read_bytes(), original_bytes)
        self.assertEqual(artifact_path.read_bytes(), artifact_bytes)
        with self.assertRaises(FileExistsError):
            rescore_run(root, [row], None, None)
        # Old reports without context require an explicit offline value.
        original.pop('context')
        original_path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'supply --context-tokens'):
            rescore_run(root, [row], None, root / 'older.rescored.json')
        destination = rescore_run(root, [row], 262144, root / 'older.rescored.json')
        self.assertEqual(json.loads(destination.read_text())['context']['n_ctx'], 262144)


class JsonCandidateTests(unittest.TestCase):
    def test_source_examples_and_first_opener_behavior(self):
        for reply_text, expected in (
            ('```json\n[1,2]\n```', '[1,2]'),
            ('Here you go: [1,2] — done', '[1,2]'),
            ('"a summary"', '"a summary"'),
            ('Sure! {"Add":{"fact":"x"}}', '{"Add":{"fact":"x"}}'),
            ('prefix "quote" before {"name":"x"}', '"quote" before {"name":"x"'),
            ('```[1,2]```', '[1,2]'),
            ('```json\n[1,2]', '[1,2]'),
        ):
            with self.subTest(reply_text=reply_text):
                self.assertEqual(json_candidate(reply_text), expected)


class CandidateRenderTests(unittest.TestCase):
    def test_mapping_preserves_members_and_revision_keyed_map(self):
        members = [{'node': 1, 'state': 'current', 'assessments': [{'revision': 11, 'subject': '{{ level }} <&>'}]},
                   {'node': 2, 'state': 'inactive_unknown'}]
        variables, missing = candidate_variables({'community': {'community': 5, 'level': 0}, 'members': members})
        self.assertEqual(missing, [])
        self.assertEqual(json.loads(variables['members']), members)
        self.assertEqual(json.loads(variables['assessments']), {'11': members[0]['assessments'][0]})
        rendered = render_candidate('{{ community }}\\n{{ level }}\\n{{ members }}\\n{{ assessments }}', variables)
        self.assertTrue(rendered.startswith('5\\n0\\n'))
        self.assertIn('{{ level }} <&>', rendered)

    def test_missing_fields_are_not_invented(self):
        _, missing = candidate_variables({'community': {'community': 1}, 'members': [{'node': 1, 'state': 'current'}]})
        self.assertEqual(missing, ['level', 'assessments:members[0]'])

    def test_unsupported_tera_is_rejected(self):
        for template in ('{% if level %}x{% endif %}', '{{ members | safe }}', '{{ absent }}'):
            with self.assertRaisesRegex(ValueError, 'unsupported Tera'):
                render_candidate(template, {})

    def test_hash_checked_before_loading_and_missing_rows_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / 'hub_summary@1.tera'
            template.write_text('{{ level }}')
            (root / 'hub_summary.jsonl').write_text(json.dumps({'id': 'hs-01', 'input': {'community': {'community': 1}, 'members': []}, 'checks': {}}))
            with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
                candidate_rows(root, template)
            with patch('generative_benchmark.CANDIDATE_SHA256', digest('{{ level }}')):
                rows = candidate_rows(root, template)
            self.assertEqual(rows[0]['missing_variables'], ['level'])
            self.assertEqual(rows[0]['messages'], [])

if __name__ == '__main__':
    unittest.main()
