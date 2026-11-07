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
    digest,
    render_candidate,
    run_row,
    summarize,
)


class Handler(BaseHTTPRequestHandler):
    response: ClassVar[dict] = {}
    status = 200
    requests: ClassVar[list] = []

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
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = ChatClient(f'http://127.0.0.1:{self.server.server_port}/v1', 'test-key', 5, artifacts_dir=Path(self.tmp.name))

    def run_case(self, row):
        result = run_row(row, self.client, 'test-model', {}, 100)
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

    def test_missing_variables_fail_without_http(self):
        row = summary()
        row.update(prompt_variant=CANDIDATE_STATUS, missing_variables=['level'])
        result = run_row(row, self.client, 'test-model', {}, 100)
        self.assertEqual(Handler.requests, [])
        self.assertEqual(result['reason'], 'missing_variable')
        self.assertEqual(result['missing_variables'], ['level'])
        self.assertFalse(result['passed'])

    def test_candidate_and_proposal_scores_remain_separate(self):
        Handler.response = reply({'sections': [{'text': 'Fact', 'members': [1, 2], 'assessments': [11, 22]}], 'uncertainty': [], 'conflicts': []})
        proposal = summary()
        proposal['prompt_variant'] = 'UNVERIFIED_PROPOSED_EVAL_ONLY'
        candidate = {**proposal, 'prompt_variant': CANDIDATE_STATUS}
        results = [run_row(row, self.client, 'test-model', {}, 100) for row in (proposal, candidate)]
        groups = summarize(results, 1)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(group['rows'] == 1 and group['passed'] == 1 for group in groups.values()))


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
