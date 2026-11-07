import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from agent_workload_benchmark import ChatClient
from generative_benchmark import run_row


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


if __name__ == '__main__':
    unittest.main()
