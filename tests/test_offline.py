"""Offline regression checks; no remote model or retrieval service is contacted."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types
import io as stdio

from redyntot import metrics, runtime, io


class OfflineChecks(unittest.TestCase):
    def test_openai_transport_without_network(self):
        from redyntot.openai_client import ChatClient
        recorded = []
        def response(request, timeout):
            recorded.append(json.loads(request.data))
            body = {'model': 'served-test-model', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'answer'}, 'logprobs': {'content': [{'token': 'answer', 'logprob': -.1}]}}], 'usage': {'total_tokens': 3}}
            return stdio.BytesIO(json.dumps(body).encode())
        with patch.dict('os.environ', {'REDYNTOT_CHAT_URL': 'https://example.invalid/chat/completions'}), patch('urllib.request.urlopen', side_effect=response):
            client = ChatClient('requested-test-model')
            choice = client.complete('system prompt', 'user prompt', stop=['\n\n'])
            self.assertEqual(recorded[0]['model'], 'requested-test-model')
            self.assertEqual(recorded[0]['messages'][0], {'role': 'system', 'content': 'system prompt'})
            self.assertTrue(recorded[0]['logprobs'])
            self.assertEqual(recorded[0]['temperature'], 0)
            self.assertEqual(recorded[0]['stop'], ['\n\n'])
            self.assertEqual(client.calls[0]['returned_model'], 'served-test-model')
            self.assertEqual(choice['logprobs']['content'][0]['logprob'], -.1)

    def test_sparse_query_fields_match_each_profile(self):
        from redyntot.sparse import SparseRetriever
        class Connection:
            def __init__(self, *args, **kwargs):
                self.request = None
            def search(self, **kwargs):
                self.request = kwargs
                return {'hits': {'hits': [{'_source': {'title': 'T', 'text': 'Evidence'}, '_score': 2.0}]}}
        fake = types.ModuleType('elasticsearch')
        fake.Elasticsearch = Connection
        with patch.dict('sys.modules', {'elasticsearch': fake}):
            for profile in runtime.PROFILES:
                cfg = io.load_json(runtime.PACKAGE_ROOT / 'configs' / (profile + '.json'))
                retriever = SparseRetriever(cfg['es_index'], cfg['bm25_fields'])
                hits = retriever.search('Question?', k=50)
                expected = ['title^1.25', 'title_unescape^1.25', 'text', 'text_bigram'] if profile == 'musique' else ['title^1.25', 'text']
                self.assertEqual(retriever.connection.request['query']['multi_match']['fields'], expected)
                self.assertEqual(retriever.connection.request['size'], 50)
                self.assertEqual(hits[0]['text'], 'Evidence')

    def test_initial_generation_and_reflection_offline(self):
        class FakeClient:
            def __init__(self, model):
                self.calls, self.failures = [], []
            def complete(self, system, user, **kwargs):
                text = json.dumps({'Root?': ['First?', 'Second about #1?']})
                lp = -.2 if not self.calls else -.01
                self.calls.append({'system': system, 'user': user})
                return {'message': {'content': text}, 'logprobs': {'content': [{'token': c, 'logprob': lp} for c in text]}}
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, 'ChatClient', FakeClient):
            source, output = Path(directory) / 'prompts.jsonl', Path(directory) / 'output.jsonl'
            io.save_rows(source, [{'id': 'offline', 'question': 'Root?', 'prompt': 'Q: Root?\nA:'}])
            for profile in runtime.PROFILES:
                runtime.decompose(source, runtime.PACKAGE_ROOT / 'configs' / (profile + '.json'), output)
                row = io.load_rows(output)[0]
                self.assertEqual(row['status'], 'ok', profile)
                self.assertEqual(row['reconstructions'], 1, profile)
                self.assertEqual(len(row['trace']), 2, profile)

    def test_categorical_scoring(self):
        self.assertEqual(metrics.answer_metrics('yes indeed', 'yes')['f1'], 0)
        self.assertEqual(metrics.answer_metrics('The Eiffel Tower', 'Eiffel Tower')['em'], 1)
        self.assertAlmostEqual(metrics.answer_metrics('red blue', 'blue green')['f1'], .5)

    def test_missing_and_failed_predictions_keep_denominator(self):
        qs = [{'id': str(i), 'question': str(i), 'answer': 'yes'} for i in range(3)]
        predictions = [{'id': '0', 'answer': 'yes'}, {'id': '1', 'answer': 'yes', 'status': 'execution_error'}]
        score = metrics.evaluate(qs, predictions)
        self.assertEqual(score['n'], 3)
        self.assertAlmostEqual(score['em'], 1/3)
        self.assertEqual(score['missing_predictions'], 1)
        with self.assertRaises(ValueError):
            metrics.evaluate(qs, predictions + [predictions[0]])

    def test_threshold_and_structural_triggers(self):
        nodes = {'Root?': {'decomposition': ['A?', 'B?']}}
        self.assertFalse(any(runtime.violations(nodes, [-.107], -.107).values()))
        self.assertTrue(runtime.violations(nodes, [-.108], -.107)['low_confidence'])
        self.assertTrue(runtime.violations({'Root?': {'decomposition': ['Root?']}}, [], -.107)['self_reference'])
        with self.assertRaises(ValueError):
            runtime.reject_cycles('a', {'a': ['b'], 'b': ['a']})

    def test_pool_and_requested_k(self):
        pool = io.load_json(runtime.PACKAGE_ROOT / 'data/exemplar_pool_600.json')
        self.assertEqual(len(pool), 600)
        self.assertEqual(len({r['question'] for r in pool}), 600)
        for profile in runtime.PROFILES:
            matching = runtime.module(profile, 'matching')
            matching.configure(pool, None, 7)
            query = [{'wh': [], 'raw_paths': [], 'deps': [], 'pos': []}]
            self.assertEqual(len(matching.retrieve_examples(query)), 7)
            matching.configure(pool, None, 3)
            self.assertEqual(len(matching.retrieve_examples(query)), 3)

    def test_erqt_order_all_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / 'input.jsonl', Path(temporary) / 'tree.jsonl'
            io.save_rows(source, [{'id': 'synthetic', 'question': 'Root?', 'status': 'ok', 'decomposition': {'Root?': {'decomposition': ['First?', 'Second about #1?']}}}])
            for profile in runtime.PROFILES:
                runtime.prepare_trees(source, profile, output)
                record = io.load_rows(output)[0]
                self.assertEqual(record['status'], 'ok', profile)
                self.assertEqual(record['nodes'][-1]['question'], 'Root?')
                self.assertEqual(record['nodes'][1]['placeholders'], [1])

    def test_mocked_execution_all_profiles(self):
        class Retriever:
            def search(self, *args, **kwargs):
                return [{'title': 'Synthetic', 'text': 'Synthetic evidence', 'score': 1, 'bm25_score': 1}]
        class Reranker:
            def rerank(self, question, hits, top_k):
                return hits[:top_k]
        nodes = [{'root': 'Root?', 'id': 1, 'question': 'First?', 'placeholders': [], 'sons': [], 'group': [1]}, {'root': 'Root?', 'id': 2, 'question': 'Second about #1?', 'placeholders': [1], 'sons': [], 'group': [1]}, {'root': 'Root?', 'id': 3, 'question': 'Root?', 'placeholders': [], 'sons': [1, 2], 'group': [0]}]
        for profile in runtime.PROFILES:
            execution = runtime.module(profile, 'execution')
            execution._dr = execution._bm25 = Retriever()
            execution._rr = Reranker()
            execution.extract_local_answer = lambda *a: ('retrieved', -.2)
            execution.extract_local_answer1 = lambda *a: ('upstream', -.2)
            execution.aggregate_with_children = lambda *a: ('aggregated', -.1)
            answer, trace = execution.process_tree(nodes)
            self.assertEqual(answer[0], 'aggregated', profile)
            self.assertEqual(len(trace), 3)


if __name__ == '__main__':
    unittest.main()
