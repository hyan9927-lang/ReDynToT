"""Cross-encoder scoring and reranking of fused passage candidates."""
from typing import List, Dict
from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self, model_name: str='cross-encoder/ms-marco-MiniLM-L-12-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int=10) -> List[Dict]:
        pairs = [(query, c['text']) for c in candidates]
        scores = self.model.predict(pairs, batch_size=16)
        for c, s in zip(candidates, scores):
            c['rerank_score'] = float(s)
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        return candidates[:top_k]
