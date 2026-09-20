"""Dense passage retrieval using normalized embeddings and FAISS."""
import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class DenseRetriever:

    def __init__(self, model_name: str='sentence-transformers/all-mpnet-base-v2', passages_path: str=None, index_path: str='data/faiss_index.idx', embeddings_path: str='data/embeddings.npy', rebuild: bool=False):
        if passages_path is None:
            base = os.path.dirname(os.path.dirname(__file__))
            passages_path = os.path.join(base, 'data', 'musique_wikipedia.json')
        assert os.path.exists(passages_path), f'无法找到段落文件：{passages_path}'
        self.index_path = os.path.join(os.getcwd(), index_path)
        self.emb_path = os.path.join(os.getcwd(), embeddings_path)
        self.passages = self._load_passages(passages_path)
        self.model = SentenceTransformer(model_name, local_files_only=True)
        if rebuild or not os.path.exists(self.index_path) or (not os.path.exists(self.emb_path)):
            self._build_index()
        else:
            self._load_index()

    def _load_passages(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [{'id': p['id'], 'text': p['text'], 'title': p.get('title', '')} for p in data]

    def _build_index(self):
        os.makedirs(os.path.dirname(self.emb_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        texts = [p['text'] for p in self.passages]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        np.save(self.emb_path, embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)
        self.index = index
        self.embeddings = embeddings

    def _load_index(self):
        self.embeddings = np.load(self.emb_path)
        self.index = faiss.read_index(self.index_path)

    def search(self, query: str, top_k: int=20):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, ids = self.index.search(q_emb, top_k)
        results = []
        for idx, score in zip(ids[0], scores[0]):
            p = self.passages[idx]
            results.append({'id': p['id'], 'title': p['title'], 'text': p['text'], 'score': float(score)})
        return results
