"""Dense passage retrieval using normalized embeddings and FAISS."""
import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def stream_passages(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            text = p.get('text') or ' '.join((sent for _, sent_list in p.get('context', []) for sent in sent_list))
            yield {'id': p['id'], 'title': p.get('title', ''), 'text': text}

def chunks(iterator, batch_size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

class DenseRetriever:

    def __init__(self, model_name: str='sentence-transformers/all-mpnet-base-v2', passages_path: str=None, index_path: str='data/faiss_index.idx', embeddings_path: str='data/embeddings.npy', rebuild: bool=False, batch_size: int=1024):
        base = os.path.dirname(os.path.abspath(__file__))
        if passages_path is None:
            passages_path = os.path.join(base, '..', 'data', 'hotpotqa_wikipedia.jsonl')
        assert os.path.exists(passages_path), f'段落文件不存在: {passages_path}'
        self.passages_path = passages_path
        from pathlib import Path
        self._offsets = []
        with open(self.passages_path, 'rb') as _f:
            pos = 0
            for line in _f:
                self._offsets.append(pos)
                pos += len(line)
        self.index_path = os.path.join(base, index_path)
        self.emb_path = os.path.join(base, embeddings_path)
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        if rebuild or not os.path.exists(self.index_path) or (not os.path.exists(self.emb_path)):
            self._build_index()
        else:
            self._load_index()

    def _build_index(self):
        total_items = sum((1 for _ in open(self.passages_path, 'r', encoding='utf-8')))
        total_batches = (total_items + self.batch_size - 1) // self.batch_size
        first_batch = True
        index = None
        processed = 0
        emb_file = None
        for batch in tqdm(chunks(stream_passages(self.passages_path), self.batch_size), total=total_batches, desc='Building Index'):
            texts = [p['text'] for p in batch]
            embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(embs)
            if first_batch:
                dim = embs.shape[1]
                index = faiss.IndexFlatIP(dim)
                emb_file = np.lib.format.open_memmap(self.emb_path, mode='w+', dtype='float32', shape=(total_items, dim))
                first_batch = False
            index.add(embs)
            start = processed
            end = start + embs.shape[0]
            emb_file[start:end, :] = embs
            processed += embs.shape[0]
            if processed % (self.batch_size * 10) == 0:
                pass
        faiss.write_index(index, self.index_path)
        self.index = index

    def _load_index(self):
        self.embeddings = np.load(self.emb_path, mmap_mode='r')
        self.index = faiss.read_index(self.index_path)

    def search(self, query: str, top_k: int=20):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, ids = self.index.search(q_emb, top_k)
        results = []
        with open(self.passages_path, 'rb') as f:
            for idx, score in zip(ids[0], scores[0]):
                f.seek(self._offsets[idx])
                raw = f.readline().decode('utf-8')
                p = json.loads(raw)
                text = p.get('text') or ' '.join((sent for _, sent_list in p.get('context', []) for sent in sent_list))
                results.append({'id': p['id'], 'title': p.get('title', ''), 'text': text, 'score': float(score)})
        return results
