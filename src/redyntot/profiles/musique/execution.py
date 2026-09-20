"""Dependency-preserving retrieval fusion, node answering, and arbitration."""
SHOW_TOPK_BM25_LOG = 0
BM25_TOPK = 0
SHOW_TOPK_DENSE_LOG = 0
SHOW_TOPK_RERANK_LOG = 10
SHOW_FULL_PASSAGE = True
SNIPPET_CHARS = 600
import os
import re
import time
from typing import List, Dict, Tuple
from collections import defaultdict
MODEL_NAME = ''
LLM_URL = ''
requests = None
TOP_K_DENSE = 50
TOP_K_BM25 = 50
TOP_K_RERANK = 10
EPSILON = 0.001
BM25_ONLY = False
AGGREGATION_SYS = 'You are an expert QA assistant. You will be given:\n"\n                    1) Evidence: sub‑question → answer pairs (one per line).\n"\n                    2) A parent question to answer using that evidence.\n\n"\n                    Your task:\n"\n                    - Extract the concise answer span from the evidence.\n"\n                    - Preserve all qualifiers, except for county names, where you should return only the county itself.\n"\n                    - Do NOT add explanations, rephrasings, or outside knowledge.\n"\n                    - Return only the minimal answer itself.\n\n'

def _score_of(hit):
    return hit.get('rerank_score') or hit.get('score') or hit.get('_score') or hit.get('bm25_score') or hit.get('sim') or hit.get('similarity') or None

def _text_of(hit):
    txt = hit.get('text') or hit.get('content') or ''
    if not SHOW_FULL_PASSAGE and len(txt) > SNIPPET_CHARS:
        return txt[:SNIPPET_CHARS] + '…'
    return txt

def pretty_log_hits(label: str, hits: list, top_k: int):
    k = min(top_k, len(hits))
    if k == 0:
        return
    for i, h in enumerate(hits[:k], 1):
        title = h.get('title', '(no title)')
        score = _score_of(h)
        score_str = f'{score:.4f}' if isinstance(score, (int, float)) else str(score)
        if score is not None:
            pass

def fill_placeholders(question: str, placeholders: list, answers: dict) -> str:
    counter = 0

    def repl(m):
        nonlocal counter
        if counter < len(placeholders):
            target_id = placeholders[counter]
            replacement = answers.get(target_id, ('', ''))[0]
            counter += 1
            return replacement
        return m.group(0)
    return re.sub('(#+\\d+)', repl, question)

def aggregate_with_children(prompt: str) -> Tuple[str, float]:
    payload = {'model': MODEL_NAME, 'messages': [{'role': 'system', 'content': AGGREGATION_SYS}, {'role': 'user', 'content': prompt}], 'temperature': 0, 'max_tokens': 256, 'logprobs': True}
    resp = requests.post(LLM_URL, json=payload).json()
    choice = resp['choices'][0]
    ans = choice['message']['content'].strip()
    lps = choice['logprobs']['content']
    return (ans, sum((d['logprob'] for d in lps)) / len(lps))

def make_id(hit):
    text_snippet = hit.get('text', '')[:50]
    return (hit.get('title', ''), text_snippet)

def process_tree(nodes: list) -> Tuple[Tuple[str, float], list]:
    answers = {}
    trace = []
    for node in nodes:
        node_trace = {'id': node['id'], 'raw_question': node['question']}
        qtext = node['question']
        if node['placeholders']:
            filled = fill_placeholders(qtext, node['placeholders'], answers)
            node_trace['filled_question'] = filled
            qtext = filled
        t0 = time.time()
        if BM25_ONLY:
            dense_hits = []
            bm25_hits = _bm25.search(qtext, k=BM25_TOPK)
        else:
            dense_k = max(1, TOP_K_DENSE)
            dense_hits = _dr.search(qtext, top_k=dense_k)
            bm25_hits = _bm25.search(qtext, k=TOP_K_BM25)
        pretty_log_hits('Dense 检索结果（原始顺序）', dense_hits, SHOW_TOPK_DENSE_LOG)
        pretty_log_hits('BM25 检索结果（原始顺序）', bm25_hits, SHOW_TOPK_BM25_LOG)
        seen = set((make_id(h) for h in dense_hits))
        extra = []
        for h in bm25_hits:
            hid = make_id(h)
            if hid not in seen:
                seen.add(hid)
                extra.append(h)
        all_hits = dense_hits + extra
        for i, h in enumerate(all_hits[:5], 1):
            pass
        if not all_hits:
            reranked = []
        else:
            t1 = time.time()
            rerank_k = min(TOP_K_RERANK, len(all_hits))
            reranked = _rr.rerank(qtext, all_hits, top_k=rerank_k)
            for i, h in enumerate(reranked, 1):
                pass
        pretty_log_hits('Cross-Encoder 重排结果（高→低）', reranked, SHOW_TOPK_RERANK_LOG)
        if node['placeholders']:
            upstream_lines = []
            for pid in node['placeholders']:
                child = next((n for n in nodes if n['id'] == pid))
                filled_q = fill_placeholders(child['question'], child['placeholders'], answers)
                upstream_lines.append(f'- {filled_q} → {answers[pid][0]}')
            upstream_ctx = '\n'.join(upstream_lines)
            ans1, score1 = extract_local_answer1(upstream_ctx, qtext, reranked)
        else:
            ans1, score1 = extract_local_answer(qtext, reranked)
        node_trace['retrieval_answer'] = ans1
        node_trace['retrieval_score'] = score1
        if node['sons']:
            ans_retr, score_retr = (ans1, score1)
            child_ctx_lines = []
            for cid in node['sons']:
                child_node = next((n for n in nodes if n['id'] == cid))
                filled_q = fill_placeholders(child_node['question'], child_node['placeholders'], answers)
                child_ctx_lines.append(f'- {filled_q} → {answers[cid][0]}')
            node_trace['agg_context'] = child_ctx_lines
            child_ctx = '\n'.join(child_ctx_lines)
            parent_filled = fill_placeholders(node['question'], node['placeholders'], answers)
            node_trace['agg_filled_parent'] = parent_filled
            agg_input = 'Evidence:\n' + child_ctx + '\n\nQuestion:\n' + parent_filled + '\n\nAnswer:'
            node_trace['agg_prompt'] = agg_input
            ans_agg, score_agg = aggregate_with_children(agg_input)
            node_trace['agg_answer'] = ans_agg
            node_trace['agg_score'] = score_agg
            last_ctx_line = child_ctx_lines[-1].lower()
            if ('no' in last_ctx_line or 'not' in last_ctx_line) and ('answer' in last_ctx_line or 'information' in last_ctx_line):
                ans1, score1 = (ans_retr, score_retr)
            else:
                penalized_retr_score = score_retr - EPSILON
                if penalized_retr_score > score_agg:
                    ans1, score1 = (ans_retr, score_retr)
                else:
                    ans1, score1 = (ans_agg, score_agg)
        answers[node['id']] = (ans1, score1)
        node_trace['final_answer'] = ans1
        node_trace['final_score'] = score1
        trace.append(node_trace)
    last_id = nodes[-1]['id']
    return (answers[last_id], trace)
