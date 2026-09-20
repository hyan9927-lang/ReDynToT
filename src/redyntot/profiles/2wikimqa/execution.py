"""Dependency-preserving retrieval fusion, node answering, and arbitration."""
EVIDENCE_PRINT_K = 10
SNIPPET_LEN = 5000
INCLUDE_RERANKED_EVIDENCE = True
RUN_DENSE_ONLY = False
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
RUN_BM25_ONLY = False
AGGREGATION_SYS = 'You are an expert QA assistant. You’ll receive:\n1) Evidence: sub‑question → answer pairs (one per line).\n2) A parent question.\n\nYour task:\n- For extraction questions, return the minimal answer phrase from evidence.\n- For comparative questions (contains “earlier”/“later”/“before”/“after”/“older”/“younger”/“first”), parse and compare the two values, then return only the entity name or phrase that meets the comparison.\nDo not add explanations, dates, or outside knowledge; only output the concise phrase.'

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

def format_hit(hit, idx=None, snippet_len=SNIPPET_LEN):
    t = hit.get('title', '').strip()
    txt = (hit.get('text', '') or hit.get('body', '') or '').strip()
    snip = txt[:snippet_len].replace('\n', ' ')
    prefix = f'[{idx}] ' if idx is not None else ''
    return f'{prefix}Title: {t} — {snip}...'

def guess_used_evidence(final_answer: str, hits: list, snippet_len=SNIPPET_LEN, top_k=EVIDENCE_PRINT_K):
    fa = (final_answer or '').strip().lower()
    if not fa:
        return []
    matches = []
    for i, h in enumerate(hits[:TOP_K_RERANK], 1):
        title = (h.get('title') or '').strip().lower()
        text = (h.get('text') or h.get('body') or '').strip().lower()
        if not title and (not text):
            continue
        if fa == title or fa in title or fa in text:
            matches.append(format_hit(h, idx=i, snippet_len=snippet_len))
        if len(matches) >= top_k:
            break
    return matches

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
        if RUN_BM25_ONLY:
            dense_hits = []
            bm25_hits = _bm25.search(qtext, k=TOP_K_BM25)
        elif RUN_DENSE_ONLY:
            dense_hits = _dr.search(qtext, top_k=TOP_K_DENSE)
            bm25_hits = []
        else:
            dense_hits = _dr.search(qtext, top_k=TOP_K_DENSE)
            bm25_hits = _bm25.search(qtext, k=TOP_K_BM25)
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
        dense_only_pretty = [format_hit(h, idx=i, snippet_len=SNIPPET_LEN) for i, h in enumerate(dense_hits[:EVIDENCE_PRINT_K], 1)]
        bm25_only_pretty = [format_hit(h, idx=i, snippet_len=SNIPPET_LEN) for i, h in enumerate(bm25_hits[:EVIDENCE_PRINT_K], 1)]
        node_trace['dense_only_evidence'] = dense_only_pretty
        node_trace['bm25_only_evidence'] = bm25_only_pretty
        t1 = time.time()
        reranked = _rr.rerank(qtext, all_hits, top_k=TOP_K_RERANK)
        for i, h in enumerate(reranked, 1):
            pass
        if INCLUDE_RERANKED_EVIDENCE:
            reranked_pretty = [format_hit(h, idx=i, snippet_len=SNIPPET_LEN) for i, h in enumerate(reranked[:EVIDENCE_PRINT_K], 1)]
            node_trace['reranked_evidence'] = reranked_pretty
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
            if ('no' in last_ctx_line or 'not' in last_ctx_line) and ('answer' in last_ctx_line or 'information' in last_ctx_line or 'available' in last_ctx_line):
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
        try:
            node_trace['used_evidence_guess'] = guess_used_evidence(ans1, reranked, snippet_len=SNIPPET_LEN)
        except Exception:
            node_trace['used_evidence_guess'] = []
        trace.append(node_trace)
    last_id = nodes[-1]['id']
    return (answers[last_id], trace)
