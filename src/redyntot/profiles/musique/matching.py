"""Syntactic fingerprint extraction and exemplar ranking for musique."""
import json
import re
from collections import defaultdict

def parse_fp_str(fp_str):
    parts = {'wh': [], 'raw_paths': [], 'deps': [], 'pos': []}
    for segment in fp_str.split(';'):
        if '=' not in segment:
            continue
        k, v = segment.split('=', 1)
        if k == 'WH':
            parts['wh'] = [w for w in v.split('|') if w]
        elif k == 'PATH':
            for p in v.split('|'):
                if p:
                    parts['raw_paths'].append(p.split('→'))
        elif k == 'DEPS':
            parts['deps'] = [d for d in v.split('_') if d]
        elif k == 'POS':
            parts['pos'] = [p for p in v.split('_') if p]
    return parts

def build_index(examples):
    parsed = []
    index_by_wh = defaultdict(list)
    for i, ex in enumerate(examples):
        fps = ex['fingerprint']
        fp = {'wh': sum((f['wh'] for f in fps), []), 'raw_paths': sum((f['raw_paths'] for f in fps), []), 'deps': sum((f['deps'] for f in fps), []), 'pos': sum((f['pos'] for f in fps), [])}
        parsed.append({'question': ex['question'], 'decomposition': ex['decomposition'], 'fp': fp})
        for w in fp['wh']:
            index_by_wh[w].append(i)
    return (parsed, index_by_wh)

def fingerprint_one(doc):
    wh_tokens = [tok for tok in doc if tok.lower_ in {'when', 'where', 'who', 'what', 'which', 'how'}]
    cond_tokens = wh_tokens[:]
    for tok in doc:
        if tok.dep_ == 'relcl':
            cond_tokens.append(tok)
            for sib in tok.children:
                if sib.dep_ == 'conj':
                    cond_tokens.append(sib)
    root = next((tok for tok in doc if tok.dep_ == 'ROOT'), None)
    if root:
        cond_tokens.append(root)
        for sib in root.children:
            if sib.dep_ == 'conj':
                cond_tokens.append(sib)
    raw_paths = []
    if root:
        for t0 in cond_tokens:
            path = []
            t = t0
            steps = 0
            while t.dep_ != 'ROOT' and steps < 50:
                path.append(t.dep_)
                t = t.head
                steps += 1
            if path:
                raw_paths.append(path)
    deps = [tok.dep_ for tok in doc]
    poses = [tok.pos_ for tok in doc]
    clause_types = {'advcl', 'ccomp', 'xcomp', 'relcl'}
    clause_counts = {f'{typ}_count': 0 for typ in clause_types}
    clause_flags = {f'has_{typ}': False for typ in clause_types}
    for tok in doc:
        if tok.dep_ in clause_types:
            clause_counts[f'{tok.dep_}_count'] += 1
            clause_flags[f'has_{tok.dep_}'] = True
    all_deps_in_paths = [dep for path in raw_paths for dep in path]
    has_prep = 'prep' in all_deps_in_paths
    has_mark = 'mark' in all_deps_in_paths
    prep_count = all_deps_in_paths.count('prep')
    mark_count = all_deps_in_paths.count('mark')
    return {'wh': [w.lower_ for w in wh_tokens], 'raw_paths': raw_paths, 'deps': deps, 'pos': poses, **clause_counts, **clause_flags, 'has_prep': has_prep, 'has_mark': has_mark, 'prep_count': prep_count, 'mark_count': mark_count}

def fingerprint_aggregate(text):
    text = re.sub('(?<=[\\.\\?\\!])(?=[A-Z])', ' ', text)
    doc = nlp(text)
    return [fingerprint_one(nlp(sent.text.strip())) for sent in doc.sents]

def jaccard(a, b):
    sa, sb = (set(a), set(b))
    if not sa and (not sb):
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def path_score(np, ep):
    if not np and (not ep):
        return 1.0
    if not np or not ep:
        return 0.0
    scs = []
    for p in np:
        best = 0
        for q in ep:
            best = max(best, jaccard(p, q))
        scs.append(best)
    base = sum(scs) / len(scs)
    mult = min(len(np), len(ep)) / max(len(np), len(ep))
    return base * mult

def score(fp_new, fp_ex, w_wh=0.1, w_path=0.3, w_deps=0.3, w_pos=0.3, w_clause=0.05, w_func=0.05):
    sc_wh = jaccard(fp_new['wh'], fp_ex['wh'])
    sc_path = path_score(fp_new['raw_paths'], fp_ex['raw_paths'])
    sc_deps = jaccard(fp_new['deps'], fp_ex['deps'])
    sc_pos = jaccard(fp_new['pos'], fp_ex['pos'])
    clause_keys = ['has_advcl', 'has_ccomp', 'has_xcomp', 'has_relcl']
    new_clauses = {k for k in clause_keys if fp_new.get(k, False)}
    ex_clauses = {k for k in clause_keys if fp_ex.get(k, False)}
    sc_clause = jaccard(new_clauses, ex_clauses)
    func_keys = ['has_prep', 'has_mark']
    new_funcs = {k for k in func_keys if fp_new.get(k, False)}
    ex_funcs = {k for k in func_keys if fp_ex.get(k, False)}
    sc_func = jaccard(new_funcs, ex_funcs)
    return w_wh * sc_wh + w_path * sc_path + w_deps * sc_deps + w_pos * sc_pos + w_clause * sc_clause + w_func * sc_func

def retrieve_examples(fps_new):
    fp_new = {'wh': sum((f['wh'] for f in fps_new), []), 'raw_paths': sum((f['raw_paths'] for f in fps_new), []), 'deps': sum((f['deps'] for f in fps_new), []), 'pos': sum((f['pos'] for f in fps_new), [])}
    cands = set()
    for w in fp_new['wh']:
        cands |= set(index_by_wh.get(w, []))
    if not cands:
        cands = set(range(len(parsed_examples)))
    scored = [(score(fp_new, parsed_examples[i]['fp']), i) for i in cands]
    scored.sort(reverse=True, key=lambda x: x[0])
    top5 = [parsed_examples[i] for _, i in scored[:REQUESTED_K]]
    for ex in top5:
        pass
    return top5
REQUESTED_K = 5

def configure(examples, parser, k):
    global parsed_examples, index_by_wh, nlp, REQUESTED_K
    if not isinstance(k, int) or k < 1:
        raise ValueError('K must be a positive integer')
    parsed_examples, index_by_wh = build_index(examples)
    nlp, REQUESTED_K = (parser, k)
