"""JSON decomposition parsing and children-token confidence calculation."""
import json
import re

def locate_span(tokens, pattern):
    joined = ''.join(tokens)
    joined_ns = re.sub('\\s+', '', joined)
    pat_ns = re.sub('\\s+', '', pattern)
    idx = joined_ns.find(pat_ns)
    if idx < 0:
        return (None, None)
    consumed = 0
    st = None
    for i, t in enumerate(tokens):
        consumed += len(t.replace(' ', ''))
        if consumed > idx:
            st = i
            break
    if st is None:
        return (None, None)
    for j in range(st, len(tokens)):
        if '[' in tokens[j]:
            st = j
            break
    else:
        return (None, None)
    for j in range(st, len(tokens)):
        if ']' in tokens[j]:
            return (st, j)
    return (None, None)

def compute_decomp(entry):
    prompt = entry['prompt']
    root_q = prompt.splitlines()[-2][3:].strip()
    resp = entry['response'].strip()
    json_part = resp[resp.find('{'):resp.rfind('}') + 1]
    try:
        decomp = json.loads(json_part)
    except Exception as e:
        decomp = {}
    content = entry.get('logprobs', {}).get('content') or []
    tokens = [d['token'] for d in content]
    logps = [d['logprob'] for d in content]
    if tokens and tokens[-1] == '.':
        tokens.pop()
        logps.pop()
    nodes = {}
    avgs = []
    for q, children in decomp.items():
        if isinstance(children, list) and len(children) >= 2:
            pattern = json.dumps(children, ensure_ascii=False)
            st, ed = locate_span(tokens, pattern)
            avg = None
            if st is not None and ed is not None and (ed - st > 1):
                avg = round(sum(logps[st + 1:ed]) / (ed - st - 1), 6)
                avgs.append(avg)
            nodes[q] = {'decomposition': children, 'subq_avg_logprob': avg}
    return (root_q, nodes, avgs)
