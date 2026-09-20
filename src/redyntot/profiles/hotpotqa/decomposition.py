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

def fix_llm_json_key(s: str) -> str:
    if not s or '{' not in s or '}' not in s:
        return s
    s = s.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\xa0', ' ').replace('\u2009', ' ').replace('\u202f', ' ')
    s = s.replace('“', '"').replace('”', '"').replace('„', '"').replace('‟', '"').replace('＂', '"')
    s = re.sub('"\\s+:', '":', s)

    def first_top_level_colon(txt: str) -> int:
        in_str = False
        esc = False
        for i, ch in enumerate(txt):
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if ch == ':' and (not in_str):
                return i
        return -1
    l = s.find('{')
    if l == -1:
        return s
    colon_rel = first_top_level_colon(s[l + 1:])
    if colon_rel == -1:
        return s
    c = l + 1 + colon_rel
    head = s[:c]
    tail = s[c:]
    raw_key = head[l + 1:].strip()
    if raw_key.startswith('"') and raw_key.endswith('"'):
        raw_key = raw_key[1:-1]
    key_json = json.dumps(raw_key, ensure_ascii=False)
    s = s[:l + 1] + key_json + tail
    return s

def fix_key(key, root_q=None):
    key = key.rstrip(' :')
    if root_q and key.replace('"', '').strip() in root_q.replace('"', '').strip():
        return root_q
    if key.count('"') % 2 == 1:
        key += '"'
    return key

def greedy_fix_first_key(s: str) -> str:
    if not s or '{' not in s or '}' not in s:
        return s
    l = s.find('{')
    r = s.rfind('}')
    if l == -1 or r == -1 or l >= r:
        return s
    c = s.find(':', l + 1)
    if c == -1 or c > r:
        return s
    raw_key = s[l + 1:c].strip()
    if len(raw_key) >= 2 and raw_key[0] == '"' and (raw_key[-1] == '"'):
        raw_key = raw_key[1:-1]
    raw_key = raw_key.replace('\\"', '"').replace('\\\\', '\\')
    key_json = json.dumps(raw_key, ensure_ascii=False)
    return s[:l + 1] + key_json + s[c:]

def fix_keys_recursive(d, root_q=None):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            fixed_k = fix_key(k, root_q)
            fixed_v = fix_keys_recursive(v, root_q)
            new_dict[fixed_k] = fixed_v
        return new_dict
    elif isinstance(d, list):
        return [fix_keys_recursive(x, root_q) for x in d]
    else:
        return d

def compute_decomp(entry):
    prompt = entry['prompt']
    root_q = prompt.splitlines()[-2][3:].strip()
    resp = entry['response'].strip()

    def _slice_braces_simple(text: str) -> str:
        l = text.find('{')
        r = text.rfind('}')
        return text[l:r + 1] if l != -1 and r != -1 and (r > l) else text

    def _slice_braces_balanced(text: str) -> str:
        start = text.find('{')
        if start == -1:
            return text
        depth, esc, in_str = (0, False, False)
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return text
    json_part = _slice_braces_simple(resp)

    def _try_parse_with_candidates(jtxt: str):
        candidates = [jtxt]
        try1 = fix_llm_json_key(jtxt)
        candidates.append(try1)
        try2 = greedy_fix_first_key(jtxt)
        candidates.append(try2)
        try3 = greedy_fix_first_key(try1) if try1 != jtxt else try1
        candidates.append(try3)
        seen, uniq = (set(), [])
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        last_err = None
        for idx, cnd in enumerate(uniq, 1):
            try:
                obj = json.loads(cnd)
                if idx > 1:
                    lab = ['orig', 'fix_llm', 'greedy(orig)', 'greedy(fix_llm)'][idx - 1]
                return (obj, cnd)
            except Exception as ee:
                last_err = ee
        raise last_err
    try:
        decomp = json.loads(json_part)
        chosen_part = json_part
    except Exception as e:
        try:
            decomp, chosen_part = _try_parse_with_candidates(json_part)
        except Exception:
            json_part2 = _slice_braces_balanced(resp)
            if json_part2 != json_part:
                pass
            try:
                decomp, chosen_part = _try_parse_with_candidates(json_part2)
            except Exception as e2:
                raise e
    content = entry.get('logprobs', {}).get('content') or []
    tokens = [d.get('token', '') for d in content]
    logps = [d.get('logprob', None) for d in content]
    if tokens and tokens[-1] == '.':
        tokens.pop()
        if logps:
            logps.pop()
    nodes = {}
    avgs = []
    for q, children in decomp.items() if isinstance(decomp, dict) else []:
        if isinstance(children, list):
            pattern = json.dumps(children, ensure_ascii=False)
            st, ed = locate_span(tokens, pattern)
            avg = None
            if st is not None and ed is not None and (ed - st > 1) and (len(children) >= 2):
                slice_logs = [lp for lp in logps[st + 1:ed] if isinstance(lp, (int, float))]
                if slice_logs:
                    avg = round(sum(slice_logs) / len(slice_logs), 6)
                    avgs.append(avg)
            nodes[q] = {'decomposition': children, 'subq_avg_logprob': avg}
    if isinstance(nodes, dict):
        nodes = fix_keys_recursive(nodes, root_q)
    return (root_q, nodes, avgs)
