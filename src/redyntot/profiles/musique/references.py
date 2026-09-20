"""Reference-prefix normalization and ERQT correction."""
import json
import re
from collections import deque

def adjust_refs(text, desired_count):
    return re.sub('(#+)(\\d+)', lambda m: '#' * desired_count + m.group(2), text)

def lookup_entry(old_dec, node_q):
    if node_q in old_dec:
        return (old_dec[node_q], node_q)
    low = node_q.lower()
    for k in old_dec:
        if k.lower() == low:
            return (old_dec[k], k)
    return (None, None)

def fix_refs_and_flatten(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as rf:
        data = json.load(rf)
    to_remove = set()
    for root_q, record in data.items():
        if 'decomposition' not in record:
            continue
        for info in record['decomposition'].values():
            ch = info.get('decomposition')
            if isinstance(ch, list) and len(ch) == 1:
                to_remove.add(root_q)
                break
    for root_q in to_remove:
        del data[root_q]
    for root_q, record in data.items():
        if 'decomposition' not in record:
            continue
        old_dec = record['decomposition']
        new_dec = {}
        queue = deque([(root_q, 0, {})])
        visited = set()
        while queue:
            node_q, depth, parent_obs = queue.popleft()
            if node_q in visited:
                continue
            visited.add(node_q)
            entry, real_key = lookup_entry(old_dec, node_q)
            if not isinstance(entry, dict):
                continue
            children = entry.get('decomposition', [])
            if not isinstance(children, list) or len(children) < 2:
                continue
            parent_placeholders = {m.group(0) for m in re.finditer('(#+\\d+)', node_q)}
            observed = {}
            for ch in children:
                tgt = 1 if depth == 0 else depth + 1
                tmp_fixed = adjust_refs(ch, tgt)
                raw = re.sub('^#+', '', tmp_fixed)
                observed[raw] = tgt
            corrected = []
            for ch in children:

                def decide_replacement(m):
                    ph = m.group(0)
                    hashes = m.group(1)
                    num = int(m.group(2))
                    if ph in parent_placeholders:
                        return ph
                    raw = re.sub('^#+', '', ph + m.group(2))
                    desired = parent_obs.get(raw, observed.get(raw, depth + 1 if depth > 0 else 1))
                    return '#' * desired + str(num)
                tmp = re.sub('(#+)(\\d+)', decide_replacement, ch)
                corrected.append(tmp)
                sub_entry, _ = lookup_entry(old_dec, re.sub('^#+', '', ch))
                if isinstance(sub_entry, dict):
                    sublist = sub_entry.get('decomposition')
                    if isinstance(sublist, list) and len(sublist) >= 2:
                        queue.append((re.sub('^#+', '', ch), depth + 1, observed))
            if depth == 0:
                new_key = real_key or node_q
            else:
                raw_node = re.sub('^#+', '', node_q)
                cnt_node = parent_obs.get(raw_node, depth)
                new_key = adjust_refs(real_key or node_q, cnt_node)
            new_dec[new_key] = {'decomposition': corrected, 'subq_avg_logprob': entry.get('subq_avg_logprob')}
        record['decomposition'] = new_dec
    with open(output_path, 'w', encoding='utf-8') as wf:
        json.dump(data, wf, ensure_ascii=False, indent=2)
