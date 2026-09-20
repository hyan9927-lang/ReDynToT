"""Reference-prefix normalization and ERQT correction."""
import json
import re
from collections import deque

def adjust_refs(text, desired_count):
    return re.sub('(#+)(\\d+)', lambda m: '#' * desired_count + m.group(2), text)

def normalize_question(q):
    return re.sub('\\s+', ' ', q).strip().lower().replace('?', '').replace('.', '')

def lookup_entry(old_dec_dict, node_q):
    if node_q in old_dec_dict:
        return (old_dec_dict[node_q], node_q)
    normalized_node_q = normalize_question(node_q)
    for k, v in old_dec_dict.items():
        if normalize_question(k) == normalized_node_q:
            return (v, k)
    for k_outer, v_outer in old_dec_dict.items():
        if isinstance(v_outer, dict):
            if node_q in v_outer:
                return (v_outer[node_q], node_q)
            for k_inner, v_inner in v_outer.items():
                if normalize_question(k_inner) == normalized_node_q:
                    return (v_inner, k_inner)
    return (None, None)

def fix_refs_and_flatten(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as rf:
        data = json.load(rf)
    corrected_data = {}
    to_remove = set()
    for root_q, record in data.items():
        if 'decomposition' not in record:
            corrected_data[root_q] = record
            continue
        inner_dec = record['decomposition']
        if isinstance(inner_dec, dict) and len(inner_dec) == 1:
            inner_key = next(iter(inner_dec.keys()))
            normalized_root_q = normalize_question(root_q)
            normalized_inner_key = normalize_question(inner_key)
            if root_q != inner_key:
                corrected_inner_dec_value = inner_dec[inner_key]
                new_inner_dec = {root_q: corrected_inner_dec_value}
                record['decomposition'] = new_inner_dec
                inner_dec = new_inner_dec
        actual_decomposition_info, _ = lookup_entry(inner_dec, root_q)
        if actual_decomposition_info is None:
            if isinstance(inner_dec, dict) and len(inner_dec) == 1:
                actual_decomposition_info = next(iter(inner_dec.values()))
        if actual_decomposition_info and isinstance(actual_decomposition_info, dict):
            ch = actual_decomposition_info.get('decomposition')
            if isinstance(ch, list) and len(ch) == 1:
                to_remove.add(root_q)
        else:
            to_remove.add(root_q)
        corrected_data[root_q] = record
    for root_q in to_remove:
        if root_q in corrected_data:
            del corrected_data[root_q]
    data = corrected_data
    for root_q, record in data.items():
        if 'decomposition' not in record:
            continue
        old_dec_raw = record['decomposition']
        actual_old_dec = {}
        initial_node_q = root_q
        if isinstance(old_dec_raw, dict) and old_dec_raw:
            potential_entry = old_dec_raw.get(root_q)
            if isinstance(potential_entry, dict) and 'decomposition' in potential_entry:
                actual_old_dec = old_dec_raw
                initial_node_q = root_q
            else:
                record['decomposition'] = {}
                continue
        else:
            record['decomposition'] = {}
            continue
        new_dec = {}
        queue = deque([(initial_node_q, 0, {})])
        visited = set()
        while queue:
            node_q, depth, parent_obs = queue.popleft()
            normalized_node_q = normalize_question(node_q)
            if normalized_node_q in visited:
                continue
            visited.add(normalized_node_q)
            entry, real_key = lookup_entry(actual_old_dec, node_q)
            if not isinstance(entry, dict):
                continue
            children = entry.get('decomposition', [])
            if not isinstance(children, list) or len(children) < 2:
                pass
            parent_placeholders = {m.group(0) for m in re.finditer('(#+)(\\d+)', node_q)}
            observed = {}
            for ch_item in children:
                tgt = 1 if depth == 0 else depth + 1
                tmp_fixed = adjust_refs(ch_item, tgt)
                raw = re.sub('^#+', '', tmp_fixed)
                observed[raw] = tgt
            corrected = []
            for ch_item in children:

                def decide_replacement(m):
                    ph = m.group(0)
                    hashes = m.group(1)
                    num = int(m.group(2))
                    if ph in parent_placeholders:
                        return ph
                    desired = depth + 1
                    return '#' * desired + str(num)
                tmp = re.sub('(#+)(\\d+)', decide_replacement, ch_item)
                corrected.append(tmp)
                sub_entry, _ = lookup_entry(actual_old_dec, ch_item)
                if isinstance(sub_entry, dict):
                    sublist = sub_entry.get('decomposition')
                    if isinstance(sublist, list) and len(sublist) >= 2:
                        queue.append((ch_item, depth + 1, observed))
            if depth == 0:
                final_node_key = root_q
            else:
                raw_node_text = re.sub('^#+', '', node_q)
                expected_hashes = parent_obs.get(raw_node_text, depth)
                final_node_key = adjust_refs(node_q, expected_hashes)
            if final_node_key not in new_dec:
                new_dec[final_node_key] = {'decomposition': corrected, 'subq_avg_logprob': entry.get('subq_avg_logprob')}
        record['decomposition'] = new_dec
    with open(output_path, 'w', encoding='utf-8') as wf:
        json.dump(data, wf, ensure_ascii=False, indent=2)
