"""Post-order ERQT flattening and placeholder dependency construction."""
import json
import re
import time

def tqdm(iterable, **kwargs):
    return iterable

def build_decomposition_map(data):
    start_time = time.time()
    root_maps = {}
    for root, info in data.items():
        dec = info.get('decomposition', {})
        dec_map = {}
        for q, node in dec.items():
            children = node.get('decomposition', [])
            dec_map[q] = [c for c in children if c != q]
        root_maps[root] = dec_map
    elapsed = time.time() - start_time
    return root_maps

def compute_depths(root, dec_map):
    depths = {root: 0}
    stack = [root]
    while stack:
        q = stack.pop()
        for child in dec_map.get(q, []):
            depths[child] = depths[q] + 1
            stack.append(child)
    return depths

def traverse_postorder(node, parent, dec_map, flat_list, visited=None):
    if visited is None:
        visited = set()
    sig = (node, parent)
    if sig in visited:
        return
    visited.add(sig)
    for child in dec_map.get(node, []):
        traverse_postorder(child, node, dec_map, flat_list, visited)
    flat_list.append((node, parent))

def flatten_root(root, dec_map):
    flat_list = []
    traverse_postorder(root, None, dec_map, flat_list, visited=set())
    return flat_list

def assign_ids(flat_list):
    return {pair: i + 1 for i, pair in enumerate(flat_list)}

def build_group_nodes(depths, id_map):
    group_nodes = {d: [] for d in set(depths.values())}
    for (q, parent), id_ in sorted(id_map.items(), key=lambda x: x[1]):
        group_nodes[depths[q]].append(id_)
    return group_nodes

def extract_placeholders(question, parent, depth, id_map, dec_map, group_nodes):
    placeholders = []
    for m in re.finditer('(#+)(\\d+)', question):
        k = len(m.group(1))
        n = int(m.group(2))
        if k == depth:
            children = dec_map.get(parent, [])
            if 1 <= n <= len(children):
                placeholders.append(id_map[children[n - 1], parent])
        elif k < depth:
            layer = group_nodes.get(k, [])
            if 1 <= n <= len(layer):
                placeholders.append(layer[n - 1])
    return placeholders

def build_flattened_list(data):
    all_entries = []
    root_maps = build_decomposition_map(data)
    for idx, (root, dec_map) in enumerate(tqdm(root_maps.items(), desc='Roots'), start=1):
        root_start = time.time()
        depths = compute_depths(root, dec_map)
        flat_qs = flatten_root(root, dec_map)
        id_map = assign_ids(flat_qs)
        group_nodes = build_group_nodes(depths, id_map)
        for q, parent in flat_qs:
            d = depths[q]
            entry = {'root': root, 'id': id_map[q, parent], 'question': q, 'placeholders': extract_placeholders(q, parent, d, id_map, dec_map, group_nodes), 'sons': [id_map[child, q] for child in dec_map.get(q, [])], 'group': [d]}
            all_entries.append(entry)
        elapsed = time.time() - root_start
    return all_entries
