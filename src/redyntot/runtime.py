"""Stage orchestration around the retained dataset-specific ReDynToT kernels."""
import importlib
import json
from pathlib import Path
import tempfile
from collections import defaultdict

from .io import load_json, load_rows, questions, save_json, save_rows, record_manifest
from .openai_client import ChatClient


PROFILES = {"hotpotqa", "2wikimqa", "musique"}
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def module(profile, name):
    if profile not in PROFILES:
        raise ValueError("Unknown dataset profile")
    return importlib.import_module(f"redyntot.profiles.{profile}.{name}")


def prompt_text(profile, filename):
    return (PACKAGE_ROOT / "prompts" / profile / filename).read_text(encoding="utf-8")


def build_prompts(question_file, config_file, output):
    import spacy
    cfg = load_json(config_file)
    matcher = module(cfg["profile"], "matching")
    parser = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    parser.add_pipe("sentencizer", before="parser")
    pool_file = PACKAGE_ROOT / "data/exemplar_pool_600.json"
    pool = load_json(pool_file)
    matcher.configure(pool, parser, cfg["k"])
    identifiers = {row["question"]: f"EX{i + 1:04d}" for i, row in enumerate(pool)}
    output_rows = []
    for row in questions(question_file):
        examples = matcher.retrieve_examples(matcher.fingerprint_aggregate(row["question"]))
        pairs = []
        for example in examples:
            pairs.extend(["Q: " + example["question"], "A: " + json.dumps(example["decomposition"], ensure_ascii=False)])
        user = "\n".join(pairs) + "\n\nQ: " + row["question"] + "\nA:"
        output_rows.append({"id": row["id"], "question": row["question"], "prompt": user, "requested_k": cfg["k"], "actual_k": len(examples), "example_ids": [identifiers[x["question"]] for x in examples]})
    save_rows(output, output_rows)
    record_manifest(output, cfg, [question_file, config_file, pool_file])


def violations(nodes, averages, threshold):
    return {"low_confidence": any(x is not None and x < threshold for x in averages), "self_reference": any(q in value["decomposition"] for q, value in nodes.items()), "single_child": any(len(value["decomposition"]) == 1 for value in nodes.values())}


def decompose(prompt_file, config_file, output):
    cfg = load_json(config_file)
    profile = cfg["profile"]
    parsing = module(profile, "decomposition")
    client = ChatClient(cfg["model"])
    records = []
    inputs = load_rows(prompt_file)
    if len({str(r['id']) for r in inputs}) != len(inputs):
        raise ValueError('Duplicate input IDs')
    for row in inputs:
        trace = []
        try:
            choice = client.complete(prompt_text(profile, "decomposition_system.txt"), row["prompt"], stop=["\n\n"])
            text = choice["message"]["content"]
            _, nodes, averages = parsing.compute_decomp({"prompt": row["prompt"], "response": text, "logprobs": choice["logprobs"]})
            trace.append({"round": 0, "nodes": nodes, "raw": choice})
            snippet = text[text.find("{"):text.rfind("}") + 1]
            rounds = 0
            while cfg["reflection_enabled"] and rounds < cfg["max_reconstructions"] and any(violations(nodes, averages, cfg["reflection_threshold"]).values()):
                rounds += 1
                request = prompt_text(profile, "reconstruction_user_template.txt").format(example_json=snippet)
                choice = client.complete(prompt_text(profile, "reconstruction_system.txt"), request)
                _, nodes, averages = parsing.compute_decomp({"prompt": row["prompt"], "response": choice["message"]["content"], "logprobs": choice["logprobs"]})
                trace.append({"round": rounds, "nodes": nodes, "raw": choice})
                snippet = json.dumps(nodes, ensure_ascii=False)
            record = {"id": row["id"], "question": row["question"], "status": "ok", "decomposition": nodes, "reconstructions": rounds, "trace": trace}
        except Exception as error:
            record = {"id": row["id"], "question": row["question"], "status": "generation_error", "error_type": type(error).__name__, "decomposition": {}, "trace": trace}
        records.append(record)
        save_rows(output, records)
    record_manifest(output, cfg, [prompt_file, config_file])
    save_json(str(output) + ".usage.json", {"calls": client.calls, "failures": client.failures})


def reject_cycles(root, mapping):
    active, visited = set(), set()
    def visit(node):
        if node in active:
            raise ValueError("Cyclic decomposition")
        if node in visited:
            return
        active.add(node)
        for child in mapping.get(node, []):
            if child != node:
                visit(child)
        active.remove(node)
        visited.add(node)
    visit(root)


def prepare_trees(decompositions, profile, output):
    refs, tree = module(profile, "references"), module(profile, "tree")
    rows = load_rows(decompositions)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate input IDs")
    records = []
    for row in rows:
        flattened, status = [], row.get("status", "ok")
        try:
            if status != "ok":
                raise ValueError("Upstream generation failed")
            wrapped = {row["question"]: {"decomposition": row["decomposition"]}}
            raw_map = {q: value["decomposition"] for q, value in row["decomposition"].items()}
            reject_cycles(row["question"], raw_map)
            with tempfile.TemporaryDirectory() as temporary:
                source, corrected = Path(temporary) / "source.json", Path(temporary) / "corrected.json"
                save_json(source, wrapped)
                refs.fix_refs_and_flatten(str(source), str(corrected))
                flattened = tree.build_flattened_list(load_json(corrected))
            if not flattened:
                status = "empty_tree"
            else:
                done = set()
                for node in flattened:
                    if any(x not in done for x in node["sons"] + node["placeholders"]):
                        raise ValueError("ERQT dependencies must precede their consumers")
                    done.add(node["id"])
        except Exception as error:
            status, flattened = "tree_error", []
        records.append({"id": row["id"], "question": row["question"], "status": status, "nodes": flattened})
    save_rows(output, records)
    record_manifest(output, {"profile": profile}, [decompositions])


def answer_trees(tree_file, config_file, corpus, output, index_dir):
    from .sparse import SparseRetriever
    cfg = load_json(config_file)
    inputs = load_rows(tree_file)
    if len({str(r['id']) for r in inputs}) != len(inputs):
        raise ValueError('Duplicate input IDs')
    profile = cfg["profile"]
    node_evidence, node_context, execution = [module(profile, part) for part in ["node_evidence_answer", "node_context_answer", "execution"]]
    client = ChatClient(cfg["model"])
    for part in [node_evidence, node_context, execution]:
        part.requests, part.LLM_URL, part.MODEL_NAME = client, client.url, cfg["model"]
    execution.extract_local_answer = node_evidence.extract_local_answer
    execution.extract_local_answer1 = node_context.extract_local_answer1
    execution.TOP_K_DENSE, execution.TOP_K_BM25, execution.TOP_K_RERANK = cfg["top_k_dense"], cfg["top_k_bm25"], cfg["top_k_rerank"]
    execution.EPSILON = cfg["epsilon"]
    execution._bm25 = SparseRetriever(cfg["es_index"], cfg['bm25_fields'])
    index_dir = Path(index_dir).resolve()
    index_path = index_dir / 'index.faiss'
    if not index_path.exists():
        index_path = index_dir / 'faiss_index.idx'
    embeddings_path = index_dir / 'embeddings.npy'
    if not index_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError('Existing FAISS index and embeddings are required; this entry point never rebuilds a corpus index')
    execution._dr = module(profile, "dense").DenseRetriever(model_name=cfg["dense_model"], passages_path=str(Path(corpus).resolve()), index_path=str(index_path), embeddings_path=str(embeddings_path))
    execution._rr = module(profile, "rerank").Reranker(model_name=cfg["reranker_model"])
    result = []
    for row in inputs:
        failure_count = len(client.failures)
        try:
            if row["status"] != "ok" or not row["nodes"]:
                raise ValueError("No executable ERQT")
            (answer, confidence), trace = execution.process_tree(row["nodes"])
            status = "ok" if len(client.failures) == failure_count else "transport_error"
            record = {"id": row["id"], "question": row["question"], "answer": answer, "confidence": confidence, "status": status, "trace": trace}
        except Exception as error:
            record = {"id": row["id"], "question": row["question"], "answer": "", "status": "execution_error", "error_type": type(error).__name__}
        result.append(record)
        save_rows(output, result)
    record_manifest(output, cfg, [tree_file, config_file, corpus])
    save_json(str(output) + ".usage.json", {"calls": client.calls, "failures": client.failures})
