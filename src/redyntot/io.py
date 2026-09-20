"""Explicit question IDs and JSON serialization shared by the command-line stages."""
import hashlib
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON array or JSONL records")
    return rows


def questions(path):
    output = []
    for row in load_rows(path):
        rid = row.get("id", row.get("question_id", row.get("_id")))
        question = row.get("question", row.get("question_text"))
        if rid is None or not isinstance(question, str) or not question.strip():
            raise ValueError("Every question needs a stable id and nonempty question text")
        if "\n" in question or "\r" in question:
            raise ValueError("Multiline questions require explicit preprocessing before prompt generation")
        answer = row.get("answer", row.get("gold_answer"))
        if answer is None and row.get("answers_objects"):
            answer = row["answers_objects"][0]["spans"][0]
        output.append({"id": str(rid), "question": question.strip(), "answer": answer})
    if len({row["id"] for row in output}) != len(output):
        raise ValueError("Duplicate question IDs")
    if len({row["question"] for row in output}) != len(output):
        raise ValueError("Duplicate question text cannot share a root-keyed ERQT output")
    return output


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_manifest(output, config, inputs):
    save_json(str(output) + ".manifest.json", {
        "config": config,
        "inputs": [{"file": Path(p).name, "sha256": file_hash(p)} for p in inputs],
        "output_sha256": file_hash(output),
    })
