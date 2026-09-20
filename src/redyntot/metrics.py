"""Answer EM and token F1, preserving benchmark categorical-answer conventions."""
from collections import Counter
import re
import string


def canonical_answer(text):
    lower = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(?:a|an|the)\b", " ", lower).split())


def answer_metrics(prediction, gold):
    predicted, expected = canonical_answer(prediction), canonical_answer(gold)
    exact = float(predicted == expected)
    categories = {"yes", "no", "noanswer"}
    if predicted != expected and (predicted in categories or expected in categories):
        return {"em": exact, "f1": 0.0}
    pt, gt = predicted.split(), expected.split()
    overlap = sum((Counter(pt) & Counter(gt)).values())
    return {"em": exact, "f1": 2 * overlap / (len(pt) + len(gt)) if overlap else 0.0}


def evaluate(questions, predictions):
    known = {x["id"] for x in questions}
    by_id = {}
    for row in predictions:
        rid = str(row["id"])
        if rid not in known or rid in by_id:
            raise ValueError("Predictions contain unknown or duplicate IDs")
        by_id[rid] = row
    evaluated = []
    for row in questions:
        if row["answer"] is None:
            raise ValueError("Gold answer missing")
        entry = by_id.get(row["id"])
        prediction = str(entry.get("answer") or "") if entry else ""
        score = answer_metrics(prediction, str(row["answer"]))
        if entry is None or entry.get("status", "ok") != "ok":
            score = {"em": 0.0, "f1": 0.0}
        evaluated.append({"id": row["id"], "question": row["question"], "prediction": prediction, "gold": row["answer"], "status": entry.get("status", "ok") if entry else "missing", **score})
    n = len(evaluated)
    return {"n": n, "em": sum(x["em"] for x in evaluated) / n if n else 0.0, "f1": sum(x["f1"] for x in evaluated) / n if n else 0.0, "missing_predictions": len(known - set(by_id)), "per_question": evaluated}
