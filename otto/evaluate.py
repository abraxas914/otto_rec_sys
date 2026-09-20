"""Strict, dependency-free evaluator for local OTTO competition labels."""

import argparse
import csv
import json
from pathlib import Path

WEIGHTS = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}


def item_set(values):
    if not isinstance(values, list) or any(type(x) is not int or x < 0 for x in values):
        raise ValueError("Item labels must be lists of nonnegative integer IDs")
    return set(values)


def evaluate(labels_path, predictions_path):
    predictions = {}
    with Path(predictions_path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["session_type", "labels"]:
            raise ValueError("Expected CSV columns: session_type,labels")
        for row in reader:
            if None in row or row["labels"] is None:
                raise ValueError("Malformed prediction row")
            sid, kind = row["session_type"].rsplit("_", 1)
            key = (int(sid), kind)
            if key[0] < 0 or kind not in WEIGHTS or key in predictions:
                raise ValueError("Invalid or duplicate session_type: " + row["session_type"])
            values = [int(x) for x in row["labels"].split()]
            item_set(values)
            predictions[key] = set(values[:20])

    totals = {kind: {"hits": 0, "denominator": 0} for kind in WEIGHTS}
    sessions = set()
    with Path(labels_path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            sid = record["session"]
            if type(sid) is not int or sid < 0 or sid in sessions:
                raise ValueError("Invalid or duplicate label session")
            sessions.add(sid)
            truth = record["labels"]
            if not isinstance(truth, dict) or set(truth) - WEIGHTS.keys():
                raise ValueError("Unknown label type")
            for kind in WEIGHTS:
                values = truth.get(kind, [])
                if kind == "clicks" and type(values) is int:
                    values = [values]
                target = item_set(values)
                if kind == "clicks" and len(values) > 1:
                    raise ValueError("Clicks target must contain at most the next click")
                key = (sid, kind)
                if key not in predictions:
                    raise ValueError("Missing prediction for " + str(key))
                predicted = predictions.pop(key)
                totals[kind]["hits"] += len(predicted & target)
                totals[kind]["denominator"] += min(20, len(target))
    if predictions:
        raise ValueError("Predictions include sessions absent from labels")
    for kind, stats in totals.items():
        if not stats["denominator"]:
            raise ValueError("Undefined recall: no targets for " + kind)
        stats["recall@20"] = stats["hits"] / stats["denominator"]
    return {
        "protocol": "otto-competition-recall@20",
        "sessions": len(sessions),
        "tasks": totals,
        "score": sum(WEIGHTS[k] * totals[k]["recall@20"] for k in WEIGHTS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.labels, args.predictions), indent=2))


if __name__ == "__main__":
    main()
