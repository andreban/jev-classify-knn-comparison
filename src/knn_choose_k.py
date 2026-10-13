"""Choose k-NN's K without touching the evaluation set.

Leave-one-out k-NN on the 8,000 training rows (each training comment is scored
from its K nearest *other* training comments) picks K by macro average
precision; the chosen K is then scored once on the 2,000 evaluation comments.

    PYTHONPATH=src .venv/bin/python src/knn_choose_k.py
"""
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from toxicity_dataset import get_or_create_split_indices, load_full_records

KS = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
FILES = {"gemini": "YT-comments-text-embedding-004-768-10k.jsonl", "minilm": "SP-all-MiniLM-L6-v2.jsonl"}


def scores(d, y, k):
    idx = torch.topk(d, k, dim=1, largest=False).indices
    return y[idx].mean(1).numpy()


def metrics(y, p):
    return {"toxic_ap": float(average_precision_score(y[:, 0], p[:, 0])),
            "toxic_auc": float(roc_auc_score(y[:, 0], p[:, 0])),
            "macro_ap": float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in range(6)]))}


out = {}
for emb, f in FILES.items():
    recs = load_full_records(f)
    tr, va = get_or_create_split_indices(num_samples=len(recs))
    x = torch.tensor([r["embeddings"] for r in recs], dtype=torch.float32)
    y = torch.tensor([r["labels"] for r in recs], dtype=torch.float32)
    xtr, ytr, xva, yva = x[tr], y[tr], x[va], y[va].numpy()
    dtr = torch.cdist(xtr, xtr)
    dtr.fill_diagonal_(float("inf"))  # leave-one-out
    dva = torch.cdist(xva, xtr)
    loo = {k: metrics(ytr.numpy(), scores(dtr, ytr, k)) for k in KS}
    best = max(KS, key=lambda k: loo[k]["macro_ap"])
    out[emb] = {"loo_train": loo, "chosen_k": best,
                "eval_chosen_k": metrics(yva, scores(dva, ytr, best)),
                "eval_k7": metrics(yva, scores(dva, ytr, 7))}
    print(emb, "chosen K =", best)
    for k in KS:
        print(f"  K={k:2d} LOO train: toxic AP {loo[k]['toxic_ap']:.3f} macro AP {loo[k]['macro_ap']:.3f}")
    print("  eval chosen K:", {a: round(b, 3) for a, b in out[emb]["eval_chosen_k"].items()})
    print("  eval K=7:     ", {a: round(b, 3) for a, b in out[emb]["eval_k7"].items()})
json.dump(out, open("results/knn_choose_k.json", "w"), indent=1)

# Save evaluation predictions for the chosen K so they can be compared like the other models.
from toxicity_dataset import LABELS  # noqa: E402

for emb, f in FILES.items():
    recs = load_full_records(f)
    tr, va = get_or_create_split_indices(num_samples=len(recs))
    x = torch.tensor([r["embeddings"] for r in recs], dtype=torch.float32)
    y = torch.tensor([r["labels"] for r in recs], dtype=torch.float32)
    k = out[emb]["chosen_k"]
    p = scores(torch.cdist(x[va], x[tr]), y[tr], k)
    with open(f"results/knn_{emb}_chosen_k.jsonl", "w", encoding="utf-8") as fh:
        for i, oi in enumerate(va):
            fh.write(json.dumps({"id": recs[oi]["id"], "k": k,
                                 "probabilities": {l: float(p[i, j]) for j, l in enumerate(LABELS)},
                                 "expected": {l: int(recs[oi]["labels"][j]) for j, l in enumerate(LABELS)}}) + "\n")
