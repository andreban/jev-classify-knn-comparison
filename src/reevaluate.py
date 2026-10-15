"""Re-evaluation of the Jev vs. PyTorch MLP vs. k-NN benchmark.

Fixes three problems with the original analysis:

1. Thresholds. Jev's 0.95 threshold was picked on the same 2,000 validation rows
   it was scored on, while the trained models were held to 0.50. Here every
   method gets the same treatment: threshold-free metrics (ROC-AUC, average
   precision) plus F1 with a threshold chosen on one half of the validation set
   and scored on the other half (repeated random splits).
2. Uncertainty. Bootstrap 95% intervals and paired differences, so ties are
   reported as ties.
3. Data-efficiency sweep. The original sweep used a single subset per N (the
   first N rows) and trained the MLP for a fixed number of *epochs*, so N=15 got
   ~100 gradient steps and N=8,000 got ~1,600. Here every N is repeated over
   REPEATS random subsets, and the MLP always gets the same number of
   optimizer steps as the full-data run.

It also measures how many labeled examples Jev needs to pick a good threshold.

Outputs results/reevaluation.json. Requires ./scripts/restore_data.sh for the
sweep (the embedding .jsonl files); the full-data analysis uses cached
results/*.jsonl predictions.

    PYTHONPATH=src .venv/bin/python src/reevaluate.py
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from toxicity_dataset import LABELS

SEED = 0
BOOTSTRAP = 1000
CV_REPEATS = 300
REPEATS = 10
SWEEP_SIZES = [15, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]
CALIB_SIZES = [10, 25, 50, 100, 200, 500, 1000]
KNN_KS = [1, 3, 7]

CONFIGS = {
    "pytorch_gemini": "results/pytorch_gemini.jsonl",
    "knn_gemini": "results/knn_gemini.jsonl",
    "jev": "results/jev.jsonl",
    "pytorch_minilm": "results/pytorch_minilm.jsonl",
    "knn_minilm": "results/knn_minilm.jsonl",
}
# Models added after the article was published. They draw their threshold
# splits from their own RNG so the shared RNG stream above, and with it every
# published number, stays exactly the same.
LATER_CONFIGS = {
    "bert": "results/bert.jsonl",
}
EMBEDDINGS = {
    "gemini": {"file": "YT-comments-text-embedding-004-768-10k.jsonl", "steps": 1600},
    "minilm": {"file": "SP-all-MiniLM-L6-v2.jsonl", "steps": 2400},
}


def load_predictions(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    p = np.array([[r["probabilities"][l] for l in LABELS] for r in rows], dtype=np.float64)
    y = np.array([[r["expected"][l] for l in LABELS] for r in rows], dtype=np.int32)
    return p, y


def f1_at(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    denom = int(pred.sum()) + int(y.sum())
    return 2 * tp / denom if denom else 0.0, tp, int((pred & (y == 0)).sum())


def best_threshold(y, p):
    if y.sum() == 0:
        return 0.5
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.argmax(f))])


def cv_threshold_f1(y, p, rng, repeats=CV_REPEATS):
    """Pick the best-F1 threshold on a random half, score F1 on the other half."""
    f1s, ths = [], []
    n = len(y)
    for _ in range(repeats):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        if y[a].sum() == 0 or y[b].sum() == 0:
            continue
        t = best_threshold(y[a], p[a])
        f1s.append(f1_at(y[b], p[b], t)[0])
        ths.append(t)
    return float(np.mean(f1s)), float(np.median(ths))


def full_data_analysis(rng):
    preds = {k: load_predictions(v) for k, v in CONFIGS.items()}
    preds |= {k: load_predictions(v) for k, v in LATER_CONFIGS.items() if Path(v).exists()}
    y = preds["jev"][1]
    for k, (_, yk) in preds.items():
        assert (yk == y).all(), k

    out = {"support": {l: int(y[:, j].sum()) for j, l in enumerate(LABELS)}, "models": {}}
    n = len(y)
    boot_idx = [rng.integers(0, n, n) for _ in range(BOOTSTRAP)]
    boot_idx = [ii for ii in boot_idx if (y[ii].sum(0) > 0).all()]

    def macro(metric, yy, pp):
        return float(np.mean([metric(yy[:, j], pp[:, j]) for j in range(6)]))

    for name, (p, _) in preds.items():
        model_rng = rng if name in CONFIGS else np.random.default_rng(SEED)
        per_label = {}
        for j, label in enumerate(LABELS):
            f05, tp, fp = f1_at(y[:, j], p[:, j], 0.5)
            cvf, cvt = cv_threshold_f1(y[:, j], p[:, j], model_rng)
            per_label[label] = {
                "roc_auc": float(roc_auc_score(y[:, j], p[:, j])),
                "avg_precision": float(average_precision_score(y[:, j], p[:, j])),
                "f1_at_0.5": f05,
                "f1_cv_threshold": cvf,
                "cv_threshold_median": cvt,
            }
        toxic_auc = [roc_auc_score(y[ii, 0], p[ii, 0]) for ii in boot_idx]
        toxic_ap = [average_precision_score(y[ii, 0], p[ii, 0]) for ii in boot_idx]
        macro_ap = [macro(average_precision_score, y[ii], p[ii]) for ii in boot_idx]
        out["models"][name] = {
            "per_label": per_label,
            "macro_roc_auc": float(np.mean([v["roc_auc"] for v in per_label.values()])),
            "macro_avg_precision": float(np.mean([v["avg_precision"] for v in per_label.values()])),
            "macro_f1_at_0.5": float(np.mean([v["f1_at_0.5"] for v in per_label.values()])),
            "macro_f1_cv_threshold": float(np.mean([v["f1_cv_threshold"] for v in per_label.values()])),
            "ci95": {
                "toxic_roc_auc": np.percentile(toxic_auc, [2.5, 97.5]).tolist(),
                "toxic_avg_precision": np.percentile(toxic_ap, [2.5, 97.5]).tolist(),
                "macro_avg_precision": np.percentile(macro_ap, [2.5, 97.5]).tolist(),
            },
        }

    # Paired bootstrap: a reference model minus each other model.
    def paired_diffs(ref):
        diffs = {}
        pr = preds[ref][0]
        for name in preds:
            if name == ref:
                continue
            po = preds[name][0]
            d = {}
            for mname, metric in [("roc_auc", roc_auc_score), ("avg_precision", average_precision_score)]:
                tox = np.array([metric(y[ii, 0], pr[ii, 0]) - metric(y[ii, 0], po[ii, 0]) for ii in boot_idx])
                mac = np.array([macro(metric, y[ii], pr[ii]) - macro(metric, y[ii], po[ii]) for ii in boot_idx])
                d[f"toxic_{mname}"] = {"median": float(np.median(tox)), "ci95": np.percentile(tox, [2.5, 97.5]).tolist(),
                                       "share_ref_ahead": float((tox > 0).mean())}
                d[f"macro_{mname}"] = {"median": float(np.median(mac)), "ci95": np.percentile(mac, [2.5, 97.5]).tolist(),
                                       "share_ref_ahead": float((mac > 0).mean())}
            diffs[name] = d
        return diffs

    pj = preds["jev"][0]
    out["jev_minus_model"] = paired_diffs("jev")
    if "bert" in preds:
        out["bert_minus_model"] = paired_diffs("bert")

    # Jev threshold table on toxic (descriptive, full validation set).
    table = []
    yt = y[:, 0]
    for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
        f, tp, fp = f1_at(yt, pj[:, 0], t)
        table.append({"threshold": t, "tp": tp, "fp": fp, "precision": tp / max(tp + fp, 1),
                      "recall": tp / yt.sum(), "f1": f, "fpr": fp / (len(yt) - yt.sum())})
    out["jev_toxic_thresholds"] = table

    # How many labeled examples does Jev need to calibrate its threshold?
    calib = []
    for m in CALIB_SIZES:
        f1s, ths = [], []
        for _ in range(CV_REPEATS):
            perm = rng.permutation(len(yt))
            a, b = perm[:m], perm[m:]
            t = best_threshold(yt[a], pj[a, 0])
            f1s.append(f1_at(yt[b], pj[b, 0], t)[0])
            ths.append(t)
        calib.append({"n_labels": m, "f1_mean": float(np.mean(f1s)),
                      "f1_p10": float(np.percentile(f1s, 10)), "f1_p90": float(np.percentile(f1s, 90)),
                      "threshold_median": float(np.median(ths))})
    out["jev_calibration_budget"] = calib
    return out


# ---------------------------------------------------------------- sweep ----

def _load_xy(emb):
    import torch  # noqa: F401
    from toxicity_dataset import get_or_create_split_indices, load_full_records
    recs = load_full_records(EMBEDDINGS[emb]["file"])
    tr, va = get_or_create_split_indices(num_samples=len(recs))
    x = np.array([r["embeddings"] for r in recs], dtype=np.float32)
    y = np.array([r["labels"] for r in recs], dtype=np.float32)
    return x[tr], y[tr], x[va], y[va]


_DATA = {}


def _data(emb):
    if emb not in _DATA:
        _DATA[emb] = _load_xy(emb)
    return _DATA[emb]


def _metrics(yv, p):
    ok = [j for j in range(6) if yv[:, j].sum() > 0]
    return {
        "toxic_auc": float(roc_auc_score(yv[:, 0], p[:, 0])),
        "toxic_ap": float(average_precision_score(yv[:, 0], p[:, 0])),
        "toxic_f1": f1_at(yv[:, 0], p[:, 0], 0.5)[0],
        "macro_auc": float(np.mean([roc_auc_score(yv[:, j], p[:, j]) for j in ok])),
        "macro_ap": float(np.mean([average_precision_score(yv[:, j], p[:, j]) for j in ok])),
        "macro_f1": float(np.mean([f1_at(yv[:, j], p[:, j], 0.5)[0] for j in range(6)])),
    }


def sweep_job(args):
    emb, n, rep = args
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    from all_minilm_l6.toxicity_model import ToxicityModel as MiniLMModel
    from text_embedding_004.toxicity_model import ToxicityModel as GeminiModel

    xtr, ytr, xva, yva = _data(emb)
    rng = np.random.default_rng(1000 * rep + n)
    sub = rng.permutation(len(xtr))[:n]
    xs, ys = torch.tensor(xtr[sub]), torch.tensor(ytr[sub])
    xv = torch.tensor(xva)
    res = {"embedding": emb, "n": n, "rep": rep, "train_positives": ys.sum(0).int().tolist()}

    # k-NN
    d = torch.cdist(xv, xs)
    for k in KNN_KS:
        kk = min(k, n)
        idx = torch.topk(d, kk, dim=1, largest=False).indices
        res[f"knn{k}"] = _metrics(yva, ys[idx].mean(1).numpy())

    # MLP with a fixed optimizer-step budget
    torch.manual_seed(rep)
    model = (GeminiModel if emb == "gemini" else MiniLMModel)()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    bs = min(512, n)
    g = torch.Generator().manual_seed(rep)
    model.train()
    step = 0
    while step < EMBEDDINGS[emb]["steps"]:
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            if len(b) < 2:
                continue
            loss = loss_fn(model(xs[b]), ys[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= EMBEDDINGS[emb]["steps"]:
                break
    model.eval()
    with torch.no_grad():
        res["mlp"] = _metrics(yva, torch.sigmoid(model(xv)).numpy())
    return res


def run_sweep():
    jobs = [(e, n, r) for e in EMBEDDINGS for n in SWEEP_SIZES for r in range(REPEATS)]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        rows = list(ex.map(sweep_job, jobs, chunksize=1))
    summary = {}
    for e in EMBEDDINGS:
        for method in ["mlp"] + [f"knn{k}" for k in KNN_KS]:
            for n in SWEEP_SIZES:
                rs = [r for r in rows if r["embedding"] == e and r["n"] == n]
                entry = {"n": n, "toxic_train_positives_median": float(np.median([r["train_positives"][0] for r in rs]))}
                for m in rs[0][method]:
                    v = np.array([r[method][m] for r in rs])
                    entry[m] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
                summary.setdefault(f"{method}_{e}", []).append(entry)
    return rows, summary


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip re-running the 10x10 training-size sweep and preserve existing sweep in results/reevaluation.json",
    )
    args = parser.parse_args()

    out_path = Path("results/reevaluation.json")
    existing_sweep = None
    if args.no_sweep and out_path.exists():
        existing_data = json.loads(out_path.read_text(encoding="utf-8"))
        existing_sweep = existing_data.get("sweep")

    rng = np.random.default_rng(SEED)
    out = {"full_data": full_data_analysis(rng)}
    if args.no_sweep:
        if existing_sweep is not None:
            out["sweep"] = existing_sweep
        print("Skipped sweep (--no-sweep); preserved existing sweep data.")
    elif all(Path(c["file"]).exists() for c in EMBEDDINGS.values()):
        rows, summary = run_sweep()
        out["sweep"] = {"repeats": REPEATS, "sizes": SWEEP_SIZES, "summary": summary, "runs": rows}
    else:
        print("Embedding files missing; skipping sweep (run ./scripts/restore_data.sh).")
    out_path.write_text(json.dumps(out, indent=1))
    print("wrote results/reevaluation.json")


if __name__ == "__main__":
    main()
