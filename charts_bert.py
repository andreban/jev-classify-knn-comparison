"""Charts for the fine-tuned BERT follow-up article (article-bert.md).

Reuses the style and data loading of charts.py, and adds BERT in MAGENTA,
which passes the dataviz palette checks next to BLUE, ORANGE and AQUA.
Reads results/reevaluation.json (src/reevaluate.py) and results/bert_sweep.json
plus results/bert_sweep_runs.jsonl (src/sweep_bert.py).

    .venv/bin/python charts_bert.py
"""
import json
import os

import numpy as np

from charts import AQUA, BLUE, FD, R, GRID, INK, INK2, ORANGE, OUT, SURF, SW, grid, plt, save

MAGENTA = "#c2417a"

MODELS = [  # key, label, color, filled marker?
    ("bert", "BERT, fine-tuned", MAGENTA, True),
    ("pytorch_gemini", "MLP, cloud embeddings", BLUE, True),
    ("knn_gemini", "k-NN, cloud embeddings", ORANGE, True),
    ("jev", "Jev (no training)", AQUA, True),
    ("pytorch_minilm", "MLP, MiniLM embeddings", BLUE, False),
    ("knn_minilm", "k-NN, MiniLM embeddings", ORANGE, False),
]


# 1. Ranking quality on `toxic`, as in the first article, with BERT added.
def ranking():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True)
    ys = np.arange(len(MODELS))[::-1]
    for ax, metric, ci_key, title in [
        (axes[0], "roc_auc", "toxic_roc_auc", "ROC-AUC"),
        (axes[1], "avg_precision", "toxic_avg_precision", "Average precision"),
    ]:
        for y, (key, label, color, filled) in zip(ys, MODELS):
            m = FD["models"][key]
            v = m["per_label"]["toxic"][metric]
            lo, hi = m["ci95"][ci_key]
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot([v], [y], "o", markersize=9, color=color if filled else SURF,
                    markeredgecolor=color, markeredgewidth=2, zorder=5)
            ax.text(hi + 0.008, y, f"{v:.2f}", va="center", fontsize=10, color=INK2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        grid(ax, "x")
    axes[0].set_xlim(0.80, 1.0)
    axes[1].set_xlim(0.45, 0.97)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([m[1] for m in MODELS])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    fig.supxlabel("Score on the toxic label (dot) with 95% bootstrap interval (line)", fontsize=10, color=INK2, y=0.02)
    fig.tight_layout()
    save(fig, "bert-ranking.png")


# 2. Average precision on all six labels: BERT vs. the cloud models and Jev.
def per_label():
    labels = ["toxic", "insult", "obscene", "severe_toxic", "identity_hate", "threat"]
    models = [("bert", "BERT, fine-tuned", MAGENTA), ("pytorch_gemini", "MLP, cloud", BLUE),
              ("knn_gemini", "k-NN, cloud", ORANGE), ("jev", "Jev", AQUA)]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ys = np.arange(len(labels))[::-1]
    for y, lab in zip(ys, labels):
        vals = [FD["models"][k]["per_label"][lab]["avg_precision"] for k, _, _ in models]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=3, zorder=1)
        for (k, name, color), v in zip(models, vals):
            ax.plot([v], [y], "o", markersize=11, color=color, markeredgecolor=SURF, markeredgewidth=2, zorder=3,
                    label=name if y == ys[0] else None)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{l} ({FD['support'][l]})" for l in labels])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Average precision (examples in the evaluation set in parentheses)")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    grid(ax, "x")
    save(fig, "bert-per-label.png")


# 3. Labels needed: toxic average precision vs. training-set size. The MLP
# lines use the same three random subsets per size as the BERT runs (reps 0-2
# of the sweep in src/reevaluate.py), so every point compares like with like.
def labels_needed():
    summary = json.load(open("results/bert_sweep.json"))["summary"]
    runs = [json.loads(l) for l in open("results/bert_sweep_runs.jsonl")]
    reps = sorted({r["rep"] for r in runs})
    full_ap = FD["models"]["bert"]["per_label"]["toxic"]["avg_precision"]
    sweep_runs = R["sweep"]["runs"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for emb, color, ls, label in [("gemini", BLUE, "-", "MLP, cloud"), ("minilm", BLUE, "--", "MLP, MiniLM")]:
        n = sorted({r["n"] for r in sweep_runs if r["n"] >= 50})
        v = [[r["mlp"]["toxic_ap"] for r in sweep_runs if r["embedding"] == emb and r["n"] == x and r["rep"] in reps]
             for x in n]
        m = [np.mean(x) for x in v]
        ax.fill_between(n, [min(x) for x in v], [max(x) for x in v], color=color, alpha=0.08, linewidth=0)
        ax.plot(n, m, color=color, linewidth=2, linestyle=ls, marker="o", markersize=6,
                markerfacecolor=color if ls == "-" else SURF, markeredgecolor=color, markeredgewidth=1.5)
        ax.text(n[-1] * 1.12, m[-1], label, va="center", fontsize=10, color=INK2)

    n = [e["n"] for e in summary] + [8000]
    m = [e["toxic_ap"]["mean"] for e in summary] + [full_ap]
    ax.plot([r["n"] for r in runs], [r["bert"]["toxic_ap"] for r in runs], "o", markersize=5,
            color=MAGENTA, alpha=0.35, markeredgewidth=0, zorder=2)
    # Solid over the sweep sizes, dotted to the single full-data run (no runs in between).
    ax.plot(n[:-1], m[:-1], color=MAGENTA, linewidth=2, zorder=4)
    ax.plot(n[-2:], m[-2:], color=MAGENTA, linewidth=2, linestyle=":", zorder=4)
    ax.plot(n, m, "o", color=MAGENTA, markersize=8, markeredgecolor=SURF, markeredgewidth=1.5, zorder=5)
    ax.text(n[-1] * 1.12, m[-1] + 0.03, "BERT, fine-tuned", va="center", fontsize=10, color=INK2)

    jap = FD["models"]["jev"]["per_label"]["toxic"]["avg_precision"]
    ax.axhline(jap, color=AQUA, linewidth=2)
    ax.text(1200, jap - 0.045, f"Jev, no labels: {jap:.2f}", color=INK2, fontsize=10)
    ax.set_xscale("log")
    ticks = [50, 100, 250, 500, 1000, 2000, 4000, 8000]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(42, 8000 * 1.05); ax.set_ylim(0, 1)
    ax.set_xlabel("Labeled training examples (log scale)")
    ax.set_ylabel("Average precision on the toxic label")
    grid(ax)
    save(fig, "bert-labels-needed.png")


if __name__ == "__main__":
    ranking()
    per_label()
    if os.path.exists("results/bert_sweep.json"):
        labels_needed()
    print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("bert-")))
