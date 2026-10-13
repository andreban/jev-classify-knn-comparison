"""Publication charts for the Jev vs. PyTorch MLP vs. k-NN article.

Reads results/reevaluation.json (src/reevaluate.py) and results/jev.jsonl.
Style matches articles/jev-comment-spam/charts.py: light surface, single
panels, no titles (the article caption carries the message), direct labels.

    .venv/bin/python charts.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

OUT = "results/plots"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})

R = json.load(open("results/reevaluation.json"))
FD = R["full_data"]
SW = R["sweep"]["summary"]

MODELS = [  # key, label, color, marker filled?
    ("jev", "Jev (no training)", AQUA, True),
    ("pytorch_gemini", "MLP, cloud embeddings", BLUE, True),
    ("knn_gemini", "k-NN, cloud embeddings", ORANGE, True),
    ("pytorch_minilm", "MLP, MiniLM embeddings", BLUE, False),
    ("knn_minilm", "k-NN, MiniLM embeddings", ORANGE, False),
]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# 1. Ranking quality on `toxic`: ROC-AUC vs. average precision, with 95% CIs.
def ranking():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
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
    axes[1].set_xlim(0.45, 0.95)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([m[1] for m in MODELS])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    fig.supxlabel("Score on the toxic label (dot) with 95% bootstrap interval (line)", fontsize=10, color=INK2, y=0.02)
    fig.tight_layout()
    save(fig, "ranking-quality.png")


# 2. Jev threshold sweep on `toxic`.
def jev_threshold():
    rows = [json.loads(l) for l in open("results/jev.jsonl")]
    p = np.array([r["probabilities"]["toxic"] for r in rows])
    y = np.array([r["expected"]["toxic"] for r in rows])
    ts = np.round(np.arange(0.30, 0.995, 0.01), 2)
    prec, rec, f1 = [], [], []
    for t in ts:
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        pr = tp / pred.sum() if pred.sum() else 1.0
        rc = tp / y.sum()
        prec.append(pr); rec.append(rc); f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for vals, c, lab, lx, dy in [(prec, BLUE, "Precision", 0.60, -0.07), (rec, ORANGE, "Recall", 0.60, 0.03),
                                 (f1, INK, "F1", 0.60, 0.03)]:
        ax.plot(ts, vals, color=c, linewidth=2, drawstyle="steps-post", label=lab)
        i = int(np.argmin(np.abs(ts - lx)))
        ax.text(ts[i], vals[i] + dy, lab, color=INK2, fontsize=10)
    for t, lab in [(0.5, "0.5"), (0.92, "0.92")]:
        ax.axvline(t, color=INK2, linewidth=1, linestyle="--")
        ax.text(t + 0.005, 0.06, lab, color=INK2, fontsize=10)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_xlim(0.3, 0.99); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Jev threshold for the toxic label")
    grid(ax)
    save(fig, "jev-threshold.png")


# 3. Labeled examples needed: toxic F1 vs. number of labels (cloud embeddings vs. Jev).
def labels_needed():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cal = FD["jev_calibration_budget"]
    series = [
        ("Jev, threshold picked\non the N labels", AQUA, [c["n_labels"] for c in cal], [c["f1_mean"] for c in cal],
         [c["f1_p10"] for c in cal], [c["f1_p90"] for c in cal]),
    ]
    for key, color, label in [("mlp_gemini", BLUE, "MLP trained on\nthe N labels"), ("knn7_gemini", ORANGE, "k-NN over\nthe N labels")]:
        s_ = SW[key]
        series.append((label, color, [e["n"] for e in s_], [e["toxic_f1"]["mean"] for e in s_],
                       [e["toxic_f1"]["p10"] for e in s_], [e["toxic_f1"]["p90"] for e in s_]))
    for label, color, n, m, lo, hi in series:
        ax.fill_between(n, lo, hi, color=color, alpha=0.10, linewidth=0)
        ax.plot(n, m, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=1.5, label=label.replace("\n", " "))
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.set_xscale("log")
    ticks = [10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(8, 8000 * 1.05); ax.set_ylim(0, 0.9)
    ax.set_xlabel("Labeled examples (log scale)")
    ax.set_ylabel("F1 on the toxic label")
    grid(ax)
    save(fig, "labels-needed.png")


# 4. Embedding vs. classifier: toxic average precision as training data grows.
def embedding_gap():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for key, color, ls, label in [("mlp_gemini", BLUE, "-", "MLP, cloud"), ("knn7_gemini", ORANGE, "-", "k-NN, cloud"),
                                  ("mlp_minilm", BLUE, "--", "MLP, MiniLM"), ("knn7_minilm", ORANGE, "--", "k-NN, MiniLM")]:
        s = SW[key]
        n = [e["n"] for e in s]
        m = [e["toxic_ap"]["mean"] for e in s]
        ax.fill_between(n, [e["toxic_ap"]["p10"] for e in s], [e["toxic_ap"]["p90"] for e in s],
                        color=color, alpha=0.08, linewidth=0)
        ax.plot(n, m, color=color, linewidth=2, linestyle=ls, marker="o", markersize=6,
                markerfacecolor=color if ls == "-" else SURF, markeredgecolor=color, markeredgewidth=1.5)
        ax.text(n[-1] * 1.12, m[-1], label, va="center", fontsize=10, color=INK2)
    jap = FD["models"]["jev"]["per_label"]["toxic"]["avg_precision"]
    ax.axhline(jap, color=AQUA, linewidth=2)
    ax.text(120, jap - 0.04, f"Jev, no labels: {jap:.2f}", color=INK2, fontsize=10)
    ax.axhline(0.0925, color=INK2, linewidth=1, linestyle=":")
    ax.text(1000, 0.0925 + 0.015, "random guess", color=INK2, fontsize=10)
    ax.set_xscale("log")
    ticks = [15, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(12, 8000 * 1.05); ax.set_ylim(0, 1)
    ax.set_xlabel("Labeled training examples (log scale)")
    ax.set_ylabel("Average precision on the toxic label")
    grid(ax)
    save(fig, "embedding-vs-classifier.png")


# 5. Jev toxic score distribution: toxic vs. clean comments (small multiples).
def jev_scores():
    rows = [json.loads(l) for l in open("results/jev.jsonl")]
    p = np.array([r["probabilities"]["toxic"] for r in rows])
    y = np.array([r["expected"]["toxic"] for r in rows])
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for ax, mask, color, label in [(axes[0], y == 1, ORANGE, f"Labeled toxic (n={int((y == 1).sum())})"),
                                   (axes[1], y == 0, BLUE, f"Labeled clean (n={int((y == 0).sum()):,})")]:
        counts, _ = np.histogram(p[mask], bins=bins)
        share = counts / mask.sum()
        ax.bar(bins[:-1], share, width=0.05, align="edge", color=color, edgecolor=SURF, linewidth=1.5)
        for t in (0.5, 0.92):
            ax.axvline(t, color=INK2, linewidth=1, linestyle="--")
        ax.set_title(label, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_ylabel("Share of comments")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        grid(ax, "y")
    axes[0].text(0.505, axes[0].get_ylim()[1] * 0.85, "0.5", color=INK2, fontsize=10)
    axes[0].text(0.925, axes[0].get_ylim()[1] * 0.85, "0.92", color=INK2, fontsize=10)
    axes[1].set_xlabel("Jev's toxic score")
    axes[1].set_xlim(0, 1)
    fig.tight_layout()
    save(fig, "jev-scores.png")


# 6. Per-label average precision for the three labels with enough examples.
def per_label():
    labels = ["toxic", "obscene", "insult"]
    models = [("jev", "Jev", AQUA), ("pytorch_gemini", "MLP, cloud", BLUE), ("knn_gemini", "k-NN, cloud", ORANGE)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ys = np.arange(len(labels))[::-1]
    for y, lab in zip(ys, labels):
        vals = [FD["models"][k]["per_label"][lab]["avg_precision"] for k, _, _ in models]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=3, zorder=1)
        for (k, name, color), v in zip(models, vals):
            ax.plot([v], [y], "o", markersize=11, color=color, markeredgecolor=SURF, markeredgewidth=2, zorder=3,
                    label=name if y == ys[0] else None)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{l} (n={FD['support'][l]})" for l in labels])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.55, 0.95)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Average precision")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    grid(ax, "x")
    save(fig, "per-label.png")


if __name__ == "__main__":
    ranking()
    jev_threshold()
    labels_needed()
    embedding_gap()
    jev_scores()
    per_label()
    print("wrote", sorted(os.listdir(OUT)))
