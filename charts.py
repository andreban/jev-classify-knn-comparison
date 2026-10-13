import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.toxicity_dataset import LABELS

# Match the exact light editorial style from articles/jev-comment-spam/charts.py
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SLATE = "#7a7873"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
})


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def load_predictions(path: str):
    probs, expected = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            probs.append([float(row["probabilities"][l]) for l in LABELS])
            expected.append([int(row["expected"][l]) for l in LABELS])
    return np.array(probs, dtype=np.float64), np.array(expected, dtype=np.int32)


def binary_metrics(probs_1d: np.ndarray, y_1d: np.ndarray, thr: float):
    yb = (probs_1d >= thr).astype(np.int32)
    tp = int(((yb == 1) & (y_1d == 1)).sum())
    fp = int(((yb == 1) & (y_1d == 0)).sum())
    fn = int(((yb == 0) & (y_1d == 1)).sum())
    tn = int(((yb == 0) & (y_1d == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return prec, rec, f1, fpr


def macro_auc(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(y_true[:, i], probs[:, i]) for i in range(len(LABELS))]))


def main():
    out_dir = Path("results/plots")
    p_jev, y_true = load_predictions("results/jev.jsonl")
    p_mlp_gem, _ = load_predictions("results/pytorch_gemini.jsonl")
    p_mlp_mini, _ = load_predictions("results/pytorch_minilm.jsonl")
    p_knn_gem, _ = load_predictions("results/knn_gemini.jsonl")
    p_knn_mini, _ = load_predictions("results/knn_minilm.jsonl")

    yt_toxic = y_true[:, 0]

    # -------------------------------------------------------------------------
    # 1. ROC Curves (roc-curves.png)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=":")
    ax.text(0.58, 0.51, "random guess", color=INK2, fontsize=9.5, rotation=42)

    curves = [
        ("Jev (System One, zero-shot)", p_jev[:, 0], macro_auc(p_jev, y_true), AQUA, "-", 2.6),
        ("k-NN K=7 (Cloud 768d, N=8,000)", p_knn_gem[:, 0], macro_auc(p_knn_gem, y_true), ORANGE, "-", 2.1),
        ("PyTorch MLP (Cloud 768d, N=8,000)", p_mlp_gem[:, 0], macro_auc(p_mlp_gem, y_true), BLUE, "-", 2.1),
        ("PyTorch MLP (Local MiniLM 384d)", p_mlp_mini[:, 0], macro_auc(p_mlp_mini, y_true), BLUE, "--", 1.7),
        ("k-NN K=7 (Local MiniLM 384d)", p_knn_mini[:, 0], macro_auc(p_knn_mini, y_true), ORANGE, "--", 1.7),
    ]
    for label, probs_t, m_auc, color, ls, lw in curves:
        fpr, tpr, _ = roc_curve(yt_toxic, probs_t)
        t_auc = roc_auc_score(yt_toxic, probs_t)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                label=f"{label}\n  toxic AUC {t_auc:.3f} · macro AUC {m_auc:.3f}")

    # Annotate Jev operating points (0.50 vs 0.95)
    _, tpr_50, _, fpr_50 = binary_metrics(p_jev[:, 0], yt_toxic, 0.50)
    _, tpr_95, _, fpr_95 = binary_metrics(p_jev[:, 0], yt_toxic, 0.95)
    ax.plot([fpr_50], [tpr_50], "o", markersize=8, color=AQUA, markeredgecolor=SURF, markeredgewidth=1.8, zorder=6)
    ax.annotate("Jev @ 0.50 (uncalibrated)\nFPR 0.19, Recall 0.97", xy=(fpr_50, tpr_50), xytext=(0.34, 0.76),
                fontsize=9.2, color=INK, bbox=dict(fc=SURF, ec=GRID, pad=3), zorder=7,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
    ax.plot([fpr_95], [tpr_95], "o", markersize=8, color=INK, markeredgecolor=SURF, markeredgewidth=1.8, zorder=6)
    ax.annotate("Jev @ 0.95 (calibrated)\nFPR 0.02, Recall 0.65", xy=(fpr_95, tpr_95), xytext=(0.16, 0.55),
                fontsize=9.2, color=INK, bbox=dict(fc=SURF, ec=GRID, pad=3), zorder=7,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate (clean comments wrongly flagged)")
    ax.set_ylabel("True positive rate (toxic comments caught)")
    ax.set_title("ROC curves on 'toxic' (2,000 validation comments)", loc="left", fontsize=12, color=INK, pad=12)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="lower right", fontsize=8.8, labelspacing=0.55)
    save(fig, out_dir, "roc-curves.png")

    # -------------------------------------------------------------------------
    # 2. Threshold Calibration Sweep (threshold-calibration.png)
    # -------------------------------------------------------------------------
    ts = np.linspace(0.10, 0.98, 89)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True)

    panels = [
        (axes[0], p_jev[:, 0], "Jev (zero-shot: judges semantics, not dataset base rate)", 0.95, "calibrated: 0.95\n(F1 = 0.71, FPR = 2.0%)"),
        (axes[1], p_mlp_gem[:, 0], "PyTorch MLP (trained on 8,000 rows: learns 9.3% base rate)", 0.50, "default: 0.50\n(F1 = 0.78, FPR = 1.6%)"),
    ]
    for ax, probs_t, title, opt_t, opt_label in panels:
        precs, recs, f1s, fprs = [], [], [], []
        for t in ts:
            p, r, f1, fpr = binary_metrics(probs_t, yt_toxic, float(t))
            precs.append(p)
            recs.append(r)
            f1s.append(f1)
            fprs.append(fpr)

        ax.plot(ts, precs, color=BLUE, linewidth=2, label="Precision")
        ax.plot(ts, recs, color=ORANGE, linewidth=2, label="Recall")
        ax.plot(ts, f1s, color=AQUA, linewidth=2.5, label="F1 score")
        ax.plot(ts, fprs, color=SLATE, linewidth=1.8, linestyle="--", label="False positive rate")

        ax.axvline(opt_t, color=INK, linewidth=1.5, linestyle="--")
        ha = "right" if opt_t > 0.7 else "left"
        offset = -0.02 if opt_t > 0.7 else 0.02
        ax.text(opt_t + offset, 0.22, opt_label, color=INK, fontsize=9.2, ha=ha,
                bbox=dict(fc=SURF, ec=GRID, pad=3), zorder=6)

        if opt_t == 0.95:
            ax.axvline(0.50, color=INK2, linewidth=1.1, linestyle=":")
            ax.text(0.50 - 0.02, 0.53, "uncalibrated 0.50\n(F1 = 0.51, FPR = 18.6%)", color=INK2, fontsize=8.8, ha="right",
                    bbox=dict(fc=SURF, ec="none", pad=1.5))

        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        ax.set_xlabel("Decision threshold")
        ax.set_xlim(0.10, 0.99)
        ax.set_ylim(0, 1.04)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Rate / Score")
    # Place legend in open space of panel 2 (bottom-left around x=0.12..0.42, y=0.15..0.50)
    axes[1].legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="center left", bbox_to_anchor=(0.03, 0.36), fontsize=9.2)
    save(fig, out_dir, "threshold-calibration.png")

    # -------------------------------------------------------------------------
    # 3. Data Efficiency & k-NN Small-Sample Degradation (data-efficiency.png)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    sweep_files = [
        ("results/pytorch_sweep_gemini.json", "PyTorch MLP (Cloud 768d)", BLUE, "-", "o"),
        ("results/knn_sweep_gemini.json", "k-NN K=7 (Cloud 768d)", ORANGE, "-", "^"),
        ("results/pytorch_sweep_minilm.json", "PyTorch MLP (Local MiniLM 384d)", BLUE, "--", "s"),
        ("results/knn_sweep_minilm.json", "k-NN K=7 (Local MiniLM 384d)", ORANGE, "--", "D"),
    ]

    for path_str, label, color, ls, marker in sweep_files:
        with open(path_str, "r", encoding="utf-8") as f:
            s_data = json.load(f)
        ns, f1s, m_aucs = [], [], []
        for entry in s_data:
            ns.append(entry["n_train"])
            sp = np.array(entry["probabilities"], dtype=np.float64)
            _, _, f1_t, _ = binary_metrics(sp[:, 0], yt_toxic, 0.50)
            f1s.append(f1_t)
            m_aucs.append(macro_auc(sp, y_true))
        axes[0].plot(ns, f1s, label=label, color=color, linestyle=ls, marker=marker, markersize=5.5, linewidth=2)
        axes[1].plot(ns, m_aucs, label=label, color=color, linestyle=ls, marker=marker, markersize=5.5, linewidth=2)

    # Jev zero-shot lines
    _, _, jev_f1_95, _ = binary_metrics(p_jev[:, 0], yt_toxic, 0.95)
    jev_m_auc = macro_auc(p_jev, y_true)
    axes[0].axhline(jev_f1_95, color=AQUA, linewidth=2.4, linestyle="-",
                    label=f"Jev zero-shot @ 0.95 (0 train rows): {jev_f1_95:.3f}")
    axes[1].axhline(jev_m_auc, color=AQUA, linewidth=2.4, linestyle="-",
                    label=f"Jev zero-shot (0 train rows): {jev_m_auc:.3f}")

    # Annotate k-NN cliff at N<=25 in clean space above the MiniLM curve
    axes[0].annotate("k-NN majority-vote cliff:\nF1 = 0.00 at N ≤ 25\n(< 4 toxic rows in train set)",
                     xy=(25, 0.01), xytext=(15, 0.36), fontsize=8.8, color=INK,
                     bbox=dict(fc=SURF, ec=GRID, pad=3), zorder=6,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    axes[0].annotate("k-NN overtakes MLP\nat N ≥ 1,000 (0.805 @ 8k)",
                     xy=(8000, 0.805), xytext=(600, 0.855), fontsize=8.8, color=INK,
                     bbox=dict(fc=SURF, ec=GRID, pad=2.5), zorder=6,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([15, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(axis="x", labelsize=9, rotation=30)
        ax.set_xlabel("Number of labeled training comments (log scale)")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_title("'toxic' F1 score vs. training set size", loc="left", fontsize=11.5, color=INK, pad=10)
    axes[0].set_ylabel("F1 score on 'toxic'")
    axes[0].set_ylim(-0.03, 0.95)
    axes[0].legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="lower right", fontsize=8.0)

    axes[1].set_title("Macro ROC-AUC across all 6 categories", loc="left", fontsize=11.5, color=INK, pad=10)
    axes[1].set_ylabel("Macro ROC-AUC (6 toxicity labels)")
    axes[1].set_ylim(0.45, 1.01)
    axes[1].legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="lower right", fontsize=8.0)

    save(fig, out_dir, "data-efficiency.png")

    # -------------------------------------------------------------------------
    # 4. Per-Category AUC & F1 Breakdown (per-category-f1.png)
    # -------------------------------------------------------------------------
    label_order = ["toxic", "insult", "obscene", "severe_toxic", "identity_hate", "threat"]
    label_Idx = [LABELS.index(l) for l in label_order]
    display_labels = [
        f"{l.replace('_', chr(10))}\n(n={int(y_true[:, LABELS.index(l)].sum())})"
        for l in label_order
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    x = np.arange(len(label_order))
    width = 0.25

    models_cat = [
        ("Jev Zero-Shot (@ 0.95)", p_jev, 0.95, AQUA),
        ("k-NN K=7 (Cloud 768d, N=8k)", p_knn_gem, 0.50, ORANGE),
        ("PyTorch MLP (Cloud 768d, N=8k)", p_mlp_gem, 0.50, BLUE),
    ]

    for m_idx, (m_label, probs_m, thr_m, color) in enumerate(models_cat):
        offset = (m_idx - 1) * width
        aucs_m = [roc_auc_score(y_true[:, i], probs_m[:, i]) for i in label_Idx]
        f1s_m = [binary_metrics(probs_m[:, i], y_true[:, i], thr_m)[2] for i in label_Idx]

        axes[0].bar(x + offset, aucs_m, width=width * 0.88, color=color, label=m_label.replace(" (@ 0.95)", ""))
        bars = axes[1].bar(x + offset, f1s_m, width=width * 0.88, color=color, label=m_label)
        for b, val in zip(bars, f1s_m):
            if val == 0:
                axes[1].text(b.get_x() + b.get_width() / 2, 0.015, "0.00", ha="center", va="bottom",
                             fontsize=7.5, color=INK2, rotation=90)

    axes[0].set_title("Per-category ROC-AUC (ranking quality)", loc="left", fontsize=11.5, color=INK, pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_labels, fontsize=9)
    axes[0].set_ylim(0.80, 1.015)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="lower left", fontsize=8.5)

    axes[1].set_title("Per-category F1 score (hard decisions)", loc="left", fontsize=11.5, color=INK, pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_labels, fontsize=9)
    axes[1].set_ylim(0.0, 0.92)
    axes[1].set_ylabel("F1 score")
    axes[1].grid(axis="y", color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="upper right", fontsize=8.5)
    axes[1].annotate("At N=8,000, k-NN's local voting\nbeats MLP on rare classes",
                     xy=(4.0, 0.635), xytext=(2.25, 0.64),
                     fontsize=8.8, color=INK, bbox=dict(fc=SURF, ec=GRID, pad=3), zorder=6,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    save(fig, out_dir, "per-category-f1.png")

    # -------------------------------------------------------------------------
    # 5. Performance & Running Costs (performance-and-cost.png)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    # Left panel: Single-comment end-to-end latency breakdown (median ms)
    approaches_lat = [
        "Local MiniLM + PyTorch MLP\n(CPU / Browser ONNX)",
        "Local MiniLM + k-NN (K=7)\n(CPU, N=8,000 vectors)",
        "Jev System One API\n(6 Noul questions / 1 HTTP call)",
        "Cloud Embedding + PyTorch MLP\n(HTTPS embed + local MLP)",
        "Cloud Embedding + k-NN (K=7)\n(HTTPS embed + 8k search)",
    ]
    embed_ms = np.array([9.18, 9.18, 0.0, 260.80, 260.80])
    classify_ms = np.array([0.04, 1.71, 194.96, 0.05, 3.72])
    y_pos = np.arange(len(approaches_lat))

    axes[0].barh(y_pos, embed_ms, color=BLUE, height=0.54, label="Embedding step (local CPU or HTTPS API)")
    axes[0].barh(y_pos, classify_ms, left=embed_ms, color=AQUA, height=0.54, label="Decision / classifier step (MLP, k-NN, or Jev)")

    totals = embed_ms + classify_ms
    for idx, tot in enumerate(totals):
        if idx == 0:
            detail = f"{tot:.1f} ms (9.2 ms embed + 0.04 ms MLP)"
        elif idx == 1:
            detail = f"{tot:.1f} ms (9.2 ms embed + 1.7 ms search)"
        elif idx == 2:
            detail = f"{tot:.1f} ms median (p10: 182, p90: 224 ms)"
        elif idx == 3:
            detail = f"{tot:.1f} ms (260.8 ms API + 0.05 ms MLP)"
        else:
            detail = f"{tot:.1f} ms (260.8 ms API + 3.7 ms search)"
        axes[0].text(tot + 6, idx, detail, va="center", ha="left", fontsize=8.5, color=INK)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(approaches_lat, fontsize=9.2)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 485)
    axes[0].set_xlabel("Median end-to-end latency per comment (ms)")
    axes[0].set_title("Single-comment latency: local vs. network round-trip", loc="left", fontsize=11.5, color=INK, pad=12)
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=8.5)

    # Right panel: Classifier memory footprint scaling with training set size N
    with open("results/knn_sweep_gemini.json", "r", encoding="utf-8") as f:
        knn_s = json.load(f)

    ns_scale = [entry["n_train"] for entry in knn_s]
    knn_mem_mb = [(n * 768 * 4) / (1024 * 1024) for n in ns_scale]
    mlp_mem_mb = [(99206 * 4) / (1024 * 1024) for _ in ns_scale]

    axes[1].plot(ns_scale, knn_mem_mb, color=ORANGE, marker="^", linewidth=2.2,
                 label="k-NN reference index in RAM (768d float32)")
    axes[1].plot(ns_scale, mlp_mem_mb, color=BLUE, marker="o", linewidth=2.2,
                 label="PyTorch MLP weights (~390 KB constant)")
    axes[1].axhline(0.0, color=AQUA, linewidth=2.2, linestyle="--",
                    label="Jev (0 MB local weights / 0 vectors hosted)")

    axes[1].annotate("23.4 MB for 8,000 vectors\n(scales linearly O(N) with data)",
                     xy=(8000, knn_mem_mb[-1]), xytext=(350, 17.5), fontsize=8.8, color=INK,
                     bbox=dict(fc=SURF, ec=GRID, pad=3), zorder=6,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    axes[1].annotate("MLP stays ~0.38 MB\nregardless of N",
                     xy=(1000, mlp_mem_mb[-1]), xytext=(150, 5.5), fontsize=8.8, color=INK,
                     bbox=dict(fc=SURF, ec=GRID, pad=3), zorder=6,
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

    axes[1].set_xscale("log")
    axes[1].set_xticks([15, 50, 100, 250, 500, 1000, 2000, 4000, 8000])
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].tick_params(axis="x", labelsize=9, rotation=30)
    axes[1].set_xlabel("Number of training examples (log scale)")
    axes[1].set_ylabel("Classifier memory footprint (MB)")
    axes[1].set_ylim(-1.2, 27.5)
    axes[1].set_title("Model footprint as the dataset grows", loc="left", fontsize=11.5, color=INK, pad=10)
    axes[1].grid(color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="upper left", fontsize=8.4)

    save(fig, out_dir, "performance-and-cost.png")

    # -------------------------------------------------------------------------
    # 6. Embedding Model Impact: Cloud 768d vs Local MiniLM 384d (embedding-comparison.png)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    fig.subplots_adjust(wspace=0.34)

    # Left panel: Per-category F1 comparison across the 4 embedding-based models (768d vs 384d)
    cat_keys = ["toxic", "obscene", "insult", "identity_hate", "severe_toxic", "threat"]
    cat_indices = [LABELS.index(c) for c in cat_keys]
    cat_names = [
        f"{c.replace('_', chr(10))}\n(n={int(y_true[:, LABELS.index(c)].sum())})"
        for c in cat_keys
    ]
    x_cat = np.arange(len(cat_keys))
    w = 0.19

    embed_series = [
        ("k-NN K=7 (Cloud 768d)", p_knn_gem, ORANGE, 1.0, None),
        ("PyTorch MLP (Cloud 768d)", p_mlp_gem, BLUE, 1.0, None),
        ("PyTorch MLP (Local MiniLM 384d)", p_mlp_mini, BLUE, 0.48, "//"),
        ("k-NN K=7 (Local MiniLM 384d)", p_knn_mini, ORANGE, 0.48, "//"),
    ]

    for s_idx, (s_label, probs_s, color, alpha, hatch) in enumerate(embed_series):
        offset = (s_idx - 1.5) * w
        f1_vals = [binary_metrics(probs_s[:, i], y_true[:, i], 0.50)[2] for i in cat_indices]
        bars = axes[0].bar(
            x_cat + offset,
            f1_vals,
            width=w * 0.90,
            color=color,
            alpha=alpha,
            edgecolor=INK if hatch else "none",
            linewidth=0.6 if hatch else 0,
            hatch=hatch,
            label=s_label,
        )
        for b, val in zip(bars, f1_vals):
            if val == 0:
                axes[0].text(b.get_x() + b.get_width() / 2, 0.015, "0.00", ha="center", va="bottom",
                             fontsize=7.0, color=INK2, rotation=90)

    axes[0].annotate("On 768d: k-NN beats MLP\n(0.805 vs 0.784)",
                     xy=(-0.28, 0.805), xytext=(0.55, 0.85),
                     fontsize=8.5, color=INK, bbox=dict(fc=SURF, ec=GRID, pad=2.5), zorder=6,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    axes[0].annotate("On MiniLM 384d: MLP beats k-NN\n(0.622 vs 0.512; k-NN drops -29 pts)",
                     xy=(0.28, 0.512), xytext=(2.35, 0.72),
                     fontsize=8.5, color=INK, bbox=dict(fc=SURF, ec=GRID, pad=2.5), zorder=6,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))

    axes[0].set_title("Per-category F1 at N=8,000: Cloud 768d vs. MiniLM 384d", loc="left", fontsize=11.2, color=INK, pad=10)
    axes[0].set_xticks(x_cat)
    axes[0].set_xticklabels(cat_names, fontsize=8.8)
    axes[0].set_ylim(0, 0.96)
    axes[0].set_ylabel("F1 score (@ threshold = 0.50)")
    axes[0].grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=8.2)

    # Right panel: Data quantity vs. Embedding quality ('toxic' F1 across sample sizes)
    milestones = [
        ("MiniLM 384d\nk-NN (N=8,000)", 0.512, ORANGE, 0.48, "//"),
        ("MiniLM 384d\nMLP (N=8,000)", 0.622, BLUE, 0.48, "//"),
        ("Cloud 768d\nk-NN (N=50)", 0.644, ORANGE, 1.0, None),
        ("Cloud 768d\nMLP (N=50)", 0.692, BLUE, 1.0, None),
        ("Jev Zero-Shot\n(N=0, @ 0.95)", 0.706, AQUA, 1.0, None),
        ("Cloud 768d\nMLP (N=8,000)", 0.784, BLUE, 1.0, None),
        ("Cloud 768d\nk-NN (N=8,000)", 0.805, ORANGE, 1.0, None),
    ]
    y_m = np.arange(len(milestones))
    for idx, (m_lbl, val, col, alp, hch) in enumerate(milestones):
        axes[1].barh(
            idx,
            val,
            height=0.58,
            color=col,
            alpha=alp,
            edgecolor=INK if hch else "none",
            linewidth=0.7 if hch else 0,
            hatch=hch,
        )
        axes[1].text(val + 0.015, idx, f"{val:.3f}", va="center", ha="left", fontsize=9.0, color=INK)

    axes[1].axhline(1.5, color=INK2, linewidth=1.1, linestyle="--")
    axes[1].text(1.15, 0.15, "8,000 rows on MiniLM 384d\nlose to 50 rows on Cloud 768d",
                 ha="right", va="center", fontsize=8.5, color=INK,
                 bbox=dict(fc=SURF, ec=GRID, pad=3))

    axes[1].set_yticks(y_m)
    axes[1].set_yticklabels([m[0] for m in milestones], fontsize=8.8)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.18)
    axes[1].set_xlabel("F1 score on 'toxic'")
    axes[1].set_title("Embedding quality beats 160× more training data", loc="left", fontsize=11.2, color=INK, pad=10)
    axes[1].grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)

    save(fig, out_dir, "embedding-comparison.png")
    print("Generated 6 publication charts in results/plots/")


if __name__ == "__main__":
    main()
