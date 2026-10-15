import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from toxicity_dataset import LABELS, get_or_create_split_indices, load_full_records

# Match the dark/clean aesthetic of bandarra.me plots
plt.rcParams.update({
    "figure.facecolor": "#1e272c",
    "axes.facecolor": "#263238",
    "axes.edgecolor": "#546e7a",
    "axes.labelcolor": "#eceff1",
    "text.color": "#eceff1",
    "xtick.color": "#b0bec5",
    "ytick.color": "#b0bec5",
    "grid.color": "#37474f",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.size": 11,
})

APPROACHES = [
    ("bert", "BERT (bert-base-uncased)", "results/bert.jsonl", "results/bert_meta.json", "#ab47bc"),
    ("pytorch_gemini", "PyTorch MLP (Gemini 768d)", "results/pytorch_gemini.jsonl", "results/pytorch_gemini_meta.json", "#4fc3f7"),
    ("pytorch_minilm", "PyTorch MLP (MiniLM 384d)", "results/pytorch_minilm.jsonl", "results/pytorch_minilm_meta.json", "#0288d1"),
    ("knn_gemini", "k-NN K=7 (Gemini 768d)", "results/knn_gemini.jsonl", "results/knn_gemini_meta.json", "#ffb74d"),
    ("knn_minilm", "k-NN K=7 (MiniLM 384d)", "results/knn_minilm.jsonl", "results/knn_minilm_meta.json", "#f57c00"),
    ("jev", "Jev (System One, Zero-Shot)", "results/jev.jsonl", "results/jev_meta.json", "#2bbc8a"),
]


def load_predictions(path: str):
    probs = []
    expected = []
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["id"])
            probs.append([float(row["probabilities"][l]) for l in LABELS])
            expected.append([int(row["expected"][l]) for l in LABELS])
    return ids, np.array(probs, dtype=np.float64), np.array(expected, dtype=np.int32)


def compute_metrics(probs: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(np.int32)
    element_acc = float((preds == y_true).mean())
    exact_match_acc = float((preds == y_true).all(axis=1).mean())

    # "any_toxic": comment has at least one of the 6 labels = 1
    y_any_true = (y_true.max(axis=1) > 0).astype(np.int32)
    p_any_prob = probs.max(axis=1)
    p_any_pred = (p_any_prob >= threshold).astype(np.int32)

    per_label = {}
    aucs = []
    f1s = []
    precs = []
    recs = []

    for i, label in enumerate(LABELS):
        yt = y_true[:, i]
        yp = probs[:, i]
        yb = preds[:, i]
        tp = int(((yb == 1) & (yt == 1)).sum())
        fp = int(((yb == 1) & (yt == 0)).sum())
        fn = int(((yb == 0) & (yt == 1)).sum())
        tn = int(((yb == 0) & (yt == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acc_l = (tp + tn) / len(yt)
        roc_auc_l = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.0

        aucs.append(roc_auc_l)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

        per_label[label] = {
            "support": int(yt.sum()),
            "accuracy": acc_l,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "roc_auc": roc_auc_l,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # Any-toxic binary metrics
    tp_any = int(((p_any_pred == 1) & (y_any_true == 1)).sum())
    fp_any = int(((p_any_pred == 1) & (y_any_true == 0)).sum())
    fn_any = int(((p_any_pred == 0) & (y_any_true == 1)).sum())
    tn_any = int(((p_any_pred == 0) & (y_any_true == 0)).sum())
    prec_any = tp_any / (tp_any + fp_any) if (tp_any + fp_any) > 0 else 0.0
    rec_any = tp_any / (tp_any + fn_any) if (tp_any + fn_any) > 0 else 0.0
    f1_any = (2 * prec_any * rec_any) / (prec_any + rec_any) if (prec_any + rec_any) > 0 else 0.0
    fpr_any = fp_any / (fp_any + tn_any) if (fp_any + tn_any) > 0 else 0.0
    auc_any = float(roc_auc_score(y_any_true, p_any_prob))

    # Micro-averaged across all 6 * N decisions
    yt_flat = y_true.ravel()
    yp_flat = probs.ravel()
    yb_flat = preds.ravel()
    micro_prec = float(precision_score(yt_flat, yb_flat, zero_division=0))
    micro_rec = float(recall_score(yt_flat, yb_flat, zero_division=0))
    micro_f1 = float(f1_score(yt_flat, yb_flat, zero_division=0))
    micro_auc = float(roc_auc_score(yt_flat, yp_flat))

    return {
        "threshold": threshold,
        "element_accuracy": element_acc,
        "exact_match_accuracy": exact_match_acc,
        "macro_roc_auc": float(np.mean(aucs)),
        "macro_f1": float(np.mean(f1s)),
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "micro_roc_auc": micro_auc,
        "micro_f1": micro_f1,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "any_toxic": {
            "support": int(y_any_true.sum()),
            "accuracy": (tp_any + tn_any) / len(y_any_true),
            "precision": prec_any,
            "recall": rec_any,
            "f1": f1_any,
            "fpr": fpr_any,
            "roc_auc": auc_any,
            "tp": tp_any,
            "fp": fp_any,
            "fn": fn_any,
            "tn": tn_any,
        },
        "per_label": per_label,
    }


def generate_plots(loaded: dict, y_true: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curves for 'toxic' (primary label) + Macro-AUC in legend
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    yt_toxic = y_true[:, 0]
    for key, label, _, _, color in APPROACHES:
        if key not in loaded:
            continue
        probs = loaded[key]["probs"]
        metrics = loaded[key]["metrics"]
        fpr, tpr, _ = roc_curve(yt_toxic, probs[:, 0])
        ls = "-" if "minilm" not in key else "--"
        lw = 2.6 if key == "jev" else 2.0
        ax.plot(
            fpr,
            tpr,
            label=f"{label} (Toxic AUC={metrics['per_label']['toxic']['roc_auc']:.3f}, Macro={metrics['macro_roc_auc']:.3f})",
            color=color,
            linestyle=ls,
            linewidth=lw,
        )
    ax.plot([0, 1], [0, 1], color="#78909c", linestyle=":", linewidth=1.2, label="Random (0.500)")
    ax.set_title("ROC Curve — 'toxic' Label (2,000 Validation Comments)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (Clean comments wrongly flagged)")
    ax.set_ylabel("True Positive Rate / Recall (Toxic comments caught)")
    ax.legend(loc="lower right", frameon=True, facecolor="#1e272c", edgecolor="#546e7a", fontsize=9.5)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png")
    plt.close(fig)

    # 2. Per-label ROC-AUC & F1 grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=160)
    plot_items = []
    for key, label, _, _, color in APPROACHES:
        if key not in loaded:
            continue
        if key == "jev":
            m95 = compute_metrics(loaded["jev"]["probs"], y_true, threshold=0.95)
            plot_items.append(("jev_95", "Jev Zero-Shot (thr=0.95)", loaded["jev"]["metrics"], m95, "#2bbc8a"))
        else:
            plot_items.append((key, label, loaded[key]["metrics"], loaded[key]["metrics"], color))

    x = np.arange(len(LABELS))
    width = 0.82 / max(len(plot_items), 1)

    for idx, (key, label, m_auc, m_f1, color) in enumerate(plot_items):
        aucs = [m_auc["per_label"][l]["roc_auc"] for l in LABELS]
        f1s = [m_f1["per_label"][l]["f1"] for l in LABELS]
        offset = (idx - (len(plot_items) - 1) / 2) * width
        axes[0].bar(x + offset, aucs, width=width * 0.92, label=label.replace(" (thr=0.95)", ""), color=color)
        axes[1].bar(x + offset, f1s, width=width * 0.92, label=label, color=color)

    axes[0].set_title("Per-Category ROC-AUC", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LABELS, rotation=25, ha="right")
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].grid(True, axis="y")

    axes[1].set_title("Per-Category F1 Score", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LABELS, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("F1 Score")
    axes[1].grid(True, axis="y")
    axes[1].legend(loc="upper right", frameon=True, facecolor="#1e272c", edgecolor="#546e7a", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_dir / "per_category_metrics.png")
    plt.close(fig)

    # 3. Data Efficiency Curve (Training Set Size Sweep vs. Zero-Shot Jev)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    sweep_configs = [
        ("results/pytorch_sweep_gemini.json", "PyTorch MLP (Gemini 768d)", "#4fc3f7", "-", "o"),
        ("results/pytorch_sweep_minilm.json", "PyTorch MLP (MiniLM 384d)", "#0288d1", "--", "s"),
        ("results/knn_sweep_gemini.json", "k-NN K=7 (Gemini 768d)", "#ffb74d", "-", "^"),
        ("results/knn_sweep_minilm.json", "k-NN K=7 (MiniLM 384d)", "#f57c00", "--", "D"),
    ]
    for sweep_file, label, color, ls, marker in sweep_configs:
        sp = Path(sweep_file)
        if not sp.exists():
            continue
        with open(sp, "r", encoding="utf-8") as f:
            s_data = json.load(f)
        ns = []
        macro_aucs = []
        toxic_f1s = []
        for entry in s_data:
            ns.append(entry["n_train"])
            s_probs = np.array(entry["probabilities"], dtype=np.float64)
            sm = compute_metrics(s_probs, y_true, threshold=0.5)
            macro_aucs.append(sm["macro_roc_auc"])
            toxic_f1s.append(sm["per_label"]["toxic"]["f1"])
        axes[0].plot(ns, macro_aucs, label=label, color=color, linestyle=ls, marker=marker, linewidth=2)
        axes[1].plot(ns, toxic_f1s, label=label, color=color, linestyle=ls, marker=marker, linewidth=2)

    if "jev" in loaded:
        jev_m_05 = loaded["jev"]["metrics"]
        jev_m_95 = compute_metrics(loaded["jev"]["probs"], y_true, threshold=0.95)
        axes[0].axhline(
            jev_m_05["macro_roc_auc"],
            color="#2bbc8a",
            linestyle="-",
            linewidth=2.5,
            label=f"Jev Zero-Shot (0 examples): {jev_m_05['macro_roc_auc']:.3f}",
        )
        axes[1].axhline(
            jev_m_95["per_label"]["toxic"]["f1"],
            color="#2bbc8a",
            linestyle="-",
            linewidth=2.5,
            label=f"Jev Zero-Shot @ thr=0.95: {jev_m_95['per_label']['toxic']['f1']:.3f}",
        )
        axes[1].axhline(
            jev_m_05["per_label"]["toxic"]["f1"],
            color="#2bbc8a",
            linestyle=":",
            linewidth=1.8,
            label=f"Jev Zero-Shot @ thr=0.50: {jev_m_05['per_label']['toxic']['f1']:.3f}",
        )

    axes[0].set_title("Macro ROC-AUC vs. Training Set Size", fontweight="bold")
    axes[0].set_xlabel("Number of Labeled Training Examples")
    axes[0].set_ylabel("Macro ROC-AUC (6 Categories)")
    axes[0].set_xscale("log")
    axes[0].grid(True)
    axes[0].legend(loc="lower right", frameon=True, facecolor="#1e272c", edgecolor="#546e7a", fontsize=8.5)

    axes[1].set_title("'toxic' F1 Score vs. Training Set Size", fontweight="bold")
    axes[1].set_xlabel("Number of Labeled Training Examples")
    axes[1].set_ylabel("F1 Score ('toxic' @ thr=0.5)")
    axes[1].set_xscale("log")
    axes[1].grid(True)
    axes[1].legend(loc="lower right", frameon=True, facecolor="#1e272c", edgecolor="#546e7a", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_dir / "data_efficiency_sweep.png")
    plt.close(fig)

    # 4. Threshold Sweep for Jev vs PyTorch (Gemini) on 'toxic'
    if "jev" in loaded and "pytorch_gemini" in loaded:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)
        thresholds = np.linspace(0.1, 0.95, 35)
        for ax, (key, title) in zip(axes, [("jev", "Jev (System One)"), ("pytorch_gemini", "PyTorch MLP (Gemini 768d)")]):
            probs_t = loaded[key]["probs"][:, 0]
            yt_t = y_true[:, 0]
            precs, recs, f1s, fprs = [], [], [], []
            for thr in thresholds:
                yb = (probs_t >= thr).astype(np.int32)
                tp = ((yb == 1) & (yt_t == 1)).sum()
                fp = ((yb == 1) & (yt_t == 0)).sum()
                fn = ((yb == 0) & (yt_t == 1)).sum()
                tn = ((yb == 0) & (yt_t == 0)).sum()
                p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                precs.append(p)
                recs.append(r)
                f1s.append(f1)
                fprs.append(fpr)
            ax.plot(thresholds, precs, label="Precision", color="#4fc3f7", linewidth=2)
            ax.plot(thresholds, recs, label="Recall", color="#ffb74d", linewidth=2)
            ax.plot(thresholds, f1s, label="F1 Score", color="#2bbc8a", linewidth=2.4)
            ax.plot(thresholds, fprs, label="False Positive Rate", color="#ef5350", linestyle="--", linewidth=1.8)
            ax.set_title(f"Threshold Sweep ('toxic') — {title}", fontweight="bold")
            ax.set_xlabel("Probability Threshold")
            ax.set_ylabel("Score")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True)
            ax.legend(loc="center left", frameon=True, facecolor="#1e272c", edgecolor="#546e7a")
        fig.tight_layout()
        fig.savefig(out_dir / "threshold_sweep.png")
        plt.close(fig)


def main():
    loaded = {}
    y_true_ref = None

    for key, label, pred_path, meta_path, color in APPROACHES:
        p = Path(pred_path)
        if not p.exists():
            print(f"Skipping {key}: {pred_path} not found")
            continue
        ids, probs, y_true = load_predictions(pred_path)
        if y_true_ref is None:
            y_true_ref = y_true
        metrics = compute_metrics(probs, y_true, threshold=0.5)
        meta = {}
        if Path(meta_path).exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        loaded[key] = {
            "label": label,
            "probs": probs,
            "metrics": metrics,
            "meta": meta,
        }

    if y_true_ref is None:
        print("No prediction files found.")
        return

    # All-zeros baseline accuracy
    all_zeros_acc = float((y_true_ref == 0).mean())
    all_zeros_exact = float((y_true_ref == 0).all(axis=1).mean())

    summary = {
        "validation_samples": int(len(y_true_ref)),
        "total_binary_decisions": int(y_true_ref.size),
        "all_zeros_element_accuracy": all_zeros_acc,
        "all_zeros_exact_match_accuracy": all_zeros_exact,
        "approaches": {
            k: {
                "label": v["label"],
                "meta": v["meta"],
                "metrics": v["metrics"],
            }
            for k, v in loaded.items()
        },
    }

    # Also compute threshold sweep table for Jev and PyTorch on 'toxic'
    threshold_tables = {}
    for key in loaded:
        probs_t = loaded[key]["probs"][:, 0]
        yt_t = y_true_ref[:, 0]
        rows = []
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
            yb = (probs_t >= thr).astype(np.int32)
            tp = int(((yb == 1) & (yt_t == 1)).sum())
            fp = int(((yb == 1) & (yt_t == 0)).sum())
            fn = int(((yb == 0) & (yt_t == 1)).sum())
            tn = int(((yb == 0) & (yt_t == 0)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            rows.append({
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": p,
                "recall": r,
                "f1": f1,
                "fpr": fpr,
            })
        threshold_tables[key] = rows
    summary["toxic_threshold_sweep"] = threshold_tables

    with open("results/comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    generate_plots(loaded, y_true_ref, Path("results/plots"))

    print("\n================ OVERALL COMPARISON SUMMARY ================")
    print(f"Validation Set: {len(y_true_ref)} comments ({y_true_ref.size} binary label decisions)")
    print(f"All-Zeros Baseline: Element Accuracy = {all_zeros_acc*100:.2f}%, Exact-Match = {all_zeros_exact*100:.2f}%\n")
    header = f"{'Approach':<30} | {'Elem Acc':>8} | {'Macro AUC':>9} | {'Toxic AUC':>9} | {'Toxic F1':>8} | {'Macro F1':>8} | {'Micro F1':>8}"
    print(header)
    print("-" * len(header))
    for key, label, _, _, _ in APPROACHES:
        if key not in loaded:
            continue
        m = loaded[key]["metrics"]
        print(
            f"{label:<30} | {m['element_accuracy']*100:>7.2f}% | "
            f"{m['macro_roc_auc']:>9.4f} | {m['per_label']['toxic']['roc_auc']:>9.4f} | "
            f"{m['per_label']['toxic']['f1']:>8.4f} | {m['macro_f1']:>8.4f} | {m['micro_f1']:>8.4f}"
        )


if __name__ == "__main__":
    main()
