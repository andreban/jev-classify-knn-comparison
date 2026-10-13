import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from toxicity_dataset import LABELS, get_or_create_split_indices, load_full_records

CONFIGS = {
    "minilm": {
        "dataset_file": "SP-all-MiniLM-L6-v2.jsonl",
        "output_file": "results/knn_minilm.jsonl",
        "meta_file": "results/knn_minilm_meta.json",
        "sweep_file": "results/knn_sweep_minilm.json",
        "k_sweep_file": "results/knn_ksweep_minilm.json",
    },
    "gemini": {
        "dataset_file": "YT-comments-text-embedding-004-768-10k.jsonl",
        "output_file": "results/knn_gemini.jsonl",
        "meta_file": "results/knn_gemini_meta.json",
        "sweep_file": "results/knn_sweep_gemini.json",
        "k_sweep_file": "results/knn_ksweep_gemini.json",
    },
}

DEFAULT_K = 7
K_VALUES = [1, 3, 5, 7, 11, 15, 21, 31]
SWEEP_SIZES = [15, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]


def knn_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    k: int = DEFAULT_K,
    metric: str = "euclidean",
) -> torch.Tensor:
    """
    Computes k-NN label probabilities (fraction of K nearest neighbors with label=1)
    for each validation sample across all 6 labels.
    Matches the Euclidean k-NN vote in embed-classification-knn/src/bin/knn-classifier.rs.
    """
    eff_k = min(k, len(x_train))
    if metric == "cosine":
        x_tr_norm = F.normalize(x_train, p=2, dim=-1)
        x_va_norm = F.normalize(x_val, p=2, dim=-1)
        # Cosine distance = 1 - cosine similarity
        dists = 1.0 - torch.mm(x_va_norm, x_tr_norm.t())
    else:
        dists = torch.cdist(x_val, x_train, p=2.0)

    # Indices of the K smallest distances: shape (N_val, eff_k)
    _, knn_indices = torch.topk(dists, k=eff_k, dim=1, largest=False, sorted=True)
    # Gather neighbor labels: shape (N_val, eff_k, 6)
    neighbor_labels = y_train[knn_indices]
    # Fraction of neighbors that are positive for each label: shape (N_val, 6)
    probs = neighbor_labels.mean(dim=1)
    return probs


def run_config(name: str, default_k: int = DEFAULT_K, metric: str = "euclidean"):
    cfg = CONFIGS[name]
    dataset_path = Path(cfg["dataset_file"])
    if not dataset_path.exists():
        print(f"[{name}] Skipping: {dataset_path} not found yet.")
        return

    print(f"=== Running k-NN evaluation for '{name}' ({dataset_path}, K={default_k}, metric={metric}) ===")
    records = load_full_records(str(dataset_path))
    train_idx, val_idx = get_or_create_split_indices(num_samples=len(records), val_pct=0.2, seed=1)

    x_all = torch.tensor([r["embeddings"] for r in records], dtype=torch.float32)
    y_all = torch.tensor([r["labels"] for r in records], dtype=torch.float32)

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]

    t0 = time.perf_counter()
    probs = knn_predict(x_train, y_train, x_val, k=default_k, metric=metric)
    inf_time_s = time.perf_counter() - t0

    binary = (probs >= 0.5).float()
    val_acc = float((binary == y_val).float().mean().item())

    print(
        f"[{name}] Full 8k k-NN (K={default_k}) complete: Val Element Accuracy={val_acc:.4f}, "
        f"Inference Time={inf_time_s*1000:.2f}ms ({inf_time_s/len(val_idx)*1000:.4f}ms/comment)"
    )

    Path("results").mkdir(parents=True, exist_ok=True)
    probs_list = probs.tolist()
    with open(cfg["output_file"], "w", encoding="utf-8") as f:
        for idx_in_val, orig_idx in enumerate(val_idx):
            rec = records[orig_idx]
            prob_vec = probs_list[idx_in_val]
            out_row = {
                "id": rec["id"],
                "probabilities": {label: prob_vec[i] for i, label in enumerate(LABELS)},
                "expected": {label: int(rec["labels"][i]) for i, label in enumerate(LABELS)},
            }
            f.write(json.dumps(out_row) + "\n")

    meta = {
        "approach": f"knn_{name}",
        "k": default_k,
        "metric": metric,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "val_element_accuracy": val_acc,
        "train_time_seconds": 0.0,
        "val_inference_time_seconds": inf_time_s,
        "per_comment_latency_ms": (inf_time_s / len(val_idx)) * 1000.0,
    }
    with open(cfg["meta_file"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Sweep over K values on full 8k training set
    k_sweep_results = []
    for k_val in K_VALUES:
        t_k0 = time.perf_counter()
        k_probs = knn_predict(x_train, y_train, x_val, k=k_val, metric=metric)
        t_k = time.perf_counter() - t_k0
        k_acc = float(((k_probs >= 0.5).float() == y_val).float().mean().item())
        print(f"  [K-Sweep K={k_val}] Val Element Acc={k_acc:.4f}")
        k_sweep_results.append({
            "k": k_val,
            "val_element_accuracy": k_acc,
            "inference_time_seconds": t_k,
            "probabilities": k_probs.tolist(),
        })
    with open(cfg["k_sweep_file"], "w", encoding="utf-8") as f:
        json.dump(k_sweep_results, f)

    # Sweep over training set sizes with K=default_k
    sweep_results = []
    for n_train in SWEEP_SIZES:
        if n_train > len(x_train):
            continue
        t_s0 = time.perf_counter()
        s_probs = knn_predict(x_train[:n_train], y_train[:n_train], x_val, k=default_k, metric=metric)
        t_s = time.perf_counter() - t_s0
        s_acc = float(((s_probs >= 0.5).float() == y_val).float().mean().item())
        print(f"  [Train-Size Sweep N={n_train}] Val Element Acc={s_acc:.4f}")
        sweep_results.append({
            "n_train": n_train,
            "val_element_accuracy": s_acc,
            "inference_time_seconds": t_s,
            "probabilities": s_probs.tolist(),
        })
    with open(cfg["sweep_file"], "w", encoding="utf-8") as f:
        json.dump(sweep_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", choices=["minilm", "gemini", "both"], default="both")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    args = parser.parse_args()

    targets = ["minilm", "gemini"] if args.embedding == "both" else [args.embedding]
    for t in targets:
        run_config(t, default_k=args.k, metric=args.metric)
