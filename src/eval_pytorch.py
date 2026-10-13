import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from all_minilm_l6.toxicity_model import ToxicityModel as MiniLMToxicityModel
from text_embedding_004.toxicity_model import ToxicityModel as GeminiToxicityModel
from toxicity_dataset import LABELS, get_or_create_split_indices, load_full_records


CONFIGS = {
    "minilm": {
        "dataset_file": "SP-all-MiniLM-L6-v2.jsonl",
        "model_cls": MiniLMToxicityModel,
        "lr": 0.0001,
        "epochs": 150,
        "batch_size": 512,
        "weights_file": "SP-all-MiniLM-L6-v2.safetensors",
        "output_file": "results/pytorch_minilm.jsonl",
        "meta_file": "results/pytorch_minilm_meta.json",
        "sweep_file": "results/pytorch_sweep_minilm.json",
    },
    "gemini": {
        "dataset_file": "YT-comments-text-embedding-004-768-10k.jsonl",
        "model_cls": GeminiToxicityModel,
        "lr": 0.0001,
        "epochs": 100,
        "batch_size": 512,
        "weights_file": "YT-comments-text-embedding-004-768-10k.safetensors",
        "output_file": "results/pytorch_gemini.jsonl",
        "meta_file": "results/pytorch_gemini_meta.json",
        "sweep_file": "results/pytorch_sweep_gemini.json",
    },
}

SWEEP_SIZES = [15, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]


def train_and_evaluate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    model_cls,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int = 1,
):
    torch.manual_seed(seed)
    model = model_cls()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    effective_bs = min(batch_size, len(x_train))
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=effective_bs,
        shuffle=True,
        drop_last=(len(x_train) > effective_bs and len(x_train) % effective_bs == 1),
        generator=gen,
    )

    t0 = time.perf_counter()
    model.train()
    last_loss = 0.0
    for epoch in range(1, epochs + 1):
        for xb, yb in train_loader:
            if len(xb) < 2:
                continue
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    train_time_s = time.perf_counter() - t0

    model.eval()
    t_inf0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x_val)
        probs = torch.sigmoid(logits)
        binary = (probs >= 0.5).float()
        val_acc = float((binary == y_val).float().mean().item())
    inf_time_s = time.perf_counter() - t_inf0

    return model, probs, val_acc, last_loss, train_time_s, inf_time_s


def run_config(name: str, run_sweep: bool = True):
    cfg = CONFIGS[name]
    dataset_path = Path(cfg["dataset_file"])
    if not dataset_path.exists():
        print(f"[{name}] Skipping: {dataset_path} not found yet.")
        return

    print(f"=== Running PyTorch evaluation for '{name}' ({dataset_path}) ===")
    records = load_full_records(str(dataset_path))
    train_idx, val_idx = get_or_create_split_indices(num_samples=len(records), val_pct=0.2, seed=1)

    x_all = torch.tensor([r["embeddings"] for r in records], dtype=torch.float32)
    y_all = torch.tensor([r["labels"] for r in records], dtype=torch.float32)

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]

    model, probs, val_acc, last_loss, train_time_s, inf_time_s = train_and_evaluate(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_cls=cfg["model_cls"],
        lr=cfg["lr"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        seed=1,
    )

    print(
        f"[{name}] Full 8k train complete: Val Accuracy={val_acc:.4f}, "
        f"Train Loss={last_loss:.4f}, Train Time={train_time_s:.2f}s, "
        f"Val Inference Time={inf_time_s*1000:.2f}ms ({inf_time_s/len(val_idx)*1000:.4f}ms/comment)"
    )

    torch.save(model.state_dict(), cfg["weights_file"])

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
        "approach": f"pytorch_{name}",
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "val_element_accuracy": val_acc,
        "train_time_seconds": train_time_s,
        "val_inference_time_seconds": inf_time_s,
        "per_comment_latency_ms": (inf_time_s / len(val_idx)) * 1000.0,
    }
    with open(cfg["meta_file"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if run_sweep:
        sweep_results = []
        for n_train in SWEEP_SIZES:
            if n_train > len(x_train):
                continue
            _, s_probs, s_acc, s_loss, s_train_t, s_inf_t = train_and_evaluate(
                x_train=x_train[:n_train],
                y_train=y_train[:n_train],
                x_val=x_val,
                y_val=y_val,
                model_cls=cfg["model_cls"],
                lr=cfg["lr"],
                epochs=cfg["epochs"],
                batch_size=min(cfg["batch_size"], max(32, n_train // 4)),
                seed=1,
            )
            print(f"  [Sweep N={n_train}] Val Element Acc={s_acc:.4f}, Train Time={s_train_t:.2f}s")
            sweep_results.append({
                "n_train": n_train,
                "val_element_accuracy": s_acc,
                "train_time_seconds": s_train_t,
                "probabilities": s_probs.tolist(),
            })
        with open(cfg["sweep_file"], "w", encoding="utf-8") as f:
            json.dump(sweep_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", choices=["minilm", "gemini", "both"], default="both")
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()

    targets = ["minilm", "gemini"] if args.embedding == "both" else [args.embedding]
    for t in targets:
        run_config(t, run_sweep=not args.no_sweep)
