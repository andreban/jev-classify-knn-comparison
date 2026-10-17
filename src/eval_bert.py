"""Fine-tune bert-base-uncased end-to-end on the 8,000 Jigsaw training comments
and evaluate on the 2,000 validation comments.

Outputs:
- results/bert.jsonl (2,000 validation predictions matching the benchmark schema)
- results/bert_meta.json (training/inference metadata and timing)

Usage:
    PYTHONPATH=src .venv/bin/python src/eval_bert.py
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from toxicity_dataset import LABELS, get_or_create_split_indices, load_text_records


DEFAULT_MODEL = "bert-base-uncased"


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_device_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def autocast_ctx(device: torch.device):
    if device.type in ("mps", "cuda"):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def load_hf_pretrained(cls, model_name: str, **kwargs):
    try:
        return cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_name, local_files_only=False, **kwargs)


def make_length_grouped_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Sort samples by length (with a little jitter), cut the sorted list into
    batches, and shuffle the batch order. Similar-length comments share a batch,
    which keeps padding low; collate_encoded pads each batch to a multiple of 64."""
    jitter = rng.uniform(-2.0, 2.0, size=len(lengths))
    order = np.argsort(lengths + jitter)
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    batch_perm = rng.permutation(len(batches))
    return [batches[i] for i in batch_perm]


def collate_encoded(
    encoded_list: list[dict], indices: np.ndarray, pad_id: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], int]:
    max_len = max(len(encoded_list[i]["input_ids"]) for i in indices)
    bucket_len = ((max_len + 63) // 64) * 64
    bs = len(indices)
    input_ids = torch.full((bs, bucket_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bs, bucket_len), dtype=torch.long)
    for row_idx, idx in enumerate(indices):
        ids = encoded_list[int(idx)]["input_ids"]
        n = len(ids)
        input_ids[row_idx, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row_idx, :n] = 1
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }, bucket_len


def evaluate_model(
    model: nn.Module,
    val_encoded: list[dict],
    y_val: np.ndarray,
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    model.eval()
    lengths = np.array([len(x["input_ids"]) for x in val_encoded])
    order = np.argsort(lengths)
    probs_sorted = np.zeros((len(val_encoded), len(LABELS)), dtype=np.float64)

    sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad(), autocast_ctx(device):
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            enc, _ = collate_encoded(val_encoded, batch_idx, pad_id, device)
            logits = model(**enc).logits.float()
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_sorted[start : start + len(batch_idx)] = probs
            sync_device(device)
    inf_time_s = time.perf_counter() - t0

    probs_arr = np.zeros_like(probs_sorted)
    probs_arr[order] = probs_sorted
    val_acc = float(((probs_arr >= 0.5) == y_val).mean())
    return probs_arr, val_acc, inf_time_s


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for multi-label toxicity classification.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="HuggingFace model checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--max-length", type=int, default=128, help="Max token sequence length")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Fraction of steps for linear warmup")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", default=None, help="Device override (cpu, mps, cuda)")
    parser.add_argument("--output-file", default="results/bert.jsonl", help="Predictions JSONL output path")
    parser.add_argument("--meta-file", default="results/bert_meta.json", help="Metadata JSON output path")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    records = load_text_records()
    train_idx, val_idx = get_or_create_split_indices(num_samples=len(records), val_pct=0.2, seed=1)

    train_texts = [records[i]["comment_text"] for i in train_idx]
    y_train = np.array([records[i]["labels"] for i in train_idx], dtype=np.float32)
    val_texts = [records[i]["comment_text"] for i in val_idx]
    y_val = np.array([records[i]["labels"] for i in val_idx], dtype=np.int32)

    print(
        f"=== Fine-tuning '{args.model_name}' on {len(train_idx)} comments "
        f"(val={len(val_idx)}, device={device}, epochs={args.epochs}, bs={args.batch_size}, max_len={args.max_length}) ===",
        flush=True,
    )

    tokenizer = load_hf_pretrained(AutoTokenizer, args.model_name)
    pad_id = tokenizer.pad_token_id or 0

    t_tok0 = time.perf_counter()
    train_enc_raw = tokenizer(train_texts, truncation=True, max_length=args.max_length, padding=False)
    val_enc_raw = tokenizer(val_texts, truncation=True, max_length=args.max_length, padding=False)
    train_encoded = [{"input_ids": ids} for ids in train_enc_raw["input_ids"]]
    val_encoded = [{"input_ids": ids} for ids in val_enc_raw["input_ids"]]
    train_lengths = np.array([len(x["input_ids"]) for x in train_encoded])
    print(f"Pre-tokenized {len(train_texts) + len(val_texts)} comments in {time.perf_counter() - t_tok0:.2f}s", flush=True)

    model = load_hf_pretrained(
        AutoModelForSequenceClassification,
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model parameters: {num_params:,} total ({trainable_params:,} trainable, ~{num_params * 4 / 1e6:.1f} MB fp32)",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = (len(train_texts) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(args.seed)
    sync_device(device)
    t_train0 = time.perf_counter()
    epoch_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = make_length_grouped_batches(train_lengths, args.batch_size, rng)
        ep_t0 = time.perf_counter()
        running_loss = 0.0
        n_steps = 0

        for step_idx, batch_indices in enumerate(batches, start=1):
            enc, _ = collate_encoded(train_encoded, batch_indices, pad_id, device)
            batch_labels = torch.tensor(y_train[batch_indices], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                logits = model(**enc).logits
                loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Synchronize each step on MPS so async command buffers don't pile up in memory
            sync_device(device)

            loss_val = float(loss.item())
            running_loss += loss_val
            n_steps += 1
            del enc, batch_labels, logits, loss

            if step_idx in (1, 5) or step_idx % 50 == 0 or step_idx == steps_per_epoch:
                elapsed = time.perf_counter() - ep_t0
                alloc_mb = torch.mps.current_allocated_memory() / 1e6 if device.type == "mps" else 0.0
                print(
                    f"  Epoch {epoch}/{args.epochs} | Step {step_idx:>3}/{steps_per_epoch} | "
                    f"Avg Loss: {running_loss / n_steps:.4f} | Elapsed: {elapsed:.1f}s | MPS Mem: {alloc_mb:.0f}MB",
                    flush=True,
                )

        sync_device(device)
        ep_train_s = time.perf_counter() - ep_t0
        empty_device_cache(device)

        probs, val_acc, val_inf_s = evaluate_model(
            model=model,
            val_encoded=val_encoded,
            y_val=y_val,
            batch_size=args.eval_batch_size,
            pad_id=pad_id,
            device=device,
        )
        toxic_auc = float(roc_auc_score(y_val[:, 0], probs[:, 0]))
        toxic_ap = float(average_precision_score(y_val[:, 0], probs[:, 0]))
        macro_auc = float(np.mean([roc_auc_score(y_val[:, j], probs[:, j]) for j in range(len(LABELS))]))
        macro_ap = float(np.mean([average_precision_score(y_val[:, j], probs[:, j]) for j in range(len(LABELS))]))

        print(
            f"--> Epoch {epoch} summary: train_time={ep_train_s:.1f}s, val_time={val_inf_s:.2f}s | "
            f"toxic AUC={toxic_auc:.4f}, toxic AP={toxic_ap:.4f} | "
            f"macro AUC={macro_auc:.4f}, macro AP={macro_ap:.4f} | "
            f"elem_acc={val_acc:.4f}",
            flush=True,
        )
        epoch_history.append({
            "epoch": epoch,
            "avg_train_loss": running_loss / max(n_steps, 1),
            "epoch_train_seconds": ep_train_s,
            "val_inference_seconds": val_inf_s,
            "val_element_accuracy": val_acc,
            "toxic_roc_auc": toxic_auc,
            "toxic_avg_precision": toxic_ap,
            "macro_roc_auc": macro_auc,
            "macro_avg_precision": macro_ap,
        })
        empty_device_cache(device)

    sync_device(device)
    total_train_s = time.perf_counter() - t_train0

    probs, val_acc, inf_time_s = evaluate_model(
        model=model,
        val_encoded=val_encoded,
        y_val=y_val,
        batch_size=args.eval_batch_size,
        pad_id=pad_id,
        device=device,
    )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    probs_list = probs.tolist()
    with open(args.output_file, "w", encoding="utf-8") as f:
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
        "approach": "bert",
        "model_name": args.model_name,
        "device": str(device),
        "num_parameters": num_params,
        "float32_bytes": num_params * 4,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "val_element_accuracy": val_acc,
        "train_time_seconds": total_train_s,
        "val_inference_time_seconds": inf_time_s,
        "per_comment_latency_ms": (inf_time_s / len(val_idx)) * 1000.0,
        "epoch_history": epoch_history,
    }
    with open(args.meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"\n[bert] Done! Saved {args.output_file} and {args.meta_file} "
        f"(total train+eval time: {total_train_s:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
