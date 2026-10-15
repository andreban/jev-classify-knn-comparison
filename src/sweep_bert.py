"""Training-set-size sweep for fine-tuned bert-base-uncased.

Uses exactly the same random training subsets as the MLP / k-NN sweep in
src/reevaluate.py (np.random.default_rng(1000 * rep + n) over the 8,000
training rows), so every BERT run can be compared with the MLP and k-NN runs
built from the same labeled comments.

Recipe: the same as src/eval_bert.py (lr 2e-5, weight decay 0.01, 10% linear
warmup, batch 16, max_length 128), trained for 3 epochs or MIN_STEPS optimizer
steps, whichever is more. Without the step floor, 100 comments would get only
21 optimizer steps, most of them spent in warmup. Nothing is tuned on the
validation set: each run is scored once, after training.

Uses a GPU (CUDA or Apple MPS) when available, one run at a time, with the
same float16 autocast as src/eval_bert.py. Without a GPU it runs several CPU
worker processes in parallel, which is slow: about 40 minutes per run on a
Ryzen 7 5800H. Each finished run is appended to results/bert_sweep_runs.jsonl,
so an interrupted sweep resumes where it stopped, even on another machine.
The summary goes to results/bert_sweep.json.

    PYTHONPATH=src .venv/bin/python src/sweep_bert.py
    PYTHONPATH=src .venv/bin/python src/sweep_bert.py --sizes 100 --repeats 1   # quick timing check
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

RUNS_PATH = Path("results/bert_sweep_runs.jsonl")
SUMMARY_PATH = Path("results/bert_sweep.json")
SIZES = [25, 50, 100, 250, 500, 1000, 2000]
REPEATS = 5
MIN_STEPS = 200
EPOCHS = 3
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
MAX_LENGTH = 128
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MODEL_NAME = "bert-base-uncased"

_DATA = {}


def _data():
    if not _DATA:
        from toxicity_dataset import get_or_create_split_indices, load_text_records

        records = load_text_records()
        tr, va = get_or_create_split_indices(num_samples=len(records))
        _DATA["train_texts"] = [records[i]["comment_text"] for i in tr]
        _DATA["y_train"] = np.array([records[i]["labels"] for i in tr], dtype=np.float32)
        _DATA["val_texts"] = [records[i]["comment_text"] for i in va]
        _DATA["y_val"] = np.array([records[i]["labels"] for i in va], dtype=np.int32)
    return _DATA


def run_job(job):
    n, rep, threads, device_name = job
    import torch
    import torch.nn as nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from eval_bert import (autocast_ctx, collate_encoded, evaluate_model, load_hf_pretrained,
                           make_length_grouped_batches, sync_device)
    from reevaluate import _metrics
    from toxicity_dataset import LABELS

    torch.set_num_threads(threads)
    device = torch.device(device_name)
    d = _data()

    # Same subset as reevaluate.sweep_job for this (n, rep).
    sub = np.random.default_rng(1000 * rep + n).permutation(len(d["train_texts"]))[:n]
    y_sub = d["y_train"][sub]

    torch.manual_seed(rep)
    tok = load_hf_pretrained(AutoTokenizer, MODEL_NAME)
    pad_id = tok.pad_token_id or 0
    enc = tok([d["train_texts"][i] for i in sub], truncation=True, max_length=MAX_LENGTH)["input_ids"]
    train_encoded = [{"input_ids": ids} for ids in enc]
    val_encoded = [{"input_ids": ids} for ids in
                   tok(d["val_texts"], truncation=True, max_length=MAX_LENGTH)["input_ids"]]
    lengths = np.array([len(x) for x in enc])

    model = load_hf_pretrained(
        AutoModelForSequenceClassification, MODEL_NAME,
        num_labels=len(LABELS), problem_type="multi_label_classification",
    ).to(device)
    steps_per_epoch = (n + BATCH_SIZE - 1) // BATCH_SIZE
    epochs = max(EPOCHS, -(-MIN_STEPS // steps_per_epoch))
    total_steps = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * WARMUP_RATIO), total_steps)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(rep)

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for b in make_length_grouped_batches(lengths, BATCH_SIZE, rng):
            batch, _ = collate_encoded(train_encoded, b, pad_id, device)
            with autocast_ctx(device):
                loss = loss_fn(model(**batch).logits, torch.tensor(y_sub[b], device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            sync_device(device)
    train_s = time.perf_counter() - t0

    probs, _, _ = evaluate_model(model, val_encoded, d["y_val"], EVAL_BATCH_SIZE, pad_id, device)
    return {
        "n": n, "rep": rep, "epochs": epochs, "steps": total_steps,
        "train_positives": y_sub.sum(0).astype(int).tolist(),
        "train_seconds": train_s, "threads": threads, "device": device_name,
        "bert": _metrics(d["y_val"].astype(np.float32), probs),
    }


def summarize(rows):
    summary = []
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        entry = {"n": n, "runs": len(rs),
                 "toxic_train_positives_median": float(np.median([r["train_positives"][0] for r in rs]))}
        for m in rs[0]["bert"]:
            v = np.array([r["bert"][m] for r in rs])
            entry[m] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
        summary.append(entry)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available)")
    ap.add_argument("--workers", type=int, default=None, help="parallel runs (default: 1 on GPU, 4 on CPU)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    args = ap.parse_args()

    from eval_bert import pick_device
    device = str(pick_device(args.device))
    workers = args.workers or (4 if device == "cpu" else 1)
    print(f"device={device} workers={workers}", flush=True)

    done = []
    if RUNS_PATH.exists():
        done = [json.loads(l) for l in RUNS_PATH.read_text().splitlines() if l.strip()]
    have = {(r["n"], r["rep"]) for r in done}
    # Largest jobs first so the pool doesn't end on one long straggler.
    jobs = [(n, rep, args.threads, device) for n in sorted(args.sizes, reverse=True)
            for rep in range(args.repeats) if (n, rep) not in have]
    print(f"{len(have)} runs cached, {len(jobs)} to go", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as ex, RUNS_PATH.open("a") as f:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            done.append(r)
            print(f"n={r['n']:>5} rep={r['rep']} steps={r['steps']:>4} {r['train_seconds']:6.0f}s "
                  f"toxic AP={r['bert']['toxic_ap']:.3f} F1={r['bert']['toxic_f1']:.3f} "
                  f"macro AP={r['bert']['macro_ap']:.3f}", flush=True)

    SUMMARY_PATH.write_text(json.dumps({
        "model": MODEL_NAME, "min_steps": MIN_STEPS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "max_length": MAX_LENGTH, "summary": summarize(done),
    }, indent=1))
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
