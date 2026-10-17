"""Single-comment latency of bert-base-uncased with a 6-label head.

Kept separate from src/bench_local_latency.py so results/latency_local.json,
which the published article quotes, is not overwritten when this runs on a
different machine. Results are stored per machine in results/latency_bert.json.

Times the forward pass of the architecture (the base checkpoint plus a fresh
6-label head); the fine-tuned weights are not needed for timing. Uses the same
comments, thread count and run count as src/bench_local_latency.py.

    PYTHONPATH=src .venv/bin/python src/bench_bert_latency.py
"""
import json
import platform
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bench_local_latency import N_RUNS, timeit
from eval_bert import load_hf_pretrained
from toxicity_dataset import LABELS, get_or_create_split_indices, load_text_records

MAX_LENGTH = 128
OUT = Path("results/latency_bert.json")


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    torch.set_num_threads(4)
    recs = load_text_records()
    _, va = get_or_create_split_indices(num_samples=len(recs))
    texts = [recs[i]["comment_text"] for i in va[:N_RUNS]]

    tok = load_hf_pretrained(AutoTokenizer, "bert-base-uncased")
    model = load_hf_pretrained(AutoModelForSequenceClassification, "bert-base-uncased", num_labels=len(LABELS)).eval()
    params = sum(p.numel() for p in model.parameters())
    info = {"params": params, "float32_bytes": params * 4, "max_length": MAX_LENGTH, "torch_threads": 4}

    machine = f"{cpu_name()} / {platform.system()}"
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    entry = {}

    it = iter(texts * 2)

    def run_cpu():
        enc = tok([next(it)], truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            model(**enc)

    entry["cpu"] = timeit(run_cpu) | info

    if torch.backends.mps.is_available():
        model_mps = model.to("mps")
        it_mps = iter(texts * 2)

        def run_mps():
            enc = tok([next(it_mps)], truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to("mps")
            with torch.no_grad():
                model_mps(**enc)
            torch.mps.synchronize()

        entry["mps"] = timeit(run_mps) | info

    out[machine] = entry
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({machine: entry}, indent=1))


if __name__ == "__main__":
    main()
