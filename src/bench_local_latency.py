"""Single-comment latency of the local pipelines (MiniLM embedding, BERT, MLP, k-NN).

Note: BERT and MLP timings benchmark the model architecture forward pass directly
(base bert-base-uncased + 6-label classification head at max_length=128, and
ToxicityModel instances) without requiring saved fine-tuned BERT weights on disk.
Network calls (cloud embedding API, Jev) are not measured here; Jev latency is
recorded per request in results/jev.jsonl.

    PYTHONPATH=src .venv/bin/python src/bench_local_latency.py
"""
import json
import platform
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from all_minilm_l6.toxicity_model import ToxicityModel as MiniLMModel
from text_embedding_004.toxicity_model import ToxicityModel as GeminiModel
from toxicity_dataset import LABELS, get_or_create_split_indices, load_full_records

N_RUNS = 300
BERT_MAX_LENGTH = 128


def load_hf_pretrained(cls, model_name: str, **kwargs):
    try:
        return cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_name, local_files_only=False, **kwargs)


def load_sentence_transformer(model_name: str, **kwargs):
    try:
        return SentenceTransformer(model_name, local_files_only=True, **kwargs)
    except Exception:
        return SentenceTransformer(model_name, local_files_only=False, **kwargs)


def timeit(fn, n=N_RUNS):
    for _ in range(20):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p10_ms": float(np.percentile(ts, 10)), "p90_ms": float(np.percentile(ts, 90))}


def main():
    torch.set_num_threads(4)
    recs = load_full_records("SP-all-MiniLM-L6-v2.jsonl")
    tr, va = get_or_create_split_indices(num_samples=len(recs))
    texts = [recs[i]["comment_text"] for i in va[:N_RUNS]]
    out = {"machine": f"{platform.processor() or platform.machine()} / {platform.system()}", "torch_threads": 4}

    st = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    it = iter(texts * 2)
    out["minilm_embed"] = timeit(lambda: st.encode([next(it)]))
    out["comment_chars_median"] = float(np.median([len(t) for t in texts]))

    bert_tok = load_hf_pretrained(AutoTokenizer, "bert-base-uncased")
    bert_cpu = load_hf_pretrained(
        AutoModelForSequenceClassification, "bert-base-uncased", num_labels=len(LABELS)
    ).eval()
    bert_params = sum(p.numel() for p in bert_cpu.parameters())
    it_bert_cpu = iter(texts * 2)

    def run_bert_cpu():
        enc = bert_tok([next(it_bert_cpu)], padding=True, truncation=True, max_length=BERT_MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            bert_cpu(**enc)

    out["bert_cpu"] = timeit(run_bert_cpu) | {
        "params": bert_params,
        "float32_bytes": bert_params * 4,
        "max_length": BERT_MAX_LENGTH,
    }

    if torch.backends.mps.is_available():
        bert_mps = bert_cpu.to("mps")
        it_bert_mps = iter(texts * 2)

        def run_bert_mps():
            enc = bert_tok([next(it_bert_mps)], padding=True, truncation=True, max_length=BERT_MAX_LENGTH, return_tensors="pt").to("mps")
            with torch.no_grad():
                bert_mps(**enc)
            torch.mps.synchronize()

        out["bert_mps"] = timeit(run_bert_mps) | {
            "params": bert_params,
            "float32_bytes": bert_params * 4,
            "max_length": BERT_MAX_LENGTH,
        }

    for name, cls, dim, f in [("minilm", MiniLMModel, 384, "SP-all-MiniLM-L6-v2.jsonl"),
                              ("gemini", GeminiModel, 768, "YT-comments-text-embedding-004-768-10k.jsonl")]:
        m = cls().eval()
        params = sum(p.numel() for p in m.parameters())
        x = torch.randn(1, dim)
        with torch.no_grad():
            out[f"mlp_{name}"] = timeit(lambda: m(x)) | {"params": params, "float32_bytes": params * 4}
        rr = recs if name == "minilm" else load_full_records(f)
        xtr = torch.tensor([rr[i]["embeddings"] for i in tr], dtype=torch.float32)
        out[f"knn_{name}"] = timeit(lambda: torch.topk(torch.cdist(x, xtr), 7, largest=False)) | {
            "index_bytes": int(xtr.numel() * 4)}

    jev = [json.loads(l) for l in open("results/jev.jsonl")]
    lat = np.array([r["latency_ms"] for r in jev])
    out["jev_recorded"] = {"median_ms": float(np.median(lat)), "p10_ms": float(np.percentile(lat, 10)),
                           "p90_ms": float(np.percentile(lat, 90)),
                           "input_tokens_mean": float(np.mean([r["input_tokens"] for r in jev])),
                           "output_tokens_mean": float(np.mean([r["output_tokens"] for r in jev]))}
    json.dump(out, open("results/latency_local.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
