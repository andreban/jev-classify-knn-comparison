"""Sequential single-comment latency and token usage for the network pipelines.

For each sampled evaluation comment, interleaves: one gemini-embedding-001 call
(CLASSIFICATION, 768 dims), one Jev call with all six questions, and one Jev call
with only the `toxic` question. One request at a time, from this machine.

    PYTHONPATH=src .venv/bin/python src/bench_network_latency.py --n 60
"""
import argparse
import json
import os
import platform
import time

import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from eval_jev import DEFAULT_BASE_URL, SYSTEM_ONE_PATH, TOXICITY_QUESTIONS
from toxicity_dataset import get_or_create_split_indices, load_full_records

load_dotenv()


def stats(xs):
    xs = np.array(xs, dtype=float)
    return {"median": float(np.median(xs)), "p10": float(np.percentile(xs, 10)),
            "p90": float(np.percentile(xs, 90)), "mean": float(xs.mean()), "n": int(len(xs))}


def main(n, jev_model):
    recs = load_full_records("SP-all-MiniLM-L6-v2.jsonl")
    _, va = get_or_create_split_indices(num_samples=len(recs))
    rng = np.random.default_rng(7)
    sample = [recs[i]["comment_text"] for i in rng.choice(va, n, replace=False)]

    g = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cfg = types.EmbedContentConfig(task_type="CLASSIFICATION", output_dimensionality=768)
    jev = httpx.Client(timeout=30.0, headers={"Authorization": f"Bearer {os.environ['JEV_API_KEY']}"})
    url = f"{DEFAULT_BASE_URL}{SYSTEM_ONE_PATH}"

    def jev_call(text, questions):
        t0 = time.perf_counter()
        r = jev.post(url, json={"state": {"comment": text}, "model": jev_model, "questions": questions})
        dt = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        u = r.json().get("usage", {})
        return dt, u.get("input_tokens", 0), u.get("output_tokens", 0)

    # Warm up connections (not recorded).
    g.models.embed_content(model="gemini-embedding-001", contents=["warm up"], config=cfg)
    jev_call("warm up", {"toxic": TOXICITY_QUESTIONS["toxic"]})

    emb_ms, emb_tok, j6, j1 = [], [], [], []
    for text in sample:
        t0 = time.perf_counter()
        g.models.embed_content(model="gemini-embedding-001", contents=[text], config=cfg)
        emb_ms.append((time.perf_counter() - t0) * 1000)
        emb_tok.append(g.models.count_tokens(model="gemini-embedding-001", contents=[text]).total_tokens)
        j6.append(jev_call(text, TOXICITY_QUESTIONS))
        j1.append(jev_call(text, {"toxic": TOXICITY_QUESTIONS["toxic"]}))

    out = {
        "machine": platform.node() and "local laptop (AMD Ryzen 7 5800H)",
        "n_comments": n,
        "jev_model": jev_model,
        "embedding_gemini_001": {"latency_ms": stats(emb_ms), "input_tokens": stats(emb_tok)},
        "jev_six_questions": {"latency_ms": stats([x[0] for x in j6]), "input_tokens": stats([x[1] for x in j6]),
                              "output_tokens": stats([x[2] for x in j6])},
        "jev_toxic_only": {"latency_ms": stats([x[0] for x in j1]), "input_tokens": stats([x[1] for x in j1]),
                           "output_tokens": stats([x[2] for x in j1])},
    }
    json.dump(out, open("results/latency_network.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--jev-model", default="jev-1.13.0")
    a = ap.parse_args()
    main(a.n, a.jev_model)
