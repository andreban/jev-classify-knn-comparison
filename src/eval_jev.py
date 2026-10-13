import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from toxicity_dataset import LABELS, get_or_create_split_indices, load_full_records

load_dotenv()

DEFAULT_BASE_URL = "https://api.typesafe.ai"
SYSTEM_ONE_PATH = "/v1/systemone"
DEFAULT_CONCURRENCY = 16

TOXICITY_QUESTIONS = {
    "toxic": {
        "type": "noul",
        "instructions": "Is `comment` toxic (rude, disrespectful, or unreasonable language likely to make someone leave a discussion)?",
        "criteria": {
            "true": (
                "Rude, disrespectful, hostile, abusive, profane, insulting, threatening, or "
                "hateful language directed at another person or group."
            ),
            "false": (
                "Civil, constructive, neutral, or polite discussion, even if it expresses "
                "strong disagreement, criticism of an article edit, or discusses controversial topics."
            ),
        },
    },
    "severe_toxic": {
        "type": "noul",
        "instructions": "Is `comment` severely toxic (extremely hateful, aggressive, or egregiously abusive)?",
        "criteria": {
            "true": (
                "Extreme, relentless hostility, severe abuse, repeated slurs or profanity, "
                "or wishes of death or severe harm."
            ),
            "false": (
                "Civil comments, or mildly rude, sarcastic, or ordinary uncivil comments "
                "that do not reach extreme abuse."
            ),
        },
    },
    "obscene": {
        "type": "noul",
        "instructions": "Is `comment` obscene (containing swear words, profanity, vulgarity, or sexually explicit language)?",
        "criteria": {
            "true": (
                "Contains explicit profanity, curse words, vulgar slang, or sexually explicit "
                "or lewd expressions."
            ),
            "false": (
                "Uses clean language without profanity or vulgarity, even if critical or argumentative."
            ),
        },
    },
    "threat": {
        "type": "noul",
        "instructions": "Does `comment` contain a threat of physical violence, harm, or death?",
        "criteria": {
            "true": (
                "Expresses an intent, warning, or desire to inflict physical violence, injury, "
                "murder, or real-world harm on a person or group."
            ),
            "false": (
                "No threat of physical violence or death, even if insulting, angry, or warning "
                "about administrative actions like blocking or reverting edits."
            ),
        },
    },
    "insult": {
        "type": "noul",
        "instructions": "Is `comment` an insult (demeaning, humiliating, name-calling, or personally attacking someone)?",
        "criteria": {
            "true": (
                "Direct personal attack, name-calling, belittling someone's intelligence or "
                "character, or derogatory remarks aimed at an individual or group."
            ),
            "false": (
                "Criticism of ideas, edits, or arguments without personal name-calling or "
                "demeaning attacks."
            ),
        },
    },
    "identity_hate": {
        "type": "noul",
        "instructions": "Does `comment` express identity hate (hatred, slurs, or dehumanization targeting a protected identity group)?",
        "criteria": {
            "true": (
                "Attacks, slurs, or hateful generalizations targeting race, ethnicity, religion, "
                "nationality, gender, or sexual orientation."
            ),
            "false": (
                "Does not attack or demean someone based on protected identity characteristics, "
                "even if it mentions nationality, religion, or gender in a factual or editorial "
                "discussion, or is a general non-identity insult."
            ),
        },
    },
}


async def classify_one(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    record: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    payload = {
        "state": {"comment": record["comment_text"]},
        "model": model,
        "questions": TOXICITY_QUESTIONS,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    async with semaphore:
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    f"{DEFAULT_BASE_URL}{SYSTEM_ONE_PATH}",
                    headers=headers,
                    json=payload,
                    timeout=20.0,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    data = resp.json()
                    answers = data.get("answers", {})
                    probs = {
                        label: float(answers[label]["noul"])
                        for label in LABELS
                    }
                    usage = data.get("usage", {})
                    return {
                        "id": record["id"],
                        "model": data.get("model", model),
                        "probabilities": probs,
                        "expected": {
                            label: int(record["labels"][i])
                            for i, label in enumerate(LABELS)
                        },
                        "latency_ms": latency_ms,
                        "input_tokens": int(usage.get("input_tokens", 0)),
                        "output_tokens": int(usage.get("output_tokens", 0)),
                    }
                if resp.status_code in (429, 500, 502, 503, 504, 529):
                    wait_s = 1.5 * (2 ** attempt)
                    await asyncio.sleep(wait_s)
                    continue
                raise RuntimeError(f"Jev HTTP {resp.status_code}: {resp.text}")
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                if attempt == 5:
                    raise
                await asyncio.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"Failed to classify comment {record['id']}")


async def main_async(dataset_file: str, concurrency: int, model: str):
    api_key = os.environ["JEV_API_KEY"]
    records = load_full_records(dataset_file)
    _, val_idx = get_or_create_split_indices(num_samples=len(records), val_pct=0.2, seed=1)
    val_records = [records[i] for i in val_idx]

    Path("results").mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path("results/jev_checkpoint.jsonl")
    output_path = Path("results/jev.jsonl")
    meta_path = Path("results/jev_meta.json")

    cached_by_id = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cached_by_id[row["id"]] = row

    todo_records = [r for r in val_records if r["id"] not in cached_by_id]
    print(
        f"=== Running Jev evaluation: {len(val_records)} validation comments "
        f"({len(cached_by_id)} cached, {len(todo_records)} remaining, concurrency={concurrency}) ==="
    )

    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)

    t_start = time.perf_counter()
    if todo_records:
        async with httpx.AsyncClient(limits=limits) as client:
            tasks = [
                asyncio.create_task(classify_one(client, api_key, model, rec, semaphore))
                for rec in todo_records
            ]
            completed = 0
            with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
                for coro in asyncio.as_completed(tasks):
                    res = await coro
                    cached_by_id[res["id"]] = res
                    ckpt_f.write(json.dumps(res) + "\n")
                    completed += 1
                    if completed % 100 == 0 or completed == len(todo_records):
                        ckpt_f.flush()
                        elapsed = time.perf_counter() - t_start
                        print(f"  Jev progress: {len(cached_by_id)} / {len(val_records)} ({elapsed:.1f}s)")

    ordered_results = [cached_by_id[r["id"]] for r in val_records]
    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in ordered_results:
            out_f.write(json.dumps(row) + "\n")

    # Compute element-wise accuracy at 0.5 threshold
    total_elements = len(ordered_results) * len(LABELS)
    correct_elements = 0
    for row in ordered_results:
        for label in LABELS:
            pred = 1 if row["probabilities"][label] >= 0.5 else 0
            if pred == row["expected"][label]:
                correct_elements += 1
    val_acc = correct_elements / total_elements

    latencies = [r["latency_ms"] for r in ordered_results]
    total_in_tokens = sum(r["input_tokens"] for r in ordered_results)
    total_out_tokens = sum(r["output_tokens"] for r in ordered_results)
    model_used = ordered_results[0].get("model", model) if ordered_results else model

    meta = {
        "approach": "jev",
        "model": model_used,
        "train_size": 0,
        "val_size": len(ordered_results),
        "val_element_accuracy": val_acc,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "mean_input_tokens_per_comment": total_in_tokens / len(ordered_results) if ordered_results else 0.0,
        "mean_output_tokens_per_comment": total_out_tokens / len(ordered_results) if ordered_results else 0.0,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Jev evaluation complete! Model={model_used}, Val Element Accuracy (thr=0.5)={val_acc:.4f}, "
        f"Mean Latency={meta['mean_latency_ms']:.1f}ms, Total Tokens={total_in_tokens} in / {total_out_tokens} out"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SP-all-MiniLM-L6-v2.jsonl")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--model", default=os.getenv("JEV_MODEL", "jev-latest"))
    args = parser.parse_args()

    asyncio.run(main_async(args.dataset, args.concurrency, args.model))
