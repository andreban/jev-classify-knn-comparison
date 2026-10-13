import csv
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_GENAI_API_KEY = os.environ["GEMINI_API_KEY"]
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
OUTPUT_DIMENSIONALITY = 768
OUTPUT_FILE = "YT-comments-text-embedding-004-768-10k.jsonl"
MAX_RECORDS = 10000
BATCH_SIZE = 100

client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)

skip_lines = 0
output_path = Path(OUTPUT_FILE)
if output_path.exists():
    with open(output_path, "r", encoding="utf-8") as file:
        skip_lines = sum(1 for _ in file)

print(f"Existing records in {OUTPUT_FILE}: {skip_lines}")
with open("data/train.csv", "r", encoding="utf-8") as dataset_file:
    dataset_csv = csv.DictReader(dataset_file)
    entries = []
    for i, entry in enumerate(dataset_csv):
        if i >= MAX_RECORDS:
            break
        if i < skip_lines:
            continue
        entries.append(entry)

if entries:
    with open(output_path, "a", encoding="utf-8") as out_f:
        for start in range(0, len(entries), BATCH_SIZE):
            batch = entries[start : start + BATCH_SIZE]
            texts = [e['comment_text'] for e in batch]

            for attempt in range(6):
                try:
                    embedding_response = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=texts,
                        config=types.EmbedContentConfig(
                            task_type="CLASSIFICATION",
                            output_dimensionality=OUTPUT_DIMENSIONALITY,
                        ),
                    )
                    break
                except Exception as exc:
                    wait_s = 2 ** attempt
                    print(f"Batch failed (attempt {attempt + 1}/6): {exc}. Retrying in {wait_s}s...")
                    time.sleep(wait_s)
            else:
                raise RuntimeError("Exceeded max retries for Gemini embedding batch")

            for entry, emb_obj in zip(batch, embedding_response.embeddings):
                result = {
                    'id': entry['id'],
                    'comment_text': entry['comment_text'],
                    'embeddings': emb_obj.values,
                    'toxic': entry['toxic'],
                    'severe_toxic': entry['severe_toxic'],
                    'obscene': entry['obscene'],
                    'threat': entry['threat'],
                    'insult': entry['insult'],
                    'identity_hate': entry['identity_hate'],
                }
                out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            print(f"Processed {skip_lines + start + len(batch)} / {MAX_RECORDS}")
