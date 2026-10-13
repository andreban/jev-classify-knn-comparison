import csv
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path

OUTPUT_FILE = "SP-all-MiniLM-L6-v2.jsonl"
MAX_RECORDS = 10000
BATCH_SIZE = 128

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # vec is 384

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
            embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
            for entry, emb in zip(batch, embeddings):
                result = {
                    'id': entry['id'],
                    'comment_text': entry['comment_text'],
                    'embeddings': emb.tolist(),
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
