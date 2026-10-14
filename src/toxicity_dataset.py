import json
from pathlib import Path
import torch

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_dataset(filename):
    full_dataset = []
    with open(filename, mode='r', encoding='utf-8') as dataset_file:
        for sample in dataset_file:
            sample = json.loads(sample)
            input = torch.tensor(sample['embeddings'])
            label = torch.tensor([
                float(sample['toxic']),
                float(sample['severe_toxic']),
                float(sample['obscene']),
                float(sample['threat']),
                float(sample['insult']),
                float(sample['identity_hate'])
            ])
            full_dataset.append((input, label))
    return full_dataset

def split_dataset(full_dataset, text_split_pct = 0.2):
    num_samples = len(full_dataset)
    num_validation = int(text_split_pct * num_samples)
    shuffled_indices = torch.randperm(num_samples)
    train_indices = shuffled_indices[:-num_validation]
    validation_indices = shuffled_indices[-num_validation:]

    training_dataset = [full_dataset[i] for i in train_indices]
    validation_dataset = [full_dataset[i] for i in validation_indices]

    return (training_dataset, validation_dataset)

def get_or_create_split_indices(num_samples=10000, val_pct=0.2, seed=1, split_path="data/split_indices.json"):
    path = Path(split_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data["train_indices"]) + len(data["validation_indices"]) == num_samples:
            return data["train_indices"], data["validation_indices"]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    shuffled = torch.randperm(num_samples, generator=gen).tolist()
    num_validation = int(val_pct * num_samples)
    train_indices = shuffled[:-num_validation]
    validation_indices = shuffled[-num_validation:]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "num_samples": num_samples,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
            },
            f,
        )
    return train_indices, validation_indices

def load_full_records(filename):
    records = []
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            records.append({
                "id": sample["id"],
                "comment_text": sample.get("comment_text", ""),
                "embeddings": sample["embeddings"],
                "labels": [float(sample[k]) for k in LABELS],
            })
    return records


def load_text_records(csv_path="data/train.csv", fallback_jsonl="SP-all-MiniLM-L6-v2.jsonl", max_records=10000):
    import csv

    csv_file = Path(csv_path)
    if csv_file.exists():
        records = []
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_records:
                    break
                records.append({
                    "id": row["id"],
                    "comment_text": row["comment_text"],
                    "labels": [float(row[k]) for k in LABELS],
                })
        return records

    return [
        {"id": r["id"], "comment_text": r["comment_text"], "labels": r["labels"]}
        for r in load_full_records(fallback_jsonl)[:max_records]
    ]
