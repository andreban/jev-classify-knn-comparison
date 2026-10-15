#!/usr/bin/env bash
# Run the BERT training-size sweep (best on a GPU machine, e.g. Apple Silicon)
# and push the results to the current branch. Resumes from finished runs.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f data/train.csv ] || ./scripts/restore_data.sh

PYTHONPATH=src .venv/bin/python src/sweep_bert.py --sizes 50 100 250 500 1000 --repeats 3

git add results/bert_sweep.json results/bert_sweep_runs.jsonl
git commit -m "Add BERT training-size sweep results"
git push
