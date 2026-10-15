# Jev vs. PyTorch MLP vs. k-NN (and BERT) Classification Benchmark

Head-to-head comparison of four text classification approaches on 10,000 comments (`8,000` train / `2,000` validation, `seed=1`) from the [Kaggle Jigsaw Toxic Comment Classification Challenge][1]:

1. **Custom PyTorch MLP (`ToxicityModel`)** trained on embeddings (`768d` Cloud `CLASSIFICATION` embeddings and `384d` local `all-MiniLM-L6-v2`).
2. **Pure $k$-Nearest Neighbors ($K = 7$)** over the same embeddings.
3. **Jev (`System One`, `jev-1.13.0`)** zero-shot classification using six `Noul` questions per comment.
4. **Fine-tuned BERT (`bert-base-uncased`)** trained end-to-end on the 8,000 training comments (`src/eval_bert.py`, reported in [notes.md](notes.md)).

See [WALKTHROUGH.md](WALKTHROUGH.md) for the methodology, [article.md](article.md) for the blog post draft, and [notes.md](notes.md) for evaluation notes.

---

## Quick Start on a New Machine (No Rebuilding Required)

All cached predictions, training-size sweeps, $K$-sweeps, model weights (`.safetensors`), split indices (`data/split_indices.json`), and publication plots are **already committed in `results/`**.

### 1. Clone and set up the Python environment

```shell
git clone https://github.com/andreban/jev-classify-knn-comparison.git
cd jev-classify-knn-comparison

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch scipy sentence-transformers google-genai httpx scikit-learn matplotlib python-dotenv tqdm datasets safetensors
```

### 2. Regenerate charts & metrics immediately (uses cached `results/*.jsonl`)

```shell
# Generate the 4 publication charts in results/plots/
.venv/bin/python charts.py

# Recompute comparison summary JSON (metrics at the 0.5 threshold)
PYTHONPATH=src .venv/bin/python src/compare_results.py
```

### 3. Restore the raw dataset & precomputed embeddings (`~103 MB` download)

The three large files (`data/train.csv`, `SP-all-MiniLM-L6-v2.jsonl`, and `YT-comments-text-embedding-004-768-10k.jsonl`) are attached to the [`v1.0-data` GitHub Release](https://github.com/andreban/jev-classify-knn-comparison/releases/tag/v1.0-data). Restore them with one command:

```shell
./scripts/restore_data.sh
```

Once restored, you can re-run the PyTorch training or $k$-NN sweeps locally without any API calls:

```shell
PYTHONPATH=src .venv/bin/python src/eval_pytorch.py --embedding both
PYTHONPATH=src .venv/bin/python src/eval_knn.py --embedding both
PYTHONPATH=src .venv/bin/python src/eval_bert.py

# Fair-threshold re-evaluation, bootstrap CIs and the repeated training-size sweep
PYTHONPATH=src .venv/bin/python src/reevaluate.py

# Local single-comment latency
PYTHONPATH=src .venv/bin/python src/bench_local_latency.py
```

*(Note: `src/eval_jev.py` automatically resumes from `results/jev_checkpoint.jsonl`, which already has all 2,000 validation comments cached.)*

[1]: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/
