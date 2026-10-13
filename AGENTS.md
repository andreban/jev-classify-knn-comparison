# Agent Instructions (`jev-classify-knn-comparison`)

This repository contains the benchmark code, cached evaluation outputs, chart generator, and blog post draft comparing three text classification approaches on 10,000 comments from the **Kaggle Jigsaw Toxic Comment Classification Challenge** (`8,000` train / `2,000` validation, deterministic `seed=1` split across 6 binary labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`):

1. **Custom PyTorch MLP (`ToxicityModel`)** trained on embeddings (`768d` Cloud `CLASSIFICATION` embeddings and `384d` local `sentence-transformers/all-MiniLM-L6-v2`).
2. **Pure $k$-Nearest Neighbors ($K = 7$)** over the same embeddings.
3. **Jev (`System One`, `jev-1.13.0`)** zero-shot classification using six `Noul` questions per comment.

---

## 1. Resuming on a New Machine (Do Not Rebuild Embeddings or Re-run APIs)

> [!IMPORTANT]
> **Do not re-run the Jev API evaluation or regenerate the 10,000-row embedding datasets from scratch.** All predictions, sweeps, and embeddings are already computed and saved.

### Already Committed in Git (`results/` & root)
- **Cached Predictions (2,000 validation rows)**:
  - `results/jev.jsonl`, `results/jev_checkpoint.jsonl`, `results/jev_meta.json`
  - `results/pytorch_gemini.jsonl`, `results/pytorch_gemini_meta.json`
  - `results/pytorch_minilm.jsonl`, `results/pytorch_minilm_meta.json`
  - `results/knn_gemini.jsonl`, `results/knn_gemini_meta.json`
  - `results/knn_minilm.jsonl`, `results/knn_minilm_meta.json`
- **Cached Sweeps**:
  - Training set size sweeps ($N \in \{15, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000\}$): `results/pytorch_sweep_{gemini,minilm}.json`, `results/knn_sweep_{gemini,minilm}.json`
  - $k$-NN neighbor sweeps ($K \in \{1, 3, 5, 7, 11, 15, 21, 31\}$): `results/knn_ksweep_{gemini,minilm}.json`
  - Aggregated metrics & threshold tables: `results/comparison_summary.json`
- **Split Indices & Weights**:
  - `data/split_indices.json` (exact 8,000 train / 2,000 val indices for `seed=1`)
  - `SP-all-MiniLM-L6-v2.safetensors` & `YT-comments-text-embedding-004-768-10k.safetensors`
- **Article, Notes & Plots**:
  - `article.md` (blog post draft)
  - `notes.md` (research & evaluation notes)
  - `WALKTHROUGH.md` (technical walkthrough)
  - `charts.py` -> outputs the 6 publication charts in `results/plots/`:
    - `roc-curves.png`
    - `threshold-calibration.png`
    - `data-efficiency.png`
    - `per-category-f1.png`
    - `embedding-comparison.png`
    - `performance-and-cost.png`

### Restoring the Raw Dataset & Precomputed Embeddings (`v1.0-data` Release)
If `data/train.csv`, `SP-all-MiniLM-L6-v2.jsonl`, or `YT-comments-text-embedding-004-768-10k.jsonl` are not present locally, restore them from the [`v1.0-data` GitHub Release](https://github.com/andreban/jev-classify-knn-comparison/releases/tag/v1.0-data) with:

```shell
./scripts/restore_data.sh
```

### Python Environment Setup
```shell
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch sentence-transformers google-genai httpx scikit-learn matplotlib python-dotenv tqdm datasets safetensors
```

### Common Commands
```shell
# Regenerate the 6 publication charts in results/plots/ (no dataset restoration needed; reads results/*.jsonl)
.venv/bin/python charts.py

# Recompute results/comparison_summary.json
PYTHONPATH=src .venv/bin/python src/compare_results.py

# Re-run local PyTorch or k-NN evaluations (requires ./scripts/restore_data.sh first)
PYTHONPATH=src .venv/bin/python src/eval_pytorch.py --embedding both --sweep
PYTHONPATH=src .venv/bin/python src/eval_knn.py --embedding both --sweep --ksweep
```

---

## 2. ZNotes Integration (`/articles/jev-classify-knn-comparison`)

The article draft, notes, chart script, and all 6 chart images are stored in **ZNotes** under `/articles/jev-classify-knn-comparison`:

- **Directory Path**: `articles/jev-classify-knn-comparison` (UUID: `da707f14-3804-41e7-8483-a20f0881c153`)
- **`article.md` Note UUID**: `0866d5e2-cc91-4f4a-9016-020ae63a1b5f`
- **`notes.md` Note UUID**: `c1c2608b-256d-4c5c-ad97-413d1a3fe393`

When editing `article.md`, `notes.md`, or regenerating plots in `charts.py`, keep the local files and ZNotes in sync:

```shell
# Update article.md and notes.md in ZNotes
znotes notes update 0866d5e2-cc91-4f4a-9016-020ae63a1b5f --file article.md
znotes notes update c1c2608b-256d-4c5c-ad97-413d1a3fe393 --file notes.md

# Inspect files attached to the article in ZNotes
znotes files list --path "articles/jev-classify-knn-comparison"

# Upload an updated chart and attach it to article.md
znotes files upload results/plots/<chart>.png --path "articles/jev-classify-knn-comparison" --note 0866d5e2-cc91-4f4a-9016-020ae63a1b5f --desc "<description>"
```

---

## 3. Article & Chart Style Conventions

- **Palette (`charts.py`)**: Matches `articles/jev-comment-spam/charts.py`:
  - `SURF = "#fcfcfb"`, `INK = "#0b0b0b"`, `INK2 = "#52514e"`, `GRID = "#e4e3df"`
  - `BLUE = "#2a78d6"` (PyTorch MLP), `ORANGE = "#eb6834"` ($k$-NN), `AQUA = "#1baf7a"` (Jev)
- **Core Narrative Takeaways**:
  1. **Jev does great out of the box on ranking (`0.967` Macro ROC-AUC with 0 training examples), but requires threshold calibration (`0.95` vs. `0.50`)** because zero-shot probabilities judge semantics rather than the dataset's `9.25%` base rate.
  2. **$k$-NN ($K=7$) is surprisingly strong when training data is abundant (`0.805` `toxic` F1 / `0.571` Macro F1 at $N=8,000$), but degrades rapidly when the training set is small**, hitting a mathematical `0.000` F1 cliff when positive examples drop below $\lceil K/2 \rceil = 4$ ($N \le 25$).
  3. **The embedding model matters more than the classifier architecture**: Switching from 768-d cloud `CLASSIFICATION` embeddings to 384-d local `all-MiniLM-L6-v2` drops `toxic` F1 by `16.2` points on the MLP (`0.784 -> 0.622`) and `29.3` points on $k$-NN (`0.805 -> 0.512`), flipping the winner between `MLP` and `$k$-NN`. Just `50` training rows on the 768-d cloud embeddings (`0.692` F1) beat `8,000` training rows on MiniLM (`0.622` F1).
  4. **Performance & Running Costs**: Local MiniLM + MLP runs in `9.2 ms` median (`0` API tokens, `196 KB` weights); Jev answers all 6 `Noul` questions in `195.0 ms` median (`945.5` input + `111.0` output tokens/comment), faster end-to-end than a cloud embedding HTTPS call (`260.8 ms`) + local classifier.
