# Agent Instructions (`jev-classify-knn-comparison`)

This repository contains the benchmark code, cached evaluation outputs, chart generator, and blog post draft comparing four text classification approaches on 10,000 comments from the **Kaggle Jigsaw Toxic Comment Classification Challenge** (`8,000` train / `2,000` validation, deterministic `seed=1` split across 6 binary labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`):

1. **Custom PyTorch MLP (`ToxicityModel`)** trained on embeddings (`768d` Cloud `CLASSIFICATION` embeddings and `384d` local `sentence-transformers/all-MiniLM-L6-v2`).
2. **Pure $k$-Nearest Neighbors ($K = 7$)** over the same embeddings.
3. **Jev (`System One`, `jev-1.13.0`)** zero-shot classification using six `Noul` questions per comment.
4. **Fine-tuned BERT (`bert-base-uncased`)** trained end-to-end on the 8,000 training comments (`src/eval_bert.py`). Added after `article.md` was published; it is the subject of the follow-up draft `article-bert.md` (metadata in `metadata-bert.md`, charts from `charts_bert.py`). Do not fold BERT into `article.md`, and keep BERT in `LATER_CONFIGS` in `src/reevaluate.py` so the published numbers stay reproducible.

---

## 1. Resuming on a New Machine (Do Not Rebuild Embeddings or Re-run APIs)

> [!IMPORTANT]
> **Do not re-run the Jev API evaluation or regenerate the 10,000-row embedding datasets from scratch.** All predictions, sweeps, and embeddings are already computed and saved.

### Already Committed in Git (`results/` & root)
- **Cached Predictions (2,000 validation rows)**:
  - `results/jev.jsonl`, `results/jev_checkpoint.jsonl`, `results/jev_meta.json`
  - `results/bert.jsonl`, `results/bert_meta.json` (fine-tuned `bert-base-uncased` from `src/eval_bert.py`)
  - `results/bert_sweep.json`, `results/bert_sweep_runs.jsonl` (BERT training-size sweep, `src/sweep_bert.py`)
  - `results/latency_bert.json` (BERT latency per machine, `src/bench_bert_latency.py`; `results/latency_local.json` holds the published Ryzen numbers and must not be overwritten from another machine)
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
  - `results/reevaluation.json` (fair-threshold metrics, bootstrap CIs, Jev calibration budget, 10x repeated sweeps) from `src/reevaluate.py`
  - `results/latency_local.json` from `src/bench_local_latency.py`
  - `charts.py` -> outputs the 6 publication charts in `results/plots/` (reads `results/reevaluation.json` and `results/jev.jsonl`), in article order:
    - `ranking-quality.png` (Figure 1), `jev-scores.png` (2), `jev-threshold.png` (3), `per-label.png` (4), `labels-needed.png` (5), `embedding-vs-classifier.png` (6)
  - `hero.py` -> `results/plots/hero.png` (data-art hero candidate, not used; the article hero is the pixel-art `hero-two-bouncers.png`, prompt in `metadata.md`)
  - `metadata.md` (title, slug, URL, description, tags, hero) mirrored to the ZNotes `metadata.md` note

### Restoring the Raw Dataset & Precomputed Embeddings (`v1.0-data` Release)
If `data/train.csv`, `SP-all-MiniLM-L6-v2.jsonl`, or `YT-comments-text-embedding-004-768-10k.jsonl` are not present locally, restore them from the [`v1.0-data` GitHub Release](https://github.com/andreban/jev-classify-knn-comparison/releases/tag/v1.0-data) with:

```shell
./scripts/restore_data.sh
```

### Python Environment Setup
```shell
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch scipy sentence-transformers google-genai httpx scikit-learn matplotlib python-dotenv tqdm datasets safetensors
```

### Common Commands
```shell
# Regenerate the 6 publication charts in results/plots/ (no dataset restoration needed)
.venv/bin/python charts.py

# Re-run the re-evaluation (full-data part uses cached predictions; sweep needs ./scripts/restore_data.sh, ~10 min on 16 cores)
PYTHONPATH=src .venv/bin/python src/reevaluate.py

# Choose k-NN K by leave-one-out on the training rows (needs restored data)
PYTHONPATH=src .venv/bin/python src/knn_choose_k.py

# Recompute results/comparison_summary.json
PYTHONPATH=src .venv/bin/python src/compare_results.py

# Re-run local PyTorch or k-NN evaluations (requires ./scripts/restore_data.sh first)
PYTHONPATH=src .venv/bin/python src/eval_pytorch.py --embedding both
PYTHONPATH=src .venv/bin/python src/eval_knn.py --embedding both
```

---

## 2. ZNotes Integration (`/articles/jev-classify-knn-comparison`)

The article draft, notes, chart script, and all 6 chart images are stored in **ZNotes** under `/articles/jev-classify-knn-comparison`:

- **Directory Path**: `articles/jev-classify-knn-comparison` (UUID: `da707f14-3804-41e7-8483-a20f0881c153`)
- **`article.md` Note UUID**: `0866d5e2-cc91-4f4a-9016-020ae63a1b5f`
- **`notes.md` Note UUID**: `c1c2608b-256d-4c5c-ad97-413d1a3fe393`
- **`metadata.md` Note UUID**: `2f22ab5e-8b2d-4965-b513-5913efa05b0a`

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
- **Core Narrative Takeaways** (re-evaluated 2026-10-13, see `notes.md` for the problems found in the first analysis):
  1. **Ranking is a tie on ROC-AUC, not a Jev win**: Jev `0.966` vs. cloud MLP `0.958` / k-NN `0.963` on `toxic`, all within bootstrap noise. On `toxic` average precision Jev (`0.79`) is significantly behind the cloud MLP (`0.85`).
  2. **With fair thresholds for everyone** (picked on half the eval set, scored on the other half), `toxic` F1 is k-NN cloud `0.80`, MLP cloud `0.77`, Jev `0.70`, MLP MiniLM `0.65`, k-NN MiniLM `0.55`. Jev's best threshold is ~`0.92`; at `0.5` its F1 is `0.51`.
  3. **Labels needed**: Jev needs ~`50` labels to pick a threshold (`0.66` F1). With cloud embeddings the MLP/k-NN pass Jev on `toxic` at ~`100` labels (~10 positives); below that they are high-variance. Across all six labels, trained models only match Jev's macro average precision (`0.53`) at ~`2,000` labels.
  4. **The embedding model matters more than the classifier**: MLP on `100` cloud-embedded rows (`0.84` AP) beats MLP on `8,000` MiniLM rows (`0.72` AP).
  5. **k-NN**: ~4 positives to flag anything at K=7, ~10 to plateau. K chosen by leave-one-out on train (`src/knn_choose_k.py`) is 51: best ranking in the benchmark (`toxic` AP `0.90`, AUC `0.98`) but no F1 gain. The article calls MLP + k-NN the "custom models".
  6. **Label noise**: ~two-thirds of Jev's 37 most confident `toxic` false positives are plainly hostile comments the raters labeled clean.
  7. **Performance and cost**: MiniLM + MLP ~`10 ms` locally and free. Sequential from the same laptop: embedding API `254 ms`, Jev six questions `209 ms`, one question `206 ms`. Per 1,000 comments: Jev six questions ~$0.039, one question ~$0.019, embedding ~$0.016 (`src/bench_network_latency.py`). Labels are the dominant cost of the custom models.
