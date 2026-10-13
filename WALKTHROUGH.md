# Methodology Walkthrough

This repository compares three ways to classify the six Jigsaw toxicity labels on the first 10,000 comments of the Kaggle Jigsaw Toxic Comment Classification Challenge, split 8,000 / 2,000 (`data/split_indices.json`, seed 1). Headline numbers live in [notes.md](notes.md); the article is [article.md](article.md).

## Approaches
1. **PyTorch MLP** (`src/eval_pytorch.py`): the `ToxicityModel` classes in `src/all_minilm_l6/` (384 -> 128 -> BN -> 32 -> BN -> 6) and `src/text_embedding_004/` (768 -> 512 -> BN -> 128 -> BN -> 6), `BCEWithLogitsLoss`, Adam `lr=1e-4`, batch 512, 150 / 100 epochs.
2. **k-NN** (`src/eval_knn.py`): Euclidean distance, K = 7, score = fraction of positive neighbors per label.
3. **Jev** (`src/eval_jev.py`): `jev-1.13.0`, six Noul questions per comment in one request. Predictions are cached in `results/jev.jsonl`; do not re-run.

Embeddings: `all-MiniLM-L6-v2` (384d, local) and `gemini-embedding-001` with `task_type=CLASSIFICATION`, 768 dims (the `YT-comments-text-embedding-004-...` file name is historical).

## Evaluation (`src/reevaluate.py` -> `results/reevaluation.json`)
- **Threshold-free metrics**: ROC-AUC and average precision per label, with 1,000-sample bootstrap 95% intervals and paired Jev-minus-model differences.
- **Fair thresholds**: for every method and label, choose the best-F1 threshold on a random half of the 2,000 evaluation comments and score F1 on the other half; average over 300 splits. This replaces the earlier comparison where only Jev's threshold was tuned, on the same rows it was scored on.
- **Jev calibration budget**: pick Jev's threshold from N labeled evaluation comments (N = 10 ... 1,000), score on the remainder, 300 repeats.
- **Training-size sweep**: N in {15, 25, 50, 100, 250, 500, 1,000, 2,000, 4,000, 8,000}, 10 random subsets each. The MLP always gets the same optimizer-step budget as the full run (1,600 cloud / 2,400 MiniLM) so small N isn't under-trained. k-NN is evaluated with K in {1, 3, 7}. Reported metrics: toxic F1 at 0.5, toxic AUC / AP, macro AP / F1.

## Latency (`src/bench_local_latency.py` -> `results/latency_local.json`)
Single-comment median over 300 runs on CPU for MiniLM embedding, MLP forward pass and k-NN search over 8,000 rows. Jev latency is the per-request time recorded during the evaluation run (16 concurrent requests). The cloud embedding API was not timed.

## Known limitations
- `threat` (6) and `identity_hate` (13) have too few validation positives to compare methods individually.
- Labels are noisy; see the label-noise section in `notes.md`.
- Jev's question wording and the MLP hyperparameters were not tuned.
