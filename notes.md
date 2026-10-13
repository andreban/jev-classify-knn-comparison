# Notes: Jev vs. PyTorch MLP vs. k-NN Classification Benchmark Article

## Sources & Repositories
- **Workspace / Benchmark Repo**: `/Users/andreban/Projects/jev-classify-knn-comparison` (cloned from `https://github.com/andreban/jigsaw-toxic-comment-classification-challenge` with evaluation scripts added in `src/`).
- **Previous Blog Posts Referenced**:
  - [*Fighting Comment Spam Without Training a Model*](https://bandarra.me/posts/fighting-comment-spam) (`articles/jev-comment-spam` in ZNotes)
  - [*From PyTorch to Browser: Creating a Web-Friendly AI Model*](https://bandarra.me/posts/from-pytorch-to-browser-creating-a-web-friendly-ai-model)
  - [*From PyTorch to Browser: a full client-side solution with ONNX and Transformers.js*](https://bandarra.me/posts/from-pytorch-to-browser-a-full-client-side-solution-with-onnx-and-transformers-js)
- **Original k-NN Reference**: `/Users/andreban/Projects/ai/embed-classification-knn/src/bin/knn-classifier.rs`
- **Original Jev Client Reference**: `https://github.com/andreban/bandarrame_rs` (`lib/src/clients/jev/` and `lib/src/agents/comment_classifier/jev.rs`)

## Dataset & Split
- **Dataset**: Kaggle Jigsaw Toxic Comment Classification Challenge (`data/train.csv`, first `10,000` rows).
- **Split**: Deterministic 80/20 split (`torch.manual_seed(1)`), saved in `data/split_indices.json` (`8,000` train / `2,000` validation).
- **Validation Positives (`n = 2,000` comments, `12,000` binary cells)**:
  - `toxic`: `185` (`9.25%`)
  - `insult`: `95` (`4.75%`)
  - `obscene`: `89` (`4.45%`)
  - `severe_toxic`: `20` (`1.00%`)
  - `identity_hate`: `13` (`0.65%`)
  - `threat`: `6` (`0.30%`)
  - Clean (`all 6 labels == 0`): `1,805` (`90.25%` exact-match, `96.60%` element-wise accuracy for an all-zeros baseline).

## Evaluation Scripts & Cached Outputs
- `src/eval_pytorch.py`: Trains `ToxicityModel` (`384->128->64->6` for MiniLM, `768->128->64->6` for Cloud 768d) and runs training set size sweeps ($N \in \{15, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000\}$).
  - Outputs: `results/pytorch_{minilm,gemini}.jsonl`, `results/pytorch_{minilm,gemini}_meta.json`, `results/pytorch_sweep_{minilm,gemini}.json`
- `src/eval_knn.py`: Runs pure $k$-NN ($K=7$, Euclidean distance) and runs training size sweeps ($N = 15..8000$) + $K$-sweeps ($K \in \{1, 3, 5, 7, 11, 15, 21, 31\}$).
  - Outputs: `results/knn_{minilm,gemini}.jsonl`, `results/knn_{minilm,gemini}_meta.json`, `results/knn_sweep_{minilm,gemini}.json`, `results/knn_ksweep_{minilm,gemini}.json`
- `src/eval_jev.py`: Calls `POST https://api.typesafe.ai/v1/systemone` (`jev-1.13.0`) with all 6 `Noul` toxicity questions per validation comment.
  - Outputs: `results/jev.jsonl`, `results/jev_checkpoint.jsonl`, `results/jev_meta.json`
- `charts.py`: Generates all 5 publication charts in `results/plots/` using the light editorial palette (`SURF="#fcfcfb"`, `INK="#0b0b0b"`, `INK2="#52514e"`, `GRID="#e4e3df"`, `BLUE="#2a78d6"`, `ORANGE="#eb6834"`, `AQUA="#1baf7a"`).

## Key Numbers Summary

### 1. Full Dataset Comparison (`N = 8,000` train, `2,000` val)
```
Approach                       | Elem Acc | Macro AUC | Toxic AUC | Toxic F1 | Macro F1 | Micro F1
---------------------------------------------------------------------------------------------------
All-Zeros Baseline             |   96.60% |    0.5000 |    0.5000 |   0.0000 |   0.0000 |   0.0000
PyTorch MLP (Cloud 768d)       |   98.28% |    0.9653 |    0.9581 |   0.7839 |   0.4528 |   0.7287
PyTorch MLP (Local MiniLM 384d)|   97.78% |    0.9019 |    0.9190 |   0.6215 |   0.3239 |   0.6042
k-NN K=7 (Cloud 768d)          |   98.19% |    0.9564 |    0.9630 |   0.8053 |   0.5706 |   0.7468
k-NN K=7 (Local MiniLM 384d)   |   97.27% |    0.8266 |    0.8634 |   0.5121 |   0.2944 |   0.4623
Jev Zero-Shot (thr = 0.50)     |   92.95% |    0.9666 |    0.9657 |   0.5107 |   0.3688 |   0.4716
Jev Zero-Shot (thr = 0.95)     |   97.70% |    0.9666 |    0.9657 |   0.7055 |   0.4310 |   0.6788
```

### 2. Per-Category ROC-AUC and F1 (`N = 8,000`) + Embedding Model Delta (`Cloud 768d` vs `Local MiniLM 384d`)
```
Category       Support | Jev AUC  MLP(768) MLP(384)  kNN(768) kNN(384) | Jev F1(@.95) MLP(768) MLP(384)  kNN(768) kNN(384)
--------------------------------------------------------------------------------------------------------------------------
toxic              185 |  0.9657    0.9581   0.9190    0.9630   0.8634 |       0.7055   0.7839   0.6215    0.8053   0.5121
insult              95 |  0.9773    0.9735   0.9277    0.9841   0.8787 |       0.6441   0.6845   0.5963    0.7447   0.4706
obscene             89 |  0.9905    0.9896   0.9589    0.9869   0.8700 |       0.7684   0.7955   0.7261    0.7442   0.4530
severe_toxic        20 |  0.9806    0.9868   0.9569    0.9344   0.8104 |       0.3000   0.0952   0.0000    0.2069   0.0952
identity_hate       13 |  0.9842    0.9940   0.9284    0.9574   0.7942 |       0.2400   0.3529   0.0000    0.6364   0.2353
threat               6 |  0.9014    0.8898   0.7196    0.9125   0.7431 |       0.0000*  0.0000   0.0000    0.2857   0.0000
(*Jev threat F1 is 0.4000 at thr = 0.85, as its top threat predictions score 0.85–0.92.)
```
- **Embedding Quality vs. Data Quantity**:
  - `Cloud 768d` at $N=50$ achieves `0.692` (`MLP`) and `0.644` (`k-NN`) `toxic` F1, and at $N=100$ achieves `0.731` (`MLP`) and `0.761` (`k-NN`).
  - `Local MiniLM 384d` at $N=8,000$ ($160\times$ more training examples) tops out at `0.622` (`MLP`) and `0.512` (`k-NN`).
  - On `Cloud 768d`, `k-NN` beats `MLP` (`0.805` vs `0.784` `toxic` F1; `0.571` vs `0.453` Macro F1). On `Local MiniLM 384d`, `MLP` beats `k-NN` (`0.622` vs `0.512` `toxic` F1; `0.324` vs `0.294` Macro F1) because the MLP's first linear layer (`384 -> 128`) learns to filter out topic-similarity dimensions, whereas `k-NN` weights all 384 dimensions equally.

### 3. Single-Comment Latency & Token Usage
- **Local MiniLM (384d, CPU) single-comment embed**: `9.18 ms` median (`10.78 ms` mean, `p10: 8.32 ms`, `p90: 14.16 ms`).
- **Cloud Embedding API (`gemini-embedding-001`, 768d) single-comment embed**: `260.80 ms` median (`265.41 ms` mean, `p10: 240.28 ms`, `p90: 296.43 ms`).
- **PyTorch MLP single-item forward pass (CPU)**: `0.036 ms` median (384d) / `0.049 ms` median (768d); `0.0010 ms` / `0.0023 ms` per comment when batched over 2,000 comments.
- **k-NN ($K=7$, $N=8,000$) single-item search (CPU)**: `1.71 ms` median (384d) / `3.72 ms` median (768d); `0.0139 ms` / `0.0177 ms` per comment when batched over 2,000 comments.
- **Jev (`jev-1.13.0`, 6 `Noul` questions in one HTTP call)**:
  - Latency: `194.96 ms` median (`202.52 ms` mean, `p10: 182.10 ms`, `p90: 224.04 ms`).
  - Tokens: `945.5` input tokens and `111.0` output tokens per comment (`1,891,010` input / `222,000` output tokens across all 2,000 validation comments).

## Charts Attached to the Article Note
1. `roc-curves.png`: ROC curves on `toxic` across all 5 configurations with `toxic` AUC and 6-category Macro AUC in legend, plus Jev `0.50` vs `0.95` operating points.
2. `threshold-calibration.png`: Two-panel threshold sweep (`0.10` to `0.98`) comparing zero-shot Jev against the trained PyTorch MLP.
3. `data-efficiency.png`: Two-panel training set size sweep ($N = 15$ to $8,000$, log scale) showing `toxic` F1 (including the $k$-NN `0.00` F1 cliff at $N \le 25$) and 6-category Macro ROC-AUC.
4. `per-category-f1.png`: Per-category ROC-AUC and F1 score across all 6 Jigsaw labels (`toxic`, `insult`, `obscene`, `severe_toxic`, `identity_hate`, `threat`).
5. `embedding-comparison.png`: Two-panel comparison of `Cloud 768d` vs `Local MiniLM 384d` embeddings across both `PyTorch MLP` and `k-NN` (showing the classifier winner flip at $N=8,000$ and how $N=50$ on Cloud 768d beats $N=8,000$ on MiniLM 384d).
6. `performance-and-cost.png`: Single-comment end-to-end latency breakdown (embedding step vs. classifier step) and classifier memory footprint scaling ($O(1)$ MLP weights vs. $O(N)$ $k$-NN vector index).
