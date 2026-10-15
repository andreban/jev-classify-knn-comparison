# Notes: Jev vs. PyTorch MLP vs. k-NN (re-evaluated 2026-10-13)

## Sources
- Previous posts:
  - Fighting Comment Spam Without Training a Model: https://bandarra.me/posts/fighting-comment-spam (ZNotes `articles/jev-comment-spam`)
  - From PyTorch to Browser, part 1 (YouTube *sentiment*, 17,872 comments, `text-embedding-004`, 768->512->3, SGD): https://bandarra.me/posts/from-pytorch-to-browser-creating-a-web-friendly-ai-model
  - From PyTorch to Browser, part 2 (Jigsaw, `all-MiniLM-L6-v2`, 384->128->BN->32->BN->6, reported "98% accuracy"): https://bandarra.me/posts/from-pytorch-to-browser-a-full-client-side-solution-with-onnx-and-transformers-js
- Part 1 did NOT use Jigsaw. The 768d Jigsaw model in this repo (768->512->BN->128->BN->6) was not in a published post.
- Cloud embeddings in this benchmark are `gemini-embedding-001` (CLASSIFICATION, 768 dims, vector norm ~0.59), not `text-embedding-004`, despite the `YT-comments-text-embedding-004-...` file name.

## Data
- First 10,000 rows of Jigsaw `train.csv`; split 8,000 / 2,000 from `data/split_indices.json` (seed 1). No duplicate texts between train and val.
- Validation positives: toxic 185, insult 95, obscene 89, severe_toxic 20, identity_hate 13, threat 6. All-zeros baseline = 96.6% element accuracy.
- Training positives: toxic 786, severe_toxic 81, obscene 438, threat 27, insult 399, identity_hate 71.

## Problems found in the first (antigravity) analysis
1. Jev threshold 0.95 was chosen on the 2,000 rows it was scored on; trained models were held to 0.5. Fixed: threshold-free metrics + F1 with the threshold tuned on one random half and scored on the other (300 splits), same for every method.
2. "Jev has the highest ROC-AUC" is within noise (paired bootstrap CI includes 0). On toxic average precision Jev is significantly below the cloud MLP (-0.059, 95% CI [-0.11, -0.014]).
3. Sweep used one subset per N (first N rows) and a fixed number of epochs, so small N got ~100 optimizer steps vs ~1,600 at N=8,000 (MiniLM models at small N never converged, outputs stuck around 0.2-0.87). Fixed: 10 random subsets per N, fixed optimizer-step budget (1,600 cloud / 2,400 MiniLM).
4. Jev "zero labels" F1 of 0.706 used 2,000 labels for calibration. Added a calibration-budget curve.
5. Wrong facts in the draft: Jev macro F1 at 0.5 is 0.454 (draft said 0.369); "90.75% clean" should be 90.25%; MLP architecture described as D->128->64->6 (actually 768->512->128->6 and 384->128->32->6); MLP sizes 196 KB / 388 KB / 99,206 params (actually 53,926 params = 216 KB and 461,446 params = 1.8 MB); "edit war" example quote was invented; README re-run flags (`--sweep`, `--ksweep`) don't exist.
6. Single-comment latency numbers (9.18 ms MiniLM, 260.8 ms cloud embedding) had no script in the repo. Local ones re-measured with `src/bench_local_latency.py`; cloud embedding latency not re-measured (no API key on this machine).
7. "MLP is conservative on rare labels / k-NN wins rare labels" was mostly a 0.5-threshold artefact. With tuned thresholds macro F1: k-NN 0.57, MLP 0.54, Jev 0.51. Rare labels (6 and 13 positives) are too small to rank methods.

## Key numbers (results/reevaluation.json)
### Full data, toxic label
| Approach | ROC-AUC [95% CI] | Avg precision [95% CI] | F1 @0.5 | F1 CV threshold (median thr) |
|---|---|---|---|---|
| BERT (`bert-base-uncased`, fine-tuned) | 0.979 [0.969, 0.987] | 0.884 [0.850, 0.918] | 0.809 | 0.801 (0.44) |
| Jev | 0.966 [0.953, 0.975] | 0.792 [0.735, 0.840] | 0.511 | 0.700 (0.92) |
| MLP cloud | 0.958 [0.937, 0.975] | 0.852 [0.811, 0.891] | 0.784 | 0.771 (0.51) |
| k-NN cloud | 0.963 [0.945, 0.978] | 0.827 [0.771, 0.880] | 0.805 | 0.799 (0.71) |
| MLP MiniLM | 0.919 | 0.711 | 0.622 | 0.646 |
| k-NN MiniLM | 0.863 | 0.558 | 0.512 | 0.545 |

Macro over 6 labels (AUC / AP / F1@0.5 / F1 CV): BERT 0.941/0.544/0.399/0.494; Jev 0.967/0.528/0.454/0.519; MLP cloud 0.965/0.614/0.453/0.540; k-NN cloud 0.956/0.615/0.571/0.566; MLP MiniLM 0.902/0.456/0.324/0.424; k-NN MiniLM 0.827/0.349/0.294/0.381.

### Jev calibration budget (toxic F1 with threshold picked on N val labels, scored on the rest)
10: 0.58 | 25: 0.63 | 50: 0.66 | 100: 0.68 | 200: 0.69 | 500: 0.69 | 1000: 0.70. Median chosen threshold 0.92-0.93 for N >= 25.

### Sweep, toxic, mean of 10 random subsets (F1 @0.5 / AP)
- MLP cloud: 25: 0.59/0.73, 50: 0.53/0.75, 100: 0.74/0.84, 1000: 0.75/0.85, 8000: 0.79/0.88
- k-NN(7) cloud: 25: 0.38/0.66, 50: 0.41/0.74, 100: 0.78/0.81, 1000: 0.78/0.82, 8000: 0.81/0.83
- MLP MiniLM: 100: 0.29/0.32, 1000: 0.48/0.53, 8000: 0.64/0.72
- k-NN(7) MiniLM: 100: 0.31/0.38, 1000: 0.44/0.47, 8000: 0.51/0.56
- At N <= 50, 10th-90th percentile F1 spans ~0 to ~0.77 (depends on which positives are sampled).
- Macro AP: trained cloud models reach Jev's 0.53 at about N = 2,000.

### Label noise
- 37 val comments labeled not-toxic get Jev toxic >= 0.95. 5 of them carry another label (insult/identity_hate/obscene) without toxic. By manual reading ~26/37 are plainly hostile ("OMG, you are an imbecile.", "Stay off my talk page, stalker.", veiled threat to an admin). 10 val rows overall have another label but toxic = 0.
- Lowest-scored toxic positives include a barnstar thank-you and "Add Mobile porn as additional WP:ALSO link".

### Latency and size (results/latency_local.json, AMD Ryzen 7 5800H, 4 torch threads)
- MiniLM embed 10.0 ms median; MLP 0.045 ms (MiniLM) / 0.077 ms (cloud); k-NN over 8,000 rows 1.6 ms (384d) / 12.7 ms (768d).
- Sizes: MiniLM model 91 MB; MLPs 53,926 params (216 KB) / 461,446 params (1.8 MB); k-NN index 12.3 MB / 24.6 MB.
- Jev (recorded during eval, concurrency 16): 195 ms median, p10 182, p90 224. 945.5 input + 111.0 output tokens per comment (six questions).

## Charts (charts.py -> results/plots/)
1. `ranking-quality.png`: toxic ROC-AUC and average precision with 95% bootstrap intervals, 5 configurations.
2. `jev-threshold.png`: Jev precision / recall / F1 vs threshold on toxic, marks 0.5 and 0.92.
3. `labels-needed.png`: toxic F1 vs labeled examples: Jev (threshold picked on N labels) vs MLP and k-NN trained on N (cloud embeddings), p10-p90 bands.
4. `embedding-vs-classifier.png`: toxic average precision vs N for MLP and k-NN on cloud vs MiniLM, Jev reference line.
5. `jev-scores.png`: Jev toxic score histograms for toxic vs clean comments (article Figure 2).
6. `per-label.png`: average precision for toxic / obscene / insult (article Figure 4).
7. `hero.png` (hero.py): t-SNE of the 2,000 eval comments' cloud embeddings, toxic in orange.

Article figure order: ranking-quality (1), jev-scores (2), jev-threshold (3), per-label (4), labels-needed (5), embedding-vs-classifier (6).

## k-NN deep dive (src/knn_choose_k.py -> results/knn_choose_k.json, results/knn_*_chosen_k.jsonl)
- Article terminology: "custom models" = MLP + k-NN (both built from labels); Jev is the only one that isn't.
- Positives needed (K=7, cloud, runs with N <= 250 grouped by toxic positives in the subset): 0-3 -> F1 0.00 (AP 0.29-0.61); 4-6 -> 0.71 / 0.73; 7-10 -> 0.78 / 0.82; 11-30 -> 0.78 / 0.81-0.82; full data (786) -> 0.81 / 0.83. With 2-3 positives K=3 gets 0.60 F1, K=1 0.67.
- K chosen by leave-one-out on the 8,000 training rows (macro AP): cloud K=51, MiniLM K=101. Larger K beyond 51 flattens.
- Cloud K=51 on eval: toxic AUC 0.981, AP 0.896 [0.858, 0.926], fair-threshold F1 0.790 (K=7: 0.796), macro fair F1 0.556 (K=7: 0.568), macro AP 0.618 (K=7: 0.615). AP beats MLP by +0.043 [0.022, 0.068] (100% of bootstrap resamples) and Jev by +0.10.
- MiniLM K=101: toxic AP 0.648, AUC 0.919, fair F1 0.619.
- Takeaway: more neighbors improves ranking, not decisions; rare labels (threat: 27 train positives) get diluted at large K.

## Per-label view (labels with enough positives)
| Label | BERT (fine-tuned) | Jev AP / fair F1 | MLP cloud | k-NN cloud (K=7) |
|---|---|---|---|---|
| toxic (185) | 0.884 / 0.801 | 0.792 / 0.700 | 0.852 / 0.771 | 0.827 / 0.799 |
| obscene (89) | 0.874 / 0.775 | 0.848 / 0.738 | 0.865 / 0.788 | 0.807 / 0.751 |
| insult (95) | 0.828 / 0.710 | 0.630 / 0.630 | 0.754 / 0.696 | 0.771 / 0.726 |
- Insult gap is a definition mismatch: 112 non-insult comments get Jev insult >= 0.9, 51 of them labeled toxic (e.g. "Man, you are really a dishonest person, disgusting!!!!"). 94 of 95 labeled insults score > 0.5. 88 of 95 insults are also toxic.
- Article section "Beyond the toxic label" covers this; severe_toxic / identity_hate / threat (20 / 13 / 6) are called too small to compare.

## Network latency and cost (src/bench_network_latency.py -> results/latency_network.json)
- 60 eval comments, sequential, interleaved, from the Ryzen laptop: gemini-embedding-001 median 254 ms (p10 236, p90 280, mean 299); Jev six questions 209 ms (189-233); Jev toxic-only 206 ms (184-242).
- Tokens: embedding ~82 mean (43 median); Jev six questions ~935 in / 111 out; Jev one question ~453 in / 21 out. ~96 tokens per extra question, ~356 fixed (includes the comment).
- Pricing (checked 2026-10-13): Jev 1.13 $0.042 per M input tokens, output free (docs.typesafe.ai/models); Google lists $0.20 per M for gemini-embedding-2 (embedding-001 not listed separately; same price assumed). Per 1,000 comments: Jev 6q $0.039, Jev 1q $0.019, embedding $0.016. Jev default rate limit 80 rps / 100K tok/s.
- MLP training on 8,000 rows (this laptop, 8 threads): 2.8 s MiniLM, 7.1 s cloud.

## Fine-tuned BERT (`src/eval_bert.py` -> `results/bert.jsonl`, `results/bert_meta.json`)
- Setup: `bert-base-uncased` (`AutoModelForSequenceClassification`, 6 binary labels, `BCEWithLogitsLoss`), 3 epochs on the 8,000 training comments (`lr=2e-5`, `weight_decay=0.01`, `warmup_ratio=0.1`, `batch_size=16`, `max_length=128`, `seed=1`, `float16` autocast on Apple M1 Pro MPS).
- Why `MiniLM + MLP` lost: Fine-tuning BERT end-to-end completely closes and surpasses the gap between local models and cloud embeddings on every label with >=80 training positives:
  - `toxic` (786 train / 185 val positives): ROC-AUC 0.979 [0.969, 0.987], AP 0.884 [0.850, 0.918], F1@0.5 0.809, fair CV threshold F1 0.801 (median thr 0.44). Paired bootstrap `jev_minus_model`: toxic AUC -0.014 [-0.026, -0.001], toxic AP -0.093 [-0.141, -0.050].
  - `obscene` (438 train / 89 val): ROC-AUC 0.990, AP 0.874, F1@0.5 0.800, fair F1 0.775.
  - `insult` (399 train / 95 val): ROC-AUC 0.986, AP 0.828, F1@0.5 0.689, fair F1 0.710 (beats cloud MLP 0.754 AP and cloud k-NN 0.771 AP).
  - `severe_toxic` (81 train / 20 val): ROC-AUC 0.994, AP 0.537, fair F1 0.511 (beats cloud MLP 0.388 AP).
- Ultra-rare labels starve during 3-epoch joint fine-tuning: `threat` (27 train / 6 val positives) gets 0.800 AUC / 0.024 AP, and `identity_hate` (71 train / 13 val) gets 0.895 AUC / 0.117 AP, which pulls unweighted 6-label macro AP (0.544) and macro AUC (0.941) below cloud MLP (0.614 / 0.965) despite BERT winning all four larger labels.
- Size, latency & training time (Apple M1 Pro):
  - Parameters: 109,486,854 (437.9 MB fp32) vs 91 MB MiniLM + 216 KB MLP.
  - Single-comment inference (`results/latency_local.json`, measured on the base `bert-base-uncased` + 6-label head architecture at `max_length=128`): 43.9 ms median on CPU (4 threads, p10 36.2, p90 51.0) and 13.4 ms median on MPS (p10 11.7, p90 16.1); batched eval (bs=32) of the fine-tuned model is 4.55 ms/comment on MPS.
  - Training time (8,000 rows, 3 epochs): 544 s (~9.1 min on M1 Pro MPS, ~171 s/epoch) vs 3–7 s for the MLP on cached embeddings.
