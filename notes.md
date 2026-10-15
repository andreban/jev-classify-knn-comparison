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
| Jev | 0.966 [0.953, 0.975] | 0.792 [0.735, 0.840] | 0.511 | 0.700 (0.92) |
| MLP cloud | 0.958 [0.937, 0.975] | 0.852 [0.811, 0.891] | 0.784 | 0.770 (0.51) |
| k-NN cloud | 0.963 [0.945, 0.978] | 0.827 [0.771, 0.880] | 0.805 | 0.798 (0.71) |
| MLP MiniLM | 0.919 | 0.711 | 0.622 | 0.646 |
| k-NN MiniLM | 0.863 | 0.558 | 0.512 | 0.547 |

Macro over 6 labels (AUC / AP / F1@0.5 / F1 CV): Jev 0.967/0.528/0.454/0.514; MLP cloud 0.965/0.614/0.453/0.541; k-NN cloud 0.956/0.615/0.571/0.568; MLP MiniLM 0.902/0.456/0.324/0.426; k-NN MiniLM 0.827/0.349/0.294/0.387.

### Jev calibration budget (toxic F1 with threshold picked on N val labels, scored on the rest)
10: 0.56 | 25: 0.62 | 50: 0.66 | 100: 0.68 | 200: 0.69 | 500: 0.70 | 1000: 0.70. Median chosen threshold 0.92-0.93 for N >= 25.

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
| Label | Jev AP / fair F1 | MLP cloud | k-NN cloud (K=7) |
|---|---|---|---|
| toxic (185) | 0.792 / 0.700 | 0.852 / 0.770 | 0.827 / 0.798 |
| obscene (89) | 0.848 / 0.736 | 0.865 / 0.786 | 0.807 / 0.750 |
| insult (95) | 0.630 / 0.631 | 0.754 / 0.697 | 0.771 / 0.731 |
- Insult gap is a definition mismatch: 112 non-insult comments get Jev insult >= 0.9, 51 of them labeled toxic (e.g. "Man, you are really a dishonest person, disgusting!!!!"). 94 of 95 labeled insults score > 0.5. 88 of 95 insults are also toxic.
- Article section "Beyond the toxic label" covers this; severe_toxic / identity_hate / threat (20 / 13 / 6) are called too small to compare.

## Network latency and cost (src/bench_network_latency.py -> results/latency_network.json)
- 60 eval comments, sequential, interleaved, from the Ryzen laptop: gemini-embedding-001 median 254 ms (p10 236, p90 280, mean 299); Jev six questions 209 ms (189-233); Jev toxic-only 206 ms (184-242).
- Tokens: embedding ~82 mean (43 median); Jev six questions ~935 in / 111 out; Jev one question ~453 in / 21 out. ~96 tokens per extra question, ~356 fixed (includes the comment).
- Pricing (checked 2026-10-13): Jev 1.13 $0.042 per M input tokens, output free (docs.typesafe.ai/models); Google lists $0.20 per M for gemini-embedding-2 (embedding-001 not listed separately; same price assumed). Per 1,000 comments: Jev 6q $0.039, Jev 1q $0.019, embedding $0.016. Jev default rate limit 80 rps / 100K tok/s.
- MLP training on 8,000 rows (this laptop, 8 threads): 2.8 s MiniLM, 7.1 s cloud.

## Fine-tuned BERT (follow-up, added 2026-10-15; `article-bert.md`)
Not part of the published article. Published numbers above are unchanged: BERT is in `LATER_CONFIGS` in `src/reevaluate.py` and uses its own RNG, so the shared RNG stream for the original five models is untouched.

- Setup (`src/eval_bert.py` -> `results/bert.jsonl`, `results/bert_meta.json`): `bert-base-uncased`, 6-label head, `BCEWithLogitsLoss`, 3 epochs on the 8,000 training rows, lr 2e-5, weight decay 0.01, 10% warmup, batch 16, max_length 128, seed 1, float16 autocast on Apple M1 Pro MPS. One run, no tuning, final epoch reported (no selection on val). 544 s training.
- Toxic: AUC 0.979 [0.969, 0.987], AP 0.884 [0.850, 0.918], F1@0.5 0.809 (146 TP, 30 FP), fair CV F1 0.801 (median thr 0.44).
- Paired bootstrap on toxic AP (`bert_minus_model` in `results/reevaluation.json`): vs MLP cloud +0.033 [0.006, 0.064] (BERT ahead in 99% of resamples); vs k-NN cloud K=7 +0.058 [0.016, 0.100]; vs Jev +0.093 [0.050, 0.141] (100%). vs k-NN cloud K=51 (0.896 AP): -0.011 [-0.035, 0.012], a tie (ad hoc check, BERT ahead in 16%).
- Fair F1 on toxic is a tie with k-NN cloud (0.801 vs 0.798), not a win.
- Per label, AP / fair F1: toxic 0.884 / 0.801; obscene 0.874 / 0.775 (MLP cloud 0.865 / 0.786: tie on AP, MLP better F1); insult 0.828 / 0.710 (best AP; k-NN cloud F1 0.731 is better); severe_toxic 0.537 / 0.511 (best, 20 eval positives).
- Rare labels fail: threat AP 0.024 (27 train / 6 val positives), max threat score over all 2,000 val rows is 0.23, positives score <= 0.09 (median rank ~121). identity_hate AP 0.117 (71 / 13), max score 0.41. Macro AP 0.544 [0.506, 0.598] (Jev 0.528, MLP cloud 0.614, k-NN cloud 0.615); macro fair F1 0.494 (below Jev 0.514).
- Label noise: of Jev's 37 clean-labeled comments with toxic >= 0.95, BERT flags 16 at 0.5 (MLP cloud 14).
- Latency (`src/bench_bert_latency.py` -> `results/latency_bert.json`, architecture forward pass, 4 threads, max_length 128): Ryzen 7 5800H CPU median 52 ms (p10 30, p90 113); M1 Pro CPU 44 ms; M1 Pro MPS 13 ms. MiniLM + MLP on the Ryzen is ~10 ms (`results/latency_local.json`, unchanged). 109.5M params, 438 MB fp32.
- Label sweep (`src/sweep_bert.py` via `scripts/run_bert_sweep.sh` -> `results/bert_sweep.json`, `results/bert_sweep_runs.jsonl`): N in {50, 100, 250, 500, 1000}, same random subsets as the MLP/k-NN sweep (`default_rng(1000 * rep + n)`, reps 0-2), 3 repeats, at least 1,500 optimizer steps per run (the full run's step count, same rule as the MLP sweep; 24 epochs at N=1000, 94 at N=250), float16 autocast on M1 Pro MPS, ~7-9 min per run, ~2 h total.
  - Toxic AP, mean of 3 (MLP cloud / MLP MiniLM on the same 3 subsets): 50: 0.63 (0.83 / 0.25); 100: 0.68 (0.85 / 0.33); 250: 0.77 (0.83 / 0.41); 500: 0.82 (0.84 / 0.44); 1000: 0.85 (0.85 / 0.55). BERT beats MLP cloud on toxic AP in 2 of 15 paired runs, MLP MiniLM in 15 of 15.
  - Passes Jev's toxic AP (0.79) at N=500 (3/3 runs; 1/3 at N=250).
  - Toxic F1 @0.5 is poor at small N: 0.31 at N=50 (MLP cloud 0.69 on the same subsets), 0.76 at N=1000. Scores rank well but are miscalibrated.
  - Macro AP: 250: 0.46, 500: 0.51, 1000: 0.53 (all 3 runs above MLP cloud on the same subsets, 0.49; Jev 0.528). About equal to the full 3-epoch run (0.544), which hints that the full run's threat/identity_hate failure is undertraining (3 epochs), not data. Per-label AP is not stored in the sweep, so this is not confirmed.
  - First attempt with a 200-step floor (kept as `results/bert_sweep_200steps.json` / `_runs.jsonl`; 2 runs CPU fp32, 13 MPS fp16): 4 of 15 runs had toxic F1 0 at 0.5 and a fifth 0.01 (scores never reached 0.5), including N=1000 rep 2 with 116 toxic positives (AP 0.76). Same undertraining problem the first MLP sweep had.
