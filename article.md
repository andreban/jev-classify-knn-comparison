In my [last post][spam], I replaced the comment spam filter I had been planning to train for this blog with [Jev][jev], TypeSafe's "System One" decision model. Instead of collecting thousands of labeled comments, generating embeddings, and training a classifier, I described what spam looks like in plain language and asked Jev a single yes/no question ([Noul][noul]) per comment. On the 102 comments my blog had accumulated so far, it reached a `0.922` ROC-AUC without ever seeing a training example.

At the end of that post, under *"What I didn't measure"*, I left an open question:

> I didn't build the custom model, so I can't say how Jev compares to one. A model trained on my own comments might well be more accurate: it would learn the exact flavor of spam this blog attracts, and a 0.922 AUC leaves room to improve.

This blog still only has a hundred comments, which is far too small to train a neural network. But I *do* have a dataset where I already built that exact model.

A while ago, I wrote a two-part series on building a comment classifier from scratch—first [training a PyTorch model on cloud embeddings][pytorch], and then [running the entire pipeline client-side in the browser with ONNX and Transformers.js][onnx] using a smaller local embedding model (`all-MiniLM-L6-v2`). Later, I also experimented with an even simpler baseline on the same embeddings: skipping the neural network entirely and using a pure **$k$-Nearest Neighbors ($k$-NN)** classifier.

Running all three approaches—a trained **PyTorch neural network**, **$k$-NN**, and **zero-shot Jev**—on that exact same dataset gives us a head-to-head benchmark across accuracy, data efficiency, latency, and running costs. Three results stood out immediately:

1. **Jev does great out of the box, but needs threshold calibration.** With zero training examples, Jev achieves the highest ranking score of any approach (`0.967` Macro ROC-AUC across all six categories), beating models trained on 8,000 labeled comments. Yet at a naive `0.50` threshold its F1 score looks mediocre (`0.511`), because zero-shot probabilities judge semantics rather than the dataset's class balance. Calibrating the threshold (`0.95`) cuts false positives by **9.2x** and lifts `toxic` F1 to `0.706`.
2. **$k$-NN is surprisingly good when data is abundant, but degrades rapidly when the training set is small.** With 8,000 training examples and strong classification embeddings, a 20-line $k$-NN classifier ($K=7$) with zero training time beats the trained PyTorch neural network on F1 (`0.805` vs. `0.784` on `toxic`, and `0.571` vs. `0.453` Macro F1). Shrink the training set below a few hundred examples, however, and $k$-NN hits a mathematical cliff—dropping to `0.000` F1 across every category at $N \le 25$.
3. **For both PyTorch and $k$-NN, the embedding model matters far more than the classifier.** Switching from the 768-d cloud `CLASSIFICATION` embeddings to the smaller 384-d local `all-MiniLM-L6-v2` embeddings costs **16.2 F1 points** on the PyTorch MLP and **29.3 F1 points** on $k$-NN—and flips the winner between the two. In fact, **50 labeled examples** on the cloud embeddings (`0.692` F1) beat **8,000 labeled examples ($160\times$ more data)** on MiniLM (`0.622` F1).

---

## The benchmark setup

The dataset comes from the [Kaggle Jigsaw Toxic Comment Classification Challenge][jigsaw], using the same [10,000-comment subset and code][repo] from my earlier PyTorch posts. Each Wikipedia talk-page comment is labeled across **six binary toxicity categories** (a single comment can belong to none, one, or several at once):

| Category | Positive examples in validation (`n = 2,000`) | Base rate |
|---|---|---|
| `toxic` | 185 | 9.25% |
| `insult` | 95 | 4.75% |
| `obscene` | 89 | 4.45% |
| `severe_toxic` | 20 | 1.00% |
| `identity_hate` | 13 | 0.65% |
| `threat` | 6 | 0.30% |
| **Clean (all zeros)** | **1,805** | **90.25%** |

Using a fixed random seed (`seed = 1`), the 10,000 comments are split **80/20** into **8,000 training comments** and **2,000 validation comments** (`12,000` binary decisions in total).

> [!NOTE]
> Because `90.25%` of the comments are completely clean (`96.60%` of all `12,000` individual label cells are `0`), **raw accuracy is a trap**. A dummy classifier that predicts `0` for everything achieves **96.60% element-wise accuracy** while catching `0%` of toxic comments. Throughout this post, we focus on **ROC-AUC** (how well a model ranks toxic comments above clean ones across all thresholds) and **F1 / Precision / Recall** on the positive class.

For the two embedding-based approaches, we test both embedding models from the original series:

- **Cloud classification embeddings (`768d`)**: Google AI's embedding API (`task_type="CLASSIFICATION"`, 768 dimensions), matching the first post.
- **Local browser-friendly embeddings (`384d`)**: [`sentence-transformers/all-MiniLM-L6-v2`][minilm] (384 dimensions), the compact model that runs locally on CPU or in the browser via Transformers.js.

Here are the three contenders:

1. **Custom PyTorch MLP (`ToxicityModel`)**: A 3-layer feed-forward network (`D -> 128 -> ReLU -> 64 -> ReLU -> 6`) trained with `BCEWithLogitsLoss` and `Adam` (`lr = 0.0001`, `batch_size = 512`, `100` epochs on 768-d cloud embeddings / `150` epochs on 384-d MiniLM).
2. **Pure $k$-NN ($K = 7$)**: No neural network and no training loop. To classify a comment, we compute the Euclidean distance from its embedding to all $N$ training embeddings, find the $K = 7$ closest neighbors, and average their labels (`m / 7` for each of the 6 categories). A category is predicted positive at `>= 0.50` when at least 4 of the 7 neighbors have that label.
3. **Jev (`System One`, Zero-Shot)**: Jev never sees the 8,000 training rows. For each validation comment, we send the comment text as JSON state (`{"comment": {"text": "…"}}`) and ask **six `Noul` questions in a single request**—one per Jigsaw category—returning six probabilities between `0.0` and `1.0`.

---

## Overall results at full training size (`N = 8,000`)

Here is how all five configurations compare across the 2,000 validation comments:

| Approach | Training rows | Element accuracy | Macro ROC-AUC (6 labels) | `toxic` ROC-AUC | `toxic` F1 | `toxic` Precision / Recall | Macro F1 (6 labels) |
|---|---|---|---|---|---|---|---|
| *All-zeros baseline* | *0* | *96.60%* | *0.500* | *0.500* | *0.000* | *0.0% / 0.0%* | *0.000* |
| **PyTorch MLP (Cloud 768d)** | 8,000 | **98.28%** | 0.965 | 0.958 | 0.784 | 83.4% / 74.1% | 0.453 |
| **PyTorch MLP (Local MiniLM 384d)** | 8,000 | 97.78% | 0.902 | 0.919 | 0.622 | 82.9% / 49.7% | 0.324 |
| **$k$-NN $K=7$ (Cloud 768d)** | 8,000 | 98.19% | 0.956 | 0.963 | **0.805** | 84.5% / 76.8% | **0.571** |
| **$k$-NN $K=7$ (Local MiniLM 384d)** | 8,000 | 97.27% | 0.827 | 0.863 | 0.512 | 76.0% / 38.6% | 0.294 |
| **Jev Zero-Shot (`threshold = 0.50`)** | **0** | 92.95% | **0.967** | **0.966** | 0.511 | 34.7% / **96.8%** | 0.369 |
| **Jev Zero-Shot (`threshold = 0.95`)** | **0** | 97.70% | **0.967** | **0.966** | 0.706 | 76.6% / 65.4% | 0.431 |

![ROC curves on the toxic category across all five models](/images/roc-curves.png)
*Figure 1: ROC curves on the primary `toxic` category (2,000 validation comments), with both `toxic` AUC and 6-category Macro AUC in the legend. Zero-shot Jev (`0.966` toxic AUC, `0.967` Macro AUC) matches or beats the models trained on 8,000 examples.*

Look at the **Macro ROC-AUC** column first. Without a single training example, **Jev achieves the highest ranking performance in the entire benchmark (`0.967` Macro ROC-AUC, and `0.981` Micro ROC-AUC)**, edging out both the PyTorch MLP (`0.965`) and $k$-NN (`0.956`) trained on 8,000 cloud embeddings, and beating the local MiniLM models (`0.902` and `0.827`) by a wide margin.

So why does Jev's uncalibrated `threshold = 0.50` row show `92.95%` accuracy and a `0.511` F1 score? That brings us to the first big takeaway.

---

## Takeaway 1: Jev does great, but needs calibration

When you train a PyTorch model with binary cross-entropy (`BCEWithLogitsLoss`) on 8,000 comments from a dataset where `90.75%` of comments are clean, the model learns **two things at once**:

1. **What toxicity looks like** (ranking comments from least to most toxic).
2. **How rare toxicity is in this specific dataset** (the base rate, or class prior).

Because fewer than 1 in 10 comments in Jigsaw are labeled `toxic`, the trained MLP pushes its output probabilities down unless the evidence is overwhelming. By the time training finishes, a `0.50` probability cutoff naturally aligns with the dataset's annotator boundary (`83.4%` precision, `74.1%` recall, `1.6%` false positive rate).

**Jev has never seen the dataset.** It doesn't know whether `1%` or `80%` of Wikipedia talk-page comments are toxic. When you ask Jev a `Noul` question, it evaluates the **semantics of the comment against your criteria**. If a Wikipedia editor writes a snarky, abrasive reply during an edit war—*"Oh, stop crying and read the source before reverting, you have no clue what you're talking about"*—human Jigsaw annotators frequently marked it `0` (not toxic enough to cross their moderation bar), whereas Jev reasonably assigns a `0.60` or `0.80` probability that the comment is rude or disrespectful.

![Threshold calibration sweep comparing zero-shot Jev and the trained PyTorch MLP](/images/threshold-calibration.png)
*Figure 2: Precision, recall, F1 score, and false positive rate on `toxic` as the decision threshold moves from `0.10` to `0.98`. The trained PyTorch MLP (right) has the dataset's `9.3%` base rate baked into its weights, so `F1` peaks near `0.50`. Zero-shot Jev (left) reaches `96.8%` recall at `0.50` with an `18.6%` false positive rate, and peaks at `0.95`.*

Here is what happens to Jev on the `toxic` category (`185` toxic, `1,815` clean comments) as we raise the threshold:

| Jev threshold | True positives (`185`) | False positives (`1,815`) | Precision | Recall | F1 score | False positive rate | Element accuracy (all 6 labels) |
|---|---|---|---|---|---|---|---|
| `0.50` | 179 | 337 | 0.35 | **0.97** | 0.51 | 0.186 | 92.95% |
| `0.70` | 174 | 258 | 0.40 | 0.94 | 0.56 | 0.142 | 94.37% |
| `0.80` | 164 | 158 | 0.51 | 0.89 | 0.65 | 0.087 | 96.08% |
| `0.90` | 140 | 69 | 0.67 | 0.76 | 0.71 | 0.038 | 97.32% |
| **`0.95`** | **121** | **37** | **0.77** | **0.65** | **0.71** | **0.020** | **97.70%** |
| `0.98` | 84 | 5 | **0.94** | 0.45 | 0.61 | **0.003** | **97.84%** |

Moving Jev's threshold from `0.50` to `0.95` eliminates **9 out of 10 false positives** (`337` down to `37`), cuts the false positive rate from `18.6%` to `2.0%`, and raises `toxic` F1 from `0.511` to `0.706`—passing the local MiniLM PyTorch model (`0.622`) and local MiniLM $k$-NN (`0.512`) without using a single training example. If you want high-precision auto-moderation (`threshold = 0.98`), precision hits **94.4%** with a **0.3%** false positive rate (`5` false positives out of `1,815` clean comments).

> [!TIP]
> **Zero-shot doesn't mean zero data.** You don't need 8,000 labeled rows to train Jev, but you *do* want 50 to 100 labeled examples to calibrate its threshold. That small evaluation set is what bridges the gap between "is this semantically rude?" and "where does my application draw the line?"

---

## Takeaway 2: $k$-NN is surprisingly strong—until the training set is small

The second surprise in the full-dataset table is **$k$-NN ($K = 7$) on the 768-d cloud embeddings**. With zero training time and no learned weights—just Euclidean distance and a 7-neighbor vote—it achieves:

- **The highest `toxic` F1 score in the benchmark (`0.805`)**, beating the PyTorch MLP (`0.784`).
- **The highest Macro F1 across all 6 categories (`0.571` vs. PyTorch's `0.453`)**.

Why does a simple nearest-neighbor lookup beat a trained neural network at $N = 8,000$? Look at the per-category breakdown:

![Per-category ROC-AUC and F1 score across all six toxicity labels](/images/per-category-f1.png)
*Figure 3: Per-category ROC-AUC (left) and hard-decision F1 score (right), ordered from the most common label (`toxic`, `n=185`) to the rarest (`threat`, `n=6`). At $N = 8,000$, $k$-NN's local neighborhood voting beats the PyTorch MLP on rare categories.*

In a multi-label dataset with extreme class imbalance (`identity_hate` is `0.65%` of comments; `threat` is `0.30%`), a global MLP trained with standard binary cross-entropy learns to be conservative on rare labels because predicting `0` is right `99.7%` of the time. On `threat`, the MLP achieves a `0.890` ROC-AUC (it ranks threats high relative to clean comments), but none of its probabilities cross `0.50`, resulting in **`0.000` F1**. On `identity_hate`, the MLP reaches `0.353` F1.

$k$-NN doesn't optimize a global loss function. Even if racial slurs or death threats represent only `0.5%` of the dataset, `0.5%` of 8,000 training comments is still **25 to 50 examples** clustered tightly together in embedding space. When a new `identity_hate` comment lands in that neighborhood, 4 or 5 of its 7 nearest neighbors share that label, triggering a positive vote (`0.636` F1 on `identity_hate`, `0.286` F1 on `threat`).

### The small-data cliff

That local voting mechanism has a catch: **what happens when you don't have 8,000 labeled examples yet?**

To find out, I ran a training set size sweep across $N \in \{15, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000\}$, training the PyTorch MLP and indexing $k$-NN ($K = 7$) on the first $N$ training comments and evaluating on the same 2,000 validation comments:

![Training set size sweep from N=15 to N=8,000 comparing PyTorch, k-NN, and zero-shot Jev](/images/data-efficiency.png)
*Figure 4: `toxic` F1 score (left) and 6-category Macro ROC-AUC (right) as the labeled training set grows from 15 to 8,000 comments (log scale), compared against zero-shot Jev (`0` training rows).*

| Training set size ($N$) | Toxic rows in train set | $k$-NN $K=7$ (Cloud 768d) `toxic` F1 | $k$-NN $K=7$ (Local MiniLM) `toxic` F1 | PyTorch MLP (Cloud 768d) `toxic` F1 | PyTorch MLP (Local MiniLM) `toxic` F1 | $k$-NN active labels (`F1 > 0` of 6) |
|---|---|---|---|---|---|---|
| **0 (Jev @ `0.95`)** | **0** | — *(Jev: **0.706**)* | — *(Jev: **0.706**)* | — *(Jev: **0.706**)* | — *(Jev: **0.706**)* | *Jev: 5 / 6 (6 / 6 @ `0.85`)* |
| **15** | 1 | **0.000** | **0.000** | 0.573 | 0.171 | **0 / 6** |
| **25** | 1 | **0.000** | **0.000** | 0.205 | 0.165 | **0 / 6** |
| **50** | 5 | 0.644 | 0.123 | 0.692 | 0.171 | **1 / 6** |
| **100** | 10 | 0.761 | 0.165 | 0.731 | 0.235 | **3 / 6** (MiniLM: **2 / 6**) |
| **250** | 26 | 0.784 | 0.405 | 0.775 | 0.329 | 4 / 6 (MiniLM: 3 / 6) |
| **500** | 51 | 0.786 | 0.400 | 0.766 | 0.388 | 4 / 6 |
| **1,000** | 95 | **0.811** | 0.449 | 0.749 | 0.442 | 4 / 6 |
| **8,000** | 773 | **0.805** | 0.512 | 0.784 | 0.622 | **6 / 6** (MiniLM: 4 / 6) |

Look at what happens at the bottom of the table:

1. **The $\lceil K/2 \rceil$ threshold cliff**: For $k$-NN with $K = 7$ to output a probability `>= 0.50`, at least **4 of the 7 nearest neighbors** must be positive (`4 / 7 = 0.571`). At $N = 15$ and $N = 25$, the training split only contains **1 toxic comment** in total. It is **mathematically impossible** for $k$-NN ($K = 7$) to ever predict `1`—resulting in **0.0% recall and `0.000` F1 across all 6 categories**.
2. **Rare categories stay dead until $N$ is huge**: Even at $N = 50$ (`5` toxic comments in the training set), `5 out of 6` toxicity categories (`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) have fewer than 4 positive examples in the entire training set and score **`0.000` F1**. At $N = 100$ (comparable to my blog's 102 comments), half of the categories (`3 of 6` on cloud embeddings, `4 of 6` on local MiniLM) still have `0.000` F1, and local MiniLM $k$-NN catches only **9.2%** of `toxic` comments (`0.165` F1).
3. **Can you fix small-$N$ $k$-NN by lowering $K$?** I also swept $K \in \{1, 3, 5, 7, 11, 15, 21, 31\}$. Lowering $K$ to `1` or `3` allows $k$-NN to fire when only 1 or 2 positive examples exist, but at $N = 25$ (`1` positive example), $K = 1$ only reaches `0.214` F1 (`0.123` on MiniLM) because a single isolated vector cannot cover the many ways people write toxic comments. Conversely, at $N = 8,000$, $K = 1$ drops `toxic` F1 from `0.805` down to `0.713` because single-neighbor decisions are noisy.

---

## Takeaway 3: The embedding model matters more than the classifier

Looking across both the PyTorch MLP and $k$-NN curves in Figure 4, another pattern jumps out: **the gap between the two embedding models (solid vs. dashed lines) is much larger than the gap between the two classifier architectures (blue vs. orange lines).**

![Comparison of Cloud 768d embeddings vs Local MiniLM 384d embeddings across PyTorch MLP and k-NN](/images/embedding-comparison.png)
*Figure 5: Left: Per-category F1 score at $N = 8,000$ comparing the 768-d cloud `CLASSIFICATION` embeddings (solid bars) against the 384-d local `all-MiniLM-L6-v2` embeddings (hatched bars). Right: `toxic` F1 across dataset sizes—8,000 training examples on MiniLM 384d lose to just 50 training examples on Cloud 768d.*

Here is the head-to-head drop when switching from the 768-d cloud embeddings to the 384-d local `all-MiniLM-L6-v2` embeddings at $N = 8,000$:

| Category (`n` in val) | PyTorch MLP (`768d`) | PyTorch MLP (`384d` MiniLM) | MLP $\Delta$ F1 | $k$-NN $K=7$ (`768d`) | $k$-NN $K=7$ (`384d` MiniLM) | $k$-NN $\Delta$ F1 |
|---|---|---|---|---|---|---|
| `toxic` (`185`) | 0.784 | 0.622 | **-0.162** | **0.805** | 0.512 | **-0.293** |
| `obscene` (`89`) | **0.800** | 0.726 | -0.074 | 0.744 | 0.453 | **-0.291** |
| `insult` (`95`) | 0.684 | 0.596 | -0.088 | **0.745** | 0.471 | **-0.274** |
| `identity_hate` (`13`) | 0.353 | 0.000 | **-0.353** | **0.636** | 0.235 | **-0.401** |
| `severe_toxic` (`20`) | 0.095 | 0.000 | -0.095 | **0.207** | 0.095 | -0.112 |
| `threat` (`6`) | 0.000 | 0.000 | 0.000 | **0.286** | 0.000 | **-0.286** |
| **6-label Macro F1** | 0.453 | 0.324 | **-0.129** | **0.571** | 0.294 | **-0.277** |
| **6-label Macro ROC-AUC** | **0.965** | 0.902 | -0.063 | 0.956 | 0.827 | **-0.129** |

Three things are happening under the hood:

1. **Embedding quality beats $160\times$ more training data.** With the 768-d cloud `CLASSIFICATION` embeddings, training on just **50 labeled comments** gives a `0.692` `toxic` F1 on the PyTorch MLP and `0.644` on $k$-NN (climbing to `0.761` at $N = 100$). With `all-MiniLM-L6-v2`, training on **all 8,000 comments** tops out at `0.622` on the MLP and `0.512` on $k$-NN. No amount of classifier tuning or extra training rows can recover signal that the embedding model threw away.
2. **Task-tuned classification embeddings vs. general topic similarity.** `all-MiniLM-L6-v2` is a compact 22M-parameter model trained to group sentences by **subject matter and paraphrase similarity**. In that space, a polite disagreement about an article on politics and a personal attack about the same article share topic vocabulary and land near each other. Google's larger embedding model, called with `task_type="CLASSIFICATION"`, warps the vector space around **tone, sentiment, and classification boundaries** rather than just topic—so hostile comments cluster together regardless of whether the discussion is about history, sports, or Wikipedia formatting.
3. **Why the winner flips (`k-NN > MLP` on 768d, but `MLP > k-NN` on MiniLM 384d).** Notice that $k$-NN suffers nearly **twice the F1 loss (`-29.3` points on `toxic`)** of the PyTorch MLP (`-16.2` points) when moving to MiniLM. That's because **$k$-NN treats all 384 dimensions equally** when computing Euclidean distance: if 350 dimensions encode *topic* and only a few dozen encode *hostility*, a comment's 7 nearest neighbors in MiniLM space are comments on the *same topic*, not comments with the *same toxicity*. The PyTorch MLP, by contrast, has a learned input layer (`384 -> 128`) that learns to zero out the topic dimensions and amplify the toxicity dimensions—recovering `11` F1 points on `toxic` (`0.622` vs. `0.512`) and `27` F1 points on `obscene` (`0.726` vs. `0.453`).

---

## Performance and running costs

Accuracy and F1 are only half the story. In production, these three approaches have very different latency profiles, memory footprints, and cost structures.

![Single-comment end-to-end latency and classifier memory footprint scaling](/images/performance-and-cost.png)
*Figure 6: Left: Median single-comment end-to-end latency decomposed into the embedding step (blue) and the classifier/decision step (green). Right: Classifier memory footprint in RAM as the labeled dataset grows from 15 to 8,000 examples.*

### 1. Training time and inference latency

In a real web application, comments arrive **one at a time** when a user clicks Submit. For the two embedding-based models, classifying a comment is a two-step pipeline: first generate the vector embedding, then run the classifier (`MLP` or `$k$-NN`).

Here is what we measured on a Mac (Apple Silicon CPU + HTTPS calls for cloud APIs):

| Pipeline | Training / build time (`N = 8,000`) | Embedding latency (1 comment, median) | Classifier latency (1 comment, median) | **Total end-to-end latency (median)** | Classifier RAM / weights size |
|---|---|---|---|---|---|
| **Local MiniLM (384d) + PyTorch MLP** | 7.3 s (150 epochs) + ~15 s embed | 9.18 ms (local CPU) | **0.04 ms** (`0.001 ms` batched) | **9.2 ms** | **196 KB** (`.safetensors`) + ~90 MB MiniLM |
| **Local MiniLM (384d) + $k$-NN ($K=7$)** | 0 s + ~15 s embed | 9.18 ms (local CPU) | 1.71 ms (`0.014 ms` batched) | **10.9 ms** | **11.7 MB** (`8k × 384` float32) + ~90 MB MiniLM |
| **Jev (`jev-1.13.0`, 6 `Noul` questions)** | **0 s** | *None* | **195.0 ms** (`p10: 182 ms, p90: 224 ms`) | **195.0 ms** | **0 KB** |
| **Cloud Embedding (768d) + PyTorch MLP** | 8.6 s (100 epochs) + API embed | 260.8 ms (HTTPS API) | 0.05 ms (`0.002 ms` batched) | **260.9 ms** | **388 KB** (`.safetensors`) |
| **Cloud Embedding (768d) + $k$-NN ($K=7$)** | 0 s + API embed | 260.8 ms (HTTPS API) | 3.72 ms (`0.018 ms` batched) | **264.5 ms** | **23.4 MB** (`8k × 768` float32) |

Three practical engineering details jump out from those numbers:

- **Once you make a network call, the local classifier time is invisible.** Running the PyTorch MLP takes `0.05 ms` (`50 microseconds`), and searching `8,000` vectors in $k$-NN takes `3.7 ms`. Yet if you call a cloud API to generate the 768-d embedding first, the HTTPS round-trip takes `~261 ms`—meaning **calling Jev directly (`195 ms` median for all 6 toxicity questions at once) is actually faster end-to-end than calling a cloud embedding API + a local model**.
- **If you need sub-10ms synchronous latency or browser execution, local MiniLM + MLP is unbeatable.** Running `all-MiniLM-L6-v2` locally with the `196 KB` MLP classifies a comment in **`9.2 ms`** with zero external API calls, and can run entirely inside the user's browser via ONNX Runtime Web and Transformers.js.
- **$k$-NN's memory and compute scale linearly $O(N)$ with your dataset.** The PyTorch MLP compresses 8,000 training comments into `388 KB` of weights (`99,206` parameters), and its latency (`0.05 ms`) never changes whether you trained on 100 rows or 10 million rows. $k$-NN *is* the training set: at $N = 8,000$ (768d), you must keep **23.4 MB** of raw vectors in memory (or **~467 MB** for the full 159,571-row Jigsaw dataset) and compute distances against all of them—or stand up a vector database / HNSW index.

### 2. Running costs vs. upfront cost

How much does each approach cost to run?

| Cost dimension | Local MiniLM + PyTorch / $k$-NN | Cloud Embedding + PyTorch / $k$-NN | Jev (`System One`) |
|---|---|---|---|
| **Upfront data cost** | High (`1,000–8,000+` labeled examples) | High (`1,000–8,000+` labeled examples) | **Near zero** (`0` to start; `~50–100` to calibrate threshold) |
| **Pipeline & infra to maintain** | Training script, ONNX/safetensors export, bundling ~23–90 MB embedding model | Training script, API key, model weights or `23+ MB` vector index in RAM / DB | **1 stateless HTTP call** (`POST /v1/systemone`) |
| **Tokens per comment** | **0 API tokens** | **~75 input tokens** (comment text only, 0 output tokens) | **~946 input + 111 output tokens** (6 toxicity questions)<br>*(or ~280 input + 18 output tokens for 1 spam question)* |
| **Tokens per 1,000 comments** | **0** | **~75,000 input tokens** | **~1.06M tokens** (6 questions) / **~300k tokens** (1 question) |
| **Adapting to new rules** | Relabel dataset + retrain + redeploy | Relabel dataset + retrain (or re-index) | **Edit the `Noul` criteria string in code** |

Across all 2,000 validation comments (12,000 binary judgments across 6 categories), Jev consumed **`1,891,010` input tokens and `222,000` output tokens**—an average of **`945.5` input tokens and `111.0` output tokens per comment** (`18.5` output tokens per `Noul` question). Most of those input tokens (~870 tokens) are the six category definitions and `true`/`false` criteria repeated on every call; for a single-question classifier like my blog's spam filter, a call is under 300 tokens total.

That token profile defines the economic trade-off clearly:

- **At high volume (hundreds of thousands or millions of comments a day)**, calling a zero-shot decision model on every comment adds up to hundreds of millions of tokens, and you almost certainly *already have* thousands of historical examples. At that scale, **Cloud Embeddings + $k$-NN / PyTorch** (which only sends ~75 tokens of raw comment text per call, ~14x fewer tokens than a 6-question Jev prompt) or **Local MiniLM + PyTorch** (`$0` API cost, `9.2 ms` latency) will pay back the engineering investment of training a model within days.
- **At low-to-medium volume (a personal blog, an internal tool, a startup MVP, or an async background moderation job)**, the API bill for a few hundred or a few thousand comments a month is pennies. In those systems, **the dominant cost isn't tokens—it's engineering time and labeled data**. Collecting and manually labeling 8,000 comments costs orders of magnitude more than the API calls ever will.

---

## Which one should you pick?

Putting all three experiments together gives a straightforward decision rule based on how much labeled data you have and how many requests you serve:

1. **You have `0` to `250` labeled examples (Cold Start / Low Volume):**
   **Use [Jev][jev] (`System One`) and calibrate its threshold.** Neither PyTorch nor $k$-NN can learn rare categories from a few dozen examples, and $k$-NN hits a hard `0.000` F1 cliff when positive examples drop below $\lceil K/2 \rceil$. Jev gives you `0.967` Macro ROC-AUC on day one, and once you have 50–100 labeled examples to tune the probability threshold (`0.85`–`0.95`), you get strong precision and recall without maintaining a training pipeline.
2. **You have `1,000+` labeled examples and want the simplest high-accuracy backend classifier:**
   **Use $k$-NN ($K = 7$) on task-tuned cloud embeddings (`task_type="CLASSIFICATION"`).** If your dataset fits comfortably in memory (`~23 MB` for 8,000 rows), $k$-NN requires zero training time, handles rare multi-label categories better than a basic MLP (`0.571` vs. `0.453` Macro F1), and reaches the highest F1 score in our benchmark (`0.805`).
3. **You need `< 10 ms` latency, zero marginal API cost, or full client-side execution in the browser:**
   **Train a small PyTorch MLP on local embeddings (`all-MiniLM-L6-v2`) and export to ONNX.** You trade some accuracy (`0.902` Macro AUC vs. `0.965` with cloud embeddings) for a `196 KB` classifier that runs in `9.2 ms` on CPU or directly inside the user's browser with zero server or token costs.

For my blog's comment queue—where volume is measured in dozens of comments, moderation runs asynchronously in a background task after the HTTP response, and collecting 8,000 spam examples would take decades—that cold-start curve confirms why skipping the custom model was the right call.

[spam]: https://bandarra.me/posts/fighting-comment-spam
[pytorch]: https://bandarra.me/posts/from-pytorch-to-browser-creating-a-web-friendly-ai-model
[onnx]: https://bandarra.me/posts/from-pytorch-to-browser-a-full-client-side-solution-with-onnx-and-transformers-js
[jev]: https://docs.typesafe.ai/concepts/system-one
[noul]: https://docs.typesafe.ai/primitives/noul
[jigsaw]: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
[repo]: https://github.com/andreban/jigsaw-toxic-comment-classification-challenge
[minilm]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
