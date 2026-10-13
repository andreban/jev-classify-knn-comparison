"""Hero image: the 2,000 evaluation comments laid out by embedding similarity.

t-SNE projection of the cloud (gemini-embedding-001) embeddings. Toxic comments
in orange, clean in blue. Same palette as the charts and the jev-comment-spam
data-art hero. Needs ./scripts/restore_data.sh.

    PYTHONPATH=src .venv/bin/python hero.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from toxicity_dataset import get_or_create_split_indices, load_full_records

BG, BLUE, ORANGE = "#f6f1e7", "#2a78d6", "#eb6834"

recs = load_full_records("YT-comments-text-embedding-004-768-10k.jsonl")
_, va = get_or_create_split_indices(num_samples=len(recs))
x = np.array([recs[i]["embeddings"] for i in va], dtype=np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True)
toxic = np.array([recs[i]["labels"][0] for i in va]) == 1
xy = TSNE(n_components=2, perplexity=40, metric="cosine", init="pca", random_state=3).fit_transform(x)
xy -= xy.mean(0)
xy /= np.abs(xy).max(0)

fig = plt.figure(figsize=(16, 9), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG)
ax.axis("off")
ax.scatter(xy[~toxic, 0] * 1.55, xy[~toxic, 1], s=26, color=BLUE, alpha=0.55, linewidths=0)
ax.scatter(xy[toxic, 0] * 1.55, xy[toxic, 1], s=46, color=ORANGE, alpha=0.95, linewidths=0)
ax.set_xlim(-1.85, 1.85)
ax.set_ylim(-1.12, 1.12)
fig.savefig("results/plots/hero.png", dpi=100, facecolor=BG)
print("wrote results/plots/hero.png")
