#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-andreban/jev-classify-knn-comparison}"
RELEASE_TAG="${2:-v1.0-data}"

echo "Downloading precomputed embeddings and dataset from ${REPO} (${RELEASE_TAG})..."
mkdir -p data /tmp/jev-knn-data-restore

gh release download "$RELEASE_TAG" --repo "$REPO" --dir /tmp/jev-knn-data-restore --clobber

echo "Decompressing data/train.csv..."
gunzip -c /tmp/jev-knn-data-restore/train.csv.gz > data/train.csv

echo "Decompressing SP-all-MiniLM-L6-v2.jsonl (384d local embeddings)..."
gunzip -c /tmp/jev-knn-data-restore/SP-all-MiniLM-L6-v2.jsonl.gz > SP-all-MiniLM-L6-v2.jsonl

echo "Decompressing YT-comments-text-embedding-004-768-10k.jsonl (768d cloud embeddings)..."
gunzip -c /tmp/jev-knn-data-restore/YT-comments-text-embedding-004-768-10k.jsonl.gz > YT-comments-text-embedding-004-768-10k.jsonl

rm -rf /tmp/jev-knn-data-restore
echo "Done! All embeddings and dataset files are restored."
