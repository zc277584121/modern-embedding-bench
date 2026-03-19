"""Prepare real MRL stress test data from STS-B dataset.

Downloads the STS Benchmark test split from HuggingFace and converts it into
retrieval-style evaluation data:
  - corpus.jsonl: all unique sentences
  - retrieval_pairs.jsonl: query-positive pairs (score >= 4.0)
  - stsb_test.jsonl: raw pairs with similarity scores

Usage:
    uv run --extra data python scripts/prepare_mrl_data.py [--preview]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mrl_stress"
SCORE_THRESHOLD = 4.0  # Pairs with score >= this are "positive" matches


def download_stsb_test() -> list[dict]:
    """Download and return STS-B test split as list of dicts."""
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    rows = []
    for row in ds:
        rows.append({
            "text_a": row["sentence1"],
            "text_b": row["sentence2"],
            "score": round(row["score"], 2),
        })
    return rows


def build_corpus(rows: list[dict]) -> list[dict]:
    """Extract all unique sentences and assign IDs."""
    seen = {}
    corpus = []
    for row in rows:
        for text in (row["text_a"], row["text_b"]):
            if text not in seen:
                seen[text] = len(corpus)
                corpus.append({"id": seen[text], "text": text})
    return corpus


def build_retrieval_pairs(rows: list[dict], corpus_lookup: dict[str, int]) -> list[dict]:
    """Build query-positive retrieval pairs from high-similarity rows."""
    pairs = []
    for row in rows:
        if row["score"] >= SCORE_THRESHOLD:
            pairs.append({
                "query": row["text_a"],
                "positive": row["text_b"],
                "corpus_id": corpus_lookup[row["text_b"]],
                "score": row["score"],
            })
    return pairs


def save_jsonl(path: Path, data: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_preview(rows: list[dict], pairs: list[dict], n: int = 10) -> None:
    """Print a preview of the data for quality checking."""
    print("=" * 70)
    print(f"  STS-B Raw Pairs (first {n})")
    print("=" * 70)
    for row in rows[:n]:
        print(f"  score={row['score']:.1f}  A: {row['text_a'][:60]}")
        print(f"            B: {row['text_b'][:60]}")
        print()

    print("=" * 70)
    print(f"  Retrieval Pairs (first {n}, score >= {SCORE_THRESHOLD})")
    print("=" * 70)
    for pair in pairs[:n]:
        print(f"  score={pair['score']:.1f}  Q: {pair['query'][:60]}")
        print(f"            +: {pair['positive'][:60]}")
        print()


def print_stats(rows: list[dict], corpus: list[dict], pairs: list[dict]) -> None:
    """Print dataset statistics."""
    scores = [r["score"] for r in rows]
    buckets = Counter(int(s) for s in scores)

    print("=" * 70)
    print("  Dataset Statistics")
    print("=" * 70)
    print(f"  Total STS-B test pairs:    {len(rows)}")
    print(f"  Unique sentences (corpus): {len(corpus)}")
    print(f"  Retrieval pairs (>= {SCORE_THRESHOLD}):  {len(pairs)}")
    print(f"  Score range:               {min(scores):.2f} - {max(scores):.2f}")
    print(f"  Mean score:                {sum(scores)/len(scores):.2f}")
    print()
    print("  Score distribution:")
    for bucket in sorted(buckets.keys()):
        bar = "#" * (buckets[bucket] // 5)
        print(f"    [{bucket}-{bucket+1}): {buckets[bucket]:4d}  {bar}")
    print()


def main() -> None:
    preview_only = "--preview" in sys.argv

    print("Downloading STS-B test split from HuggingFace...")
    rows = download_stsb_test()

    corpus = build_corpus(rows)
    corpus_lookup = {item["text"]: item["id"] for item in corpus}
    pairs = build_retrieval_pairs(rows, corpus_lookup)

    # Always print preview and stats
    print_preview(rows, pairs)
    print_stats(rows, corpus, pairs)

    if preview_only:
        print("[Preview mode] No files written. Remove --preview to save data.")
        return

    # Save files
    save_jsonl(DATA_DIR / "stsb_test.jsonl", rows)
    save_jsonl(DATA_DIR / "corpus.jsonl", corpus)
    save_jsonl(DATA_DIR / "retrieval_pairs.jsonl", pairs)

    print(f"Data saved to {DATA_DIR}/")
    print(f"  stsb_test.jsonl:       {len(rows)} pairs")
    print(f"  corpus.jsonl:          {len(corpus)} sentences")
    print(f"  retrieval_pairs.jsonl: {len(pairs)} pairs")


if __name__ == "__main__":
    main()
