"""Real data loaders for evaluation tasks.

Loads pre-prepared datasets from the data/ directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data"


@dataclass
class RealTextImagePair:
    """A real text-image pair for cross-modal retrieval."""

    text: str
    image_bytes: bytes
    category: str
    hard_negatives: list[str]


def load_mrl_real_data() -> list[tuple[str, str, bool]]:
    """Load real MRL stress test data from STS-B (binary labels).

    Returns the same format as mock: (text_a, text_b, is_similar).
    Pairs with score >= 4.0 are labeled similar, score <= 1.0 dissimilar.
    Pairs in between are excluded to create a clear binary split.
    """
    stsb_path = DATA_ROOT / "mrl_stress" / "stsb_test.jsonl"
    if not stsb_path.exists():
        raise FileNotFoundError(
            f"MRL data not found at {stsb_path}. "
            "Run: uv run --extra data python scripts/prepare_mrl_data.py"
        )

    data = []
    with open(stsb_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            score = row["score"]
            if score >= 4.0:
                data.append((row["text_a"], row["text_b"], True))
            elif score <= 1.0:
                data.append((row["text_a"], row["text_b"], False))

    return data


def load_mrl_continuous_data() -> list[tuple[str, str, float]]:
    """Load ALL STS-B pairs with continuous similarity scores.

    Returns: (text_a, text_b, score) where score is 0.0-5.0.
    This includes ALL 1379 pairs for Spearman correlation evaluation.
    """
    stsb_path = DATA_ROOT / "mrl_stress" / "stsb_test.jsonl"
    if not stsb_path.exists():
        raise FileNotFoundError(
            f"MRL data not found at {stsb_path}. "
            "Run: uv run --extra data python scripts/prepare_mrl_data.py"
        )

    data = []
    with open(stsb_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            data.append((row["text_a"], row["text_b"], row["score"]))

    return data


def load_mrl_retrieval_data() -> tuple[list[dict], list[dict]]:
    """Load retrieval-style MRL data: (corpus, query_pairs).

    Returns:
        corpus: list of {"id": int, "text": str}
        pairs: list of {"query": str, "positive": str, "corpus_id": int, "score": float}
    """
    corpus_path = DATA_ROOT / "mrl_stress" / "corpus.jsonl"
    pairs_path = DATA_ROOT / "mrl_stress" / "retrieval_pairs.jsonl"

    for path in (corpus_path, pairs_path):
        if not path.exists():
            raise FileNotFoundError(
                f"MRL data not found at {path}. "
                "Run: uv run --extra data python scripts/prepare_mrl_data.py"
            )

    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))

    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    return corpus, pairs


# =============================================================================
# Cross-Modal Retrieval
# =============================================================================


def load_cross_modal_real_data() -> list[RealTextImagePair]:
    """Load real cross-modal text-image pairs from COCO + GPT-4o-mini captions.

    Returns list of RealTextImagePair with .text, .image_bytes, .category, .hard_negatives.
    Compatible with MockTextImagePair interface (.text, .image_bytes, .category).
    """
    meta_path = DATA_ROOT / "cross_modal" / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Cross-modal data not found at {meta_path}. "
            "Run: uv run --extra data --extra openai python scripts/prepare_cross_modal_data.py --count 200"
        )

    data = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            image_path = DATA_ROOT / "cross_modal" / row["image_path"]
            if not image_path.exists():
                continue
            data.append(RealTextImagePair(
                text=row["caption"],
                image_bytes=image_path.read_bytes(),
                category=row.get("category", "other"),
                hard_negatives=row.get("hard_negatives", []),
            ))

    return data


# =============================================================================
# Cross-Lingual Retrieval
# =============================================================================


@dataclass
class CrossLingualPair:
    """A Chinese-English parallel sentence pair."""

    zh: str
    en: str
    difficulty: str
    category: str
    hard_negatives_en: list[str]
    hard_negatives_zh: list[str]


def load_crosslingual_data() -> list[CrossLingualPair]:
    """Load Chinese-English parallel sentence pairs.

    Returns list of CrossLingualPair with .zh, .en, .difficulty, .category,
    .hard_negatives_en, .hard_negatives_zh.
    """
    data_path = DATA_ROOT / "crosslingual" / "parallel_pairs.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cross-lingual data not found at {data_path}. "
            "Run: uv run python scripts/prepare_crosslingual_data.py"
        )

    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            data.append(CrossLingualPair(
                zh=row["zh"],
                en=row["en"],
                difficulty=row.get("difficulty", "medium"),
                category=row.get("category", "general"),
                hard_negatives_en=row.get("hard_negatives_en", []),
                hard_negatives_zh=row.get("hard_negatives_zh", []),
            ))

    return data


# =============================================================================
# Needle-in-Haystack
# =============================================================================


def load_needle_haystack_real_data(
    haystack_lengths: list[int] | None = None,
    needle_positions: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Load real needle-in-haystack test data from Wikipedia articles.

    Returns the same format as mock get_needle_haystack_data():
    list of {"document", "query", "needle", "position", "length"}.
    """
    haystacks_path = DATA_ROOT / "needle_haystack" / "haystacks.jsonl"
    needles_path = DATA_ROOT / "needle_haystack" / "needles.jsonl"

    for path in (haystacks_path, needles_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Needle data not found at {path}. "
                "Run: uv run python scripts/prepare_needle_data.py"
            )

    if haystack_lengths is None:
        haystack_lengths = [1000, 4000, 8000, 16000, 32000]
    if needle_positions is None:
        needle_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    haystacks = {}
    with open(haystacks_path, encoding="utf-8") as f:
        for line in f:
            hs = json.loads(line)
            haystacks[hs["length"]] = hs["text"]

    needles = []
    with open(needles_path, encoding="utf-8") as f:
        for line in f:
            needles.append(json.loads(line))

    test_cases = []
    for length in haystack_lengths:
        if length not in haystacks:
            continue
        haystack = haystacks[length]

        for pos in needle_positions:
            for needle_info in needles:
                needle_text = needle_info["needle"]
                # Insert needle at position
                if pos <= 0.0:
                    document = needle_text + " " + haystack
                elif pos >= 1.0:
                    document = haystack + " " + needle_text
                else:
                    idx = int(len(haystack) * pos)
                    space_idx = haystack.rfind(" ", 0, idx)
                    if space_idx > idx * 0.8:
                        idx = space_idx + 1
                    document = haystack[:idx] + " " + needle_text + " " + haystack[idx:]

                test_cases.append({
                    "document": document,
                    "query": needle_info["query"],
                    "needle": needle_text,
                    "position": pos,
                    "length": length,
                })

    return test_cases
