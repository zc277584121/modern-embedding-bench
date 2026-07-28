"""Real data loaders for evaluation tasks.

Loads pre-prepared datasets from the data/ directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mm_embed.benchmark.materialization import MaterializationAuthorization, MaterializationContractError

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data"


@dataclass
class RealTextImagePair:
    """A real text-image pair for cross-modal retrieval."""

    text: str
    image_bytes: bytes
    category: str
    hard_negatives: list[str]


def load_mrl_real_data(
    materialization: MaterializationAuthorization,
) -> list[tuple[str, str, bool]]:
    """Load real MRL stress test data from STS-B (binary labels).

    Returns the same format as mock: (text_a, text_b, is_similar).
    Pairs with score >= 4.0 are labeled similar, score <= 1.0 dissimilar.
    Pairs in between are excluded to create a clear binary split.
    """
    data = []
    for line in materialization.read_payload_text("mrl_stsb_pairs").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = row["score"]
        if score >= 4.0:
            data.append((row["text_a"], row["text_b"], True))
        elif score <= 1.0:
            data.append((row["text_a"], row["text_b"], False))

    return data


def load_mrl_continuous_data(
    materialization: MaterializationAuthorization,
) -> list[tuple[str, str, float]]:
    """Load ALL STS-B pairs with continuous similarity scores.

    Returns: (text_a, text_b, score) where score is 0.0-5.0.
    This includes ALL 1379 pairs for Spearman correlation evaluation.
    """
    data = []
    for line in materialization.read_payload_text("mrl_stsb_pairs").splitlines():
        if line.strip():
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


def load_cross_modal_real_data(
    materialization: MaterializationAuthorization,
) -> list[RealTextImagePair]:
    """Load real cross-modal text-image pairs from COCO + GPT-4o-mini captions.

    Returns list of RealTextImagePair with .text, .image_bytes, .category, .hard_negatives.
    Compatible with MockTextImagePair interface (.text, .image_bytes, .category).
    """
    data = []
    referenced_assets: list[str] = []
    for line in materialization.read_payload_text("cross_modal_metadata").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        asset_path = materialization.resolve_payload_reference("cross_modal_metadata", row.get("image_path"))
        referenced_assets.append(asset_path)
        data.append(RealTextImagePair(
            text=row["caption"],
            image_bytes=materialization.read_asset(asset_path),
            category=row.get("category", "other"),
            hard_negatives=row.get("hard_negatives", []),
        ))

    if tuple(referenced_assets) != materialization.asset_paths:
        raise MaterializationContractError(
            "asset_reference_mismatch",
            "cross-modal loader references do not exactly match manifest assets",
        )

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


def load_crosslingual_data(
    materialization: MaterializationAuthorization,
) -> list[CrossLingualPair]:
    """Load Chinese-English parallel sentence pairs.

    Returns list of CrossLingualPair with .zh, .en, .difficulty, .category,
    .hard_negatives_en, .hard_negatives_zh.
    """
    data = []
    for line in materialization.read_payload_text("crosslingual_pairs").splitlines():
        if not line.strip():
            continue
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
    materialization: MaterializationAuthorization,
    haystack_lengths: list[int] | None = None,
    needle_positions: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Load real needle-in-haystack test data from Wikipedia articles.

    Returns the same format as mock get_needle_haystack_data():
    list of {"document", "query", "needle", "position", "length"}.
    """
    if haystack_lengths is None:
        haystack_lengths = [1000, 4000, 8000, 16000, 32000]
    if needle_positions is None:
        needle_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    haystacks = {}
    for line in materialization.read_payload_text("needle_haystacks").splitlines():
        if not line.strip():
            continue
        hs = json.loads(line)
        if hs["length"] in haystacks:
            raise ValueError(f"Duplicate needle haystack length in materialization: {hs['length']}")
        haystacks[hs["length"]] = hs["text"]

    needles = []
    for line in materialization.read_payload_text("needle_facts").splitlines():
        if line.strip():
            needles.append(json.loads(line))

    missing_lengths = sorted(set(haystack_lengths) - set(haystacks))
    if missing_lengths:
        raise ValueError(f"Needle materialization is missing requested haystack lengths: {missing_lengths}")
    if not needles:
        raise ValueError("Needle materialization contains no needle rows")

    test_cases = []
    for length in haystack_lengths:
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
