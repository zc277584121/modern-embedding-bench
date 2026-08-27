"""Research-only BRIGHT economics and psychology pilot v0.2."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import re
import resource
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from jsonschema import Draft202012Validator

from mm_embed.benchmark.bright_v01 import EXPECTED, HF_DATASET, HF_REVISION, TrackRows, load_track, source_url
from mm_embed.benchmark.retrieval_v01 import (
    METRIC_NAMES,
    TrackData,
    aggregate_metrics,
    bm25_rank,
    bootstrap_confidence_intervals,
    dense_rank,
    json_dumps,
    paired_bootstrap_deltas,
    query_metrics,
    snapshot_identity,
    tokens,
    validate_track_data,
    write_jsonl,
)


BENCHMARK_VERSION = "bright-nontechnical-pilot-v0.2"
TRACKS = ("economics", "psychology")
BASELINE_METHODS = ("bm25", "dense", "long_dense")
FORMAL_METHODS = ("bm25", "long_dense")
TARGET_DOCUMENTS = 7_500
TARGET_QUERIES = {"economics": 103, "psychology": 101}
TARGET_QRELS = {"economics": 800, "psychology": 692}
TARGET_UNIQUE_GOLD_DOCUMENTS = {"economics": 800, "psychology": 688}
SELECTION_SALT = BENCHMARK_VERSION
REVIEW_SALT = f"{BENCHMARK_VERSION}-review-plan-120"
REVIEW_PER_TRACK = 60
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
LONG_DENSE_MODEL_ID = "BAAI/bge-m3"
LONG_DENSE_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
LONG_DENSE_SNAPSHOT_PATH = Path(
    "/data1/cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)
LONG_DENSE_DECLARED_MAX_SEQUENCE_LENGTH = 8_192
LONG_DENSE_CAPS = (512, 1_024, 2_048, 4_096, 8_192)
LONG_DENSE_GOLD_TRUNCATION_THRESHOLD = 0.10
LONG_DENSE_SELECTED_CAP = 1_024
LONG_DENSE_BATCH_SIZE = 4
LONG_DENSE_SNAPSHOT = {
    "repo_id": LONG_DENSE_MODEL_ID,
    "revision": LONG_DENSE_MODEL_REVISION,
    "files": 13,
    "bytes": 2_295_435_813,
    "aggregate_sha256": "9bfc0c6488e958ec7be6223d95c331c9692c4ae47212a9730cb38ccc190e6cec",
}
LONG_DENSE_CONTRACT_SHA256 = "fcd1b3a8c4004f56cc8c65e0438edcdd3cacc3149e33845b5365ecd1ed263d8f"
CANONICAL_MANIFEST_SHA256 = "8174b0c01a32e977cf0c5d89522356c485196b0aa0571251cfe1595c5579cb1f"
CANONICAL_RESULT_ARTIFACTS = {
    ("bm25", "economics"): {
        "rankings": "3d11bb9b82ebba295754bc213b3a9e7a8db73ff11c2bb3a813189c1b5092b590",
        "per_query_metrics": "5e7e2e71a829c424c8ff38f0a954d11a6f72cff56238c6295f0b2dbc07274640",
    },
    ("bm25", "psychology"): {
        "rankings": "cc1084fa930ca8ad23879d8004705ca934c64468dd0886efb4425bde3e3ef49e",
        "per_query_metrics": "726e746d04c9d6393d3c60ac04dffecb54161dac4854b5218bb6a039cf428a77",
    },
    ("dense", "economics"): {
        "rankings": "29b5d1b96f127ca04015ce59d52dfab940b66201e36cb899db8ae8908598fa31",
        "per_query_metrics": "bbb20c60bbaa11ec51492669eaac829668b3ed599b146dabab7a29cbfc31d98d",
    },
    ("dense", "psychology"): {
        "rankings": "604c743f644c26ea035f8f2843f623d381374c4d26933afd4e44fc3d24c25743",
        "per_query_metrics": "54a554e28ef232461a1165c0cbe1abc681d7b52322be9101e7e4895745634a01",
    },
    ("long_dense", "economics"): {
        "rankings": "4695f45ffeac45613a31cd6632fccdf33cf06a29f074d61fd90faa6af4a436b4",
        "per_query_metrics": "8ab996827e8f056d4999fcc1859798950c28b1737d6217efc07bc3fb14385d1e",
    },
    ("long_dense", "psychology"): {
        "rankings": "902a3b37784d4600032a694d679c1e513832c07cd566913b026124c8df4d09b3",
        "per_query_metrics": "e1eb68b1a0ab6c7323dcd4c8d5ffd3ebaacfbc5b1936d9f6010db043bb587db7",
    },
}
LONG_DENSE_EXPECTED_TRUNCATION = {
    "economics": {
        "queries": {512: 2, 1_024: 0, 2_048: 0, 4_096: 0, 8_192: 0},
        "candidates": {512: 711, 1_024: 148, 2_048: 23, 4_096: 11, 8_192: 1},
        "gold_documents": {512: 304, 1_024: 37, 2_048: 3, 4_096: 1, 8_192: 0},
    },
    "psychology": {
        "queries": {512: 0, 1_024: 0, 2_048: 0, 4_096: 0, 8_192: 0},
        "candidates": {512: 425, 1_024: 160, 2_048: 56, 4_096: 27, 8_192: 10},
        "gold_documents": {512: 188, 1_024: 63, 2_048: 10, 4_096: 4, 8_192: 1},
    },
}
SELECTION_ID_SHA256 = {
    "economics": "13e2bf56644a8d5b8d8701be74ead7b0f147057bbda34e6a8f246b515f4b1398",
    "psychology": "d56ba2a26cc69163b412c35ebea5c1fa5141ab03fe54215a19abd053d516d3e4",
}
PUBLICATION_GATE = "closed_pending_upstream_web_content_rights_review"
MATERIALIZED_FILES = (
    "corpus.jsonl",
    "queries.jsonl",
    "qrels.jsonl",
    "selection.jsonl",
    "audit.json",
    "sensitive-evidence.jsonl",
)
CANONICAL_TRACK_FILES: dict[str, dict[str, dict[str, Any]]] = {
    "economics": {
        "corpus.jsonl": {
            "path": "corpus.jsonl",
            "rows": 7_500,
            "bytes": 4_959_135,
            "sha256": "697bf4deb34bcac5423a7dfcaf60ce27ea0c961190c0931d189a1620a703e188",
        },
        "queries.jsonl": {
            "path": "queries.jsonl",
            "rows": 103,
            "bytes": 89_316,
            "sha256": "34f8dfb6ddb616be14582c20f291fa286f8548fe797af09e37bf444b8fb34665",
        },
        "qrels.jsonl": {
            "path": "qrels.jsonl",
            "rows": 800,
            "bytes": 70_332,
            "sha256": "e09a3c96a3b112d4233e26baacf5d0f021b6025378fded32478f94e4a2bf3fa3",
        },
        "selection.jsonl": {
            "path": "selection.jsonl",
            "rows": 7_500,
            "bytes": 1_392_327,
            "sha256": "3d2a46945d763a09a6ff8f45b917524bf0e46f7aeaef81df161fadddb54d796f",
        },
        "audit.json": {
            "path": "audit.json",
            "bytes": 3_350,
            "sha256": "1c36c9fbb596e7f8a67223bdf4b246ca322047152075d566411343f39c9c53b1",
        },
        "sensitive-evidence.jsonl": {
            "path": "sensitive-evidence.jsonl",
            "rows": 164,
            "bytes": 73_911,
            "sha256": "e4b6ece87d5863fac36f44ee1578ced439bdf3ac37229587089d45b3dd74117b",
        },
    },
    "psychology": {
        "corpus.jsonl": {
            "path": "corpus.jsonl",
            "rows": 7_500,
            "bytes": 5_022_936,
            "sha256": "a7513e2a67d2fb9b80504d12253f041863fba895637c40d6ae7efbcba1700089",
        },
        "queries.jsonl": {
            "path": "queries.jsonl",
            "rows": 101,
            "bytes": 81_114,
            "sha256": "ead6a7004ea306ca90e8db58f5c6809d05cb448e34c276834aebf7bbe0c90d9d",
        },
        "qrels.jsonl": {
            "path": "qrels.jsonl",
            "rows": 692,
            "bytes": 55_347,
            "sha256": "52360d1c95984d8db2b04e018f68e99737063836e889bf3e7f6865b76d26bc44",
        },
        "selection.jsonl": {
            "path": "selection.jsonl",
            "rows": 7_500,
            "bytes": 1_364_260,
            "sha256": "fa6a856f3cce3b97a58c5578bfdec70f7fbff327025447eacc450adc9cc0b9e9",
        },
        "audit.json": {
            "path": "audit.json",
            "bytes": 3_257,
            "sha256": "7be9cb57e1706553bfa0462c898d98a0722b08368b8c0fc4ad4629fe62e325b1",
        },
        "sensitive-evidence.jsonl": {
            "path": "sensitive-evidence.jsonl",
            "rows": 87,
            "bytes": 39_707,
            "sha256": "76131a3327e562b595ccfb6c97e0f1cb81493d8bd2423f126c1793bddd3584a1",
        },
    },
}
CANONICAL_REVIEW_PLAN: dict[str, Any] = {
    "path": "review-plan-120.jsonl",
    "rows": 120,
    "bytes": 182_458,
    "sha256": "93848a826c12acf9cb6fed50d71dd9490be7baebfcea1122dd4fffb4881fdc4e",
    "allocation": {"economics": 60, "psychology": 60},
    "fixed_before_model_runs": True,
    "model_score_used": False,
    "selection_salt": REVIEW_SALT,
}

EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b")
IPV4_PATTERN = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
US_SSN_PATTERN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CONTACT_LOCAL_PARTS = frozenset({"admin", "contact", "editor", "help", "info", "office", "press", "sales", "support", "webmaster"})


class BrightPilotError(ValueError):
    """A stable fail-closed pilot contract error."""


@dataclass(frozen=True)
class PilotTrack:
    """Selected data and restricted provenance for one independent track."""

    data: TrackData
    source_examples: tuple[dict[str, Any], ...]
    selection: tuple[dict[str, str], ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json_dumps(value) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _schema_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "schemas" / name


def _manifest_schema() -> dict[str, Any]:
    return json.loads(_schema_path("bright-multidomain-materialization-v02.schema.json").read_text(encoding="utf-8"))


def _result_schema() -> dict[str, Any]:
    return json.loads(_schema_path("bright-multidomain-result-v02.schema.json").read_text(encoding="utf-8"))


def _tokenizer_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    lengths = []
    for start in range(0, len(texts), 64):
        encoded = tokenizer(
            texts[start : start + 64],
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.extend(len(values) for values in encoded["input_ids"])
    return lengths


def _long_dense_preflight_for_tracks(tracks: dict[str, PilotTrack]) -> dict[str, Any]:
    """Select the long-context cap from tokenizer lengths without model scores."""
    from transformers import AutoTokenizer

    if not LONG_DENSE_SNAPSHOT_PATH.is_dir():
        raise BrightPilotError("fixed BGE-M3 snapshot is unavailable")
    identity = snapshot_identity(LONG_DENSE_MODEL_ID, LONG_DENSE_MODEL_REVISION)
    identity_summary = {key: identity[key] for key in LONG_DENSE_SNAPSHOT}
    if identity_summary != LONG_DENSE_SNAPSHOT:
        raise BrightPilotError("fixed BGE-M3 snapshot identity mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        LONG_DENSE_SNAPSHOT_PATH,
        trust_remote_code=False,
        local_files_only=True,
    )
    declared_maximum = int(tokenizer.model_max_length)
    if declared_maximum != LONG_DENSE_DECLARED_MAX_SEQUENCE_LENGTH:
        raise BrightPilotError("BGE-M3 tokenizer maximum sequence length mismatch")
    tokenizer.model_max_length = 10**30
    track_results = {}
    raw_counts = {}
    for track_name in TRACKS:
        data = tracks[track_name].data
        document_lengths = _tokenizer_lengths(tokenizer, [row["content"] for row in data.corpus])
        query_lengths = _tokenizer_lengths(tokenizer, [row["text"] for row in data.queries])
        lengths_by_id = {row["id"]: length for row, length in zip(data.corpus, document_lengths)}
        gold_ids = sorted({document_id for rels in data.qrels.values() for document_id in rels})
        values_by_role = {
            "queries": query_lengths,
            "candidates": document_lengths,
            "gold_documents": [lengths_by_id[document_id] for document_id in gold_ids],
        }
        raw_counts[track_name] = {
            role: {cap: sum(length > cap for length in values) for cap in LONG_DENSE_CAPS}
            for role, values in values_by_role.items()
        }
        track_results[track_name] = {
            role: {
                "total": len(values),
                "by_cap": {
                    str(cap): {
                        "truncated": raw_counts[track_name][role][cap],
                        "rate": raw_counts[track_name][role][cap] / len(values),
                    }
                    for cap in LONG_DENSE_CAPS
                },
            }
            for role, values in values_by_role.items()
        }
    if raw_counts != LONG_DENSE_EXPECTED_TRUNCATION:
        raise BrightPilotError("BGE-M3 tokenizer truncation evidence mismatch")
    eligible = [
        cap
        for cap in LONG_DENSE_CAPS
        if all(
            raw_counts[track]["gold_documents"][cap] / track_results[track]["gold_documents"]["total"]
            <= LONG_DENSE_GOLD_TRUNCATION_THRESHOLD
            for track in TRACKS
        )
    ]
    if not eligible:
        raise BrightPilotError("BGE-M3 gold truncation exceeds the threshold even at 8192 tokens")
    selected_cap = min(eligible)
    if selected_cap != LONG_DENSE_SELECTED_CAP:
        raise BrightPilotError("BGE-M3 selected cap identity mismatch")
    return {
        "method": "long_dense",
        "model": {
            **LONG_DENSE_SNAPSHOT,
            "trust_remote_code": False,
            "representation": "SentenceTransformer dense representation",
            "declared_max_sequence_length": LONG_DENSE_DECLARED_MAX_SEQUENCE_LENGTH,
        },
        "cap_selection": {
            "candidate_caps": list(LONG_DENSE_CAPS),
            "criterion": "minimum cap with unique-gold-document truncation rate <= 0.10 in every track",
            "gold_truncation_threshold": LONG_DENSE_GOLD_TRUNCATION_THRESHOLD,
            "selected_cap": selected_cap,
            "score_used": False,
            "fixed_before_model_scoring": True,
            "token_length_semantics": "tokenizer input length including special tokens; truncated iff length exceeds cap",
            "tracks": track_results,
        },
    }


def _content_sha256(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _selection_rank(track: str, content_sha256: str) -> str:
    return _sha256_bytes(f"{SELECTION_SALT}\0{track}\0{content_sha256}".encode())


def _ids_sha256(values: Iterable[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(values)) + "\n").encode())


def select_track(source: TrackRows) -> PilotTrack:
    """Create the fixed score-independent candidate slice for one track."""
    if source.track not in TRACKS:
        raise BrightPilotError(f"unsupported pilot track: {source.track}")
    documents = {row["id"]: row["content"] for row in source.documents}
    gold_ids = {doc_id for row in source.examples for doc_id in row["gold_ids"]}
    if len(gold_ids) != TARGET_UNIQUE_GOLD_DOCUMENTS[source.track]:
        raise BrightPilotError(f"unexpected unique gold count for {source.track}")
    selected_ids = set(gold_ids)
    represented_content = {_content_sha256(documents[doc_id]) for doc_id in selected_ids}
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for doc_id, content in documents.items():
        if doc_id in gold_ids:
            continue
        content_hash = _content_sha256(content)
        if content_hash not in represented_content:
            groups[content_hash].append(doc_id)
    ordered_groups = sorted(groups, key=lambda value: (_selection_rank(source.track, value), value))
    for content_hash in ordered_groups:
        selected_ids.add(min(groups[content_hash]))
        represented_content.add(content_hash)
        if len(selected_ids) == TARGET_DOCUMENTS:
            break
    if len(selected_ids) != TARGET_DOCUMENTS:
        raise BrightPilotError(f"not enough unique-content candidates for {source.track}")
    selection_hash = _ids_sha256(selected_ids)
    if selection_hash != SELECTION_ID_SHA256[source.track]:
        raise BrightPilotError(f"candidate selection identity mismatch for {source.track}")
    corpus = tuple({"id": doc_id, "content": documents[doc_id]} for doc_id in sorted(selected_ids))
    queries = tuple(
        {
            "id": row["id"],
            "text": row["query"],
            "excluded_ids": sorted(set(row["excluded_ids"]) & selected_ids),
            "gold_ids_long": row["gold_ids_long"],
        }
        for row in source.examples
    )
    qrels = {
        row["id"]: {doc_id: 1 for doc_id in sorted(row["gold_ids"])}
        for row in source.examples
    }
    if len(queries) != TARGET_QUERIES[source.track] or sum(map(len, qrels.values())) != TARGET_QRELS[source.track]:
        raise BrightPilotError(f"fixed query or qrel count mismatch for {source.track}")
    data = TrackData(source.track, corpus, queries, qrels)
    try:
        validate_track_data(data)
    except ValueError as error:
        raise BrightPilotError(str(error)) from error
    selection = tuple(
        {
            "document_id": doc_id,
            "content_sha256": _content_sha256(documents[doc_id]),
            "reason": "positive_qrel" if doc_id in gold_ids else "deduplicated_hash_filler",
        }
        for doc_id in sorted(selected_ids)
    )
    return PilotTrack(data, source.examples, selection)


def _percentiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(array, [0, 25, 50, 75, 90, 95, 99, 100])
    result = {
        key: float(value)
        for key, value in zip(("min", "p25", "p50", "p75", "p90", "p95", "p99", "max"), percentiles)
    }
    result["mean"] = float(array.mean())
    return result


def _email_classification(match: re.Match[str]) -> str:
    value = match.group(0)
    local, domain = value.rsplit("@", 1)
    domain_lower = domain.lower()
    if domain_lower.endswith(".edu") or ".ac." in domain_lower:
        return "academic_contact"
    if local.lower() in CONTACT_LOCAL_PARTS:
        return "role_or_organization_contact"
    if domain_lower in {"example.com", "example.org", "example.net", "localhost"}:
        return "synthetic_or_documentation"
    return "potential_personal_or_publisher_contact"


def _ipv4_classification(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return "invalid_ipv4_shape"
    if address.is_loopback:
        return "loopback_address"
    if address.is_private:
        return "private_address"
    if address.is_reserved or address.is_multicast or address.is_unspecified:
        return "reserved_or_special_address"
    if not address.is_global:
        return "non_global_address"
    return "public_address"


def sensitive_scan(track: PilotTrack) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Return aggregate findings and non-reversible evidence without raw matches."""
    records = [
        ("document", row["id"], row["content"])
        for row in track.data.corpus
    ] + [
        ("query", row["id"], row["text"])
        for row in track.data.queries
    ]
    counts: Counter[tuple[str, str, str]] = Counter()
    evidence = []
    salt = f"{BENCHMARK_VERSION}\0sensitive-evidence\0{track.data.name}"
    for role, record_id, text in records:
        for pattern_name, pattern in (("email", EMAIL_PATTERN), ("ipv4", IPV4_PATTERN)):
            for ordinal, match in enumerate(pattern.finditer(text), 1):
                classification = (
                    _email_classification(match)
                    if pattern_name == "email"
                    else _ipv4_classification(match.group(0))
                )
                counts[(role, pattern_name, classification)] += 1
                context_start = max(0, match.start() - 48)
                context_end = min(len(text), match.end() + 48)
                evidence.append(
                    {
                        "track": track.data.name,
                        "role": role,
                        "record_id_sha256": _sha256_bytes(f"{salt}\0{record_id}".encode()),
                        "pattern": pattern_name,
                        "classification": classification,
                        "ordinal": ordinal,
                        "match_sha256": _sha256_bytes(f"{salt}\0{match.group(0)}".encode()),
                        "context_sha256": _sha256_bytes(text[context_start:context_end].encode()),
                        "safe_example": "***@<classified-domain>" if pattern_name == "email" else "x.x.x.x",
                        "raw_text_included": False,
                        "requires_manual_review": classification
                        in {"potential_personal_or_publisher_contact", "public_address"},
                    }
                )
        counts[(role, "url", "url_shape")] += len(URL_PATTERN.findall(text))
        counts[(role, "us_ssn", "us_ssn_shape")] += len(US_SSN_PATTERN.findall(text))
    findings: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (role, pattern_name, classification), count in sorted(counts.items()):
        findings[role][pattern_name][classification] = count
    manual_review = sum(row["requires_manual_review"] for row in evidence)
    return (
        {
            "findings": {role: dict(patterns) for role, patterns in findings.items()},
            "evidence_rows": len(evidence),
            "manual_review_required_rows": manual_review,
            "raw_matches_or_context_stored": False,
            "policy": (
                "Email and IPv4-shaped values remain inside the research-only source; committed summaries contain "
                "only aggregate classifications and non-reversible evidence hashes."
            ),
        },
        tuple(evidence),
    )


def audit_track(track: PilotTrack) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Audit one selected pilot track without exposing restricted text."""
    data = track.data
    document_lengths = [len(tokens(row["content"])) for row in data.corpus]
    query_lengths = [len(tokens(row["text"])) for row in data.queries]
    content_counts = Counter(row["content"] for row in data.corpus)
    normalized_counts = Counter(" ".join(tokens(row["content"])) for row in data.corpus)
    qrel_counts = [len(data.qrels[row["id"]]) for row in data.queries]
    gold_ids = [doc_id for rels in data.qrels.values() for doc_id in rels]
    sensitive, evidence = sensitive_scan(track)
    audit = {
        "track": data.name,
        "source_documents": EXPECTED[data.name]["documents"]["rows"],
        "documents": len(data.corpus),
        "queries": len(data.queries),
        "positive_qrels": len(gold_ids),
        "grade_counts": {"1": len(gold_ids)},
        "unique_gold_documents": len(set(gold_ids)),
        "qrels_per_query": _percentiles(qrel_counts),
        "qrel_density": len(gold_ids) / (len(data.corpus) * len(data.queries)),
        "single_positive_queries": sum(value == 1 for value in qrel_counts),
        "queries_with_more_than_20_positives": sum(value > 20 for value in qrel_counts),
        "query_token_lengths": _percentiles(query_lengths),
        "document_token_lengths": _percentiles(document_lengths),
        "zero_token_documents": sum(value == 0 for value in document_lengths),
        "documents_under_five_tokens": sum(value < 5 for value in document_lengths),
        "documents_over_512_tokens": sum(value > 512 for value in document_lengths),
        "exact_duplicate_content_groups": sum(value > 1 for value in content_counts.values()),
        "exact_duplicate_rows_beyond_first": sum(value - 1 for value in content_counts.values() if value > 1),
        "normalized_duplicate_content_groups": sum(
            bool(normalized) and value > 1 for normalized, value in normalized_counts.items()
        ),
        "selection": {
            "target_documents": TARGET_DOCUMENTS,
            "selection_salt": SELECTION_SALT,
            "selection_ids_sha256": SELECTION_ID_SHA256[data.name],
            "forced_gold_documents": len(set(gold_ids)),
            "hash_fillers": TARGET_DOCUMENTS - len(set(gold_ids)),
            "model_score_used": False,
            "policy": (
                "Keep every positive-qrel document; group remaining documents by exact content SHA256; choose the "
                "lexicographically smallest ID per group; order groups by SHA256(salt NUL track NUL content_sha256)."
            ),
        },
        "candidate_policy": {
            "pool": "independent_per_track",
            "candidates_per_query": TARGET_DOCUMENTS,
            "excluded_ids": "filter before ranking and metrics; never negative qrels",
            "excluded_ids_in_selected_revision": sum(len(row.get("excluded_ids", ())) for row in data.queries),
            "unjudged_documents": "not confirmed negatives",
        },
        "label_semantics": {
            "gold_ids": "grade-1 positive passage qrels",
            "gold_ids_long": "restricted provenance metadata only; never passage qrels",
            "missing_label_risk": "high enough to require review of baseline-ranked unjudged candidates",
            "hard_negative_status": "no explicit negative labels; only review candidates may be called likely hard negatives",
        },
        "provenance_and_time": {
            "source": HF_DATASET,
            "revision": HF_REVISION,
            "collection_window": "2024-03 through 2024-05",
            "benchmark_public_since": "2024-07",
            "paper_revision": "2025-03 v4",
            "document_time": "unknown_or_mixed",
            "training_overlap_status": "unknown_not_zero_shot_verified",
            "zero_shot_claim_allowed": False,
            "realism": "naturally occurring StackExchange queries with linked web documents and author curation",
        },
        "publication": {
            "classification": "research_only",
            "public_export_allowed": False,
            "gate": PUBLICATION_GATE,
            "reason": (
                "Repository-level CC-BY-4.0 does not prove sublicensable redistribution rights for every linked "
                "upstream web document."
            ),
        },
        "sensitive_information": sensitive,
    }
    return audit, evidence


def _qrel_stratum(count: int) -> str:
    if count == 1:
        return "single"
    if count <= 5:
        return "two_to_five"
    if count <= 20:
        return "six_to_twenty"
    return "over_twenty"


def review_plan_rows(tracks: dict[str, PilotTrack]) -> tuple[dict[str, Any], ...]:
    """Select 120 review queries before any baseline rankings exist."""
    output = []
    for track_name in TRACKS:
        track = tracks[track_name]
        lengths = [len(tokens(row["text"])) for row in track.data.queries]
        p33, p67 = np.percentile(lengths, [33, 67])
        source_by_id = {row["id"]: row for row in track.source_examples}
        strata: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for query in track.data.queries:
            length = len(tokens(query["text"]))
            length_stratum = "short" if length <= p33 else ("medium" if length <= p67 else "long")
            qrel_stratum = _qrel_stratum(len(track.data.qrels[query["id"]]))
            provenance_stratum = "single_long_source" if len(query["gold_ids_long"]) == 1 else "multiple_long_sources"
            strata[(length_stratum, qrel_stratum, provenance_stratum)].append(query)
        for key in strata:
            strata[key].sort(
                key=lambda row: (
                    _sha256_bytes(f"{REVIEW_SALT}\0{track_name}\0{row['id']}".encode()),
                    row["id"],
                )
            )
        selected = []
        positions = defaultdict(int)
        keys = sorted(strata)
        while len(selected) < REVIEW_PER_TRACK:
            progressed = False
            for key in keys:
                position = positions[key]
                if position < len(strata[key]):
                    selected.append((key, strata[key][position]))
                    positions[key] += 1
                    progressed = True
                    if len(selected) == REVIEW_PER_TRACK:
                        break
            if not progressed:
                raise BrightPilotError(f"cannot allocate review plan for {track_name}")
        for (length_stratum, qrel_stratum, provenance_stratum), query in selected:
            source = source_by_id[query["id"]]
            output.append(
                {
                    "track": track_name,
                    "query_id": query["id"],
                    "query": query["text"],
                    "gold_ids": sorted(track.data.qrels[query["id"]]),
                    "gold_ids_long": source["gold_ids_long"],
                    "strata": {
                        "query_length": length_stratum,
                        "qrel_density": qrel_stratum,
                        "gold_ids_long": provenance_stratum,
                    },
                    "selection": {
                        "fixed_before_model_runs": True,
                        "model_score_used": False,
                        "salt": REVIEW_SALT,
                    },
                }
            )
    return tuple(output)


def _file_entry(root: Path, relative: str, rows: int | None = None) -> dict[str, Any]:
    path = root / relative
    entry: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        entry["rows"] = rows
    return entry


def materialize(source_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Materialize both fixed independent tracks and the pre-model review plan."""
    tracks = {track: select_track(load_track(source_root, track)) for track in TRACKS}
    long_dense = _long_dense_preflight_for_tracks(tracks)
    output = Path(output_root)
    staging = output.with_name(f"{output.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    manifest_tracks = {}
    for track_name in TRACKS:
        track = tracks[track_name]
        track_root = staging / track_name
        track_root.mkdir()
        corpus_entry = write_jsonl(track_root / "corpus.jsonl", track.data.corpus)
        query_entry = write_jsonl(track_root / "queries.jsonl", track.data.queries)
        qrel_rows = (
            {"query_id": query_id, "document_id": doc_id, "grade": grade}
            for query_id in sorted(track.data.qrels)
            for doc_id, grade in sorted(track.data.qrels[query_id].items())
        )
        qrel_entry = write_jsonl(track_root / "qrels.jsonl", qrel_rows)
        selection_entry = write_jsonl(track_root / "selection.jsonl", track.selection)
        audit, sensitive_evidence = audit_track(track)
        audit_entry = _write_json(track_root / "audit.json", audit)
        sensitive_entry = write_jsonl(track_root / "sensitive-evidence.jsonl", sensitive_evidence)
        files = {
            "corpus.jsonl": {**corpus_entry, "path": "corpus.jsonl"},
            "queries.jsonl": {**query_entry, "path": "queries.jsonl"},
            "qrels.jsonl": {**qrel_entry, "path": "qrels.jsonl"},
            "selection.jsonl": {**selection_entry, "path": "selection.jsonl"},
            "audit.json": {**audit_entry, "path": "audit.json"},
            "sensitive-evidence.jsonl": {**sensitive_entry, "path": "sensitive-evidence.jsonl"},
        }
        manifest_tracks[track_name] = {
            "documents": len(track.data.corpus),
            "queries": len(track.data.queries),
            "positive_qrels": sum(map(len, track.data.qrels.values())),
            "selection_ids_sha256": SELECTION_ID_SHA256[track_name],
            "source_files": [
                {"role": role, "url": source_url(track_name, role), **EXPECTED[track_name][role]}
                for role in ("documents", "examples")
            ],
            "files": files,
        }
    review_rows = review_plan_rows(tracks)
    review_entry = write_jsonl(staging / "review-plan-120.jsonl", review_rows)
    content_sets = {
        track: {_content_sha256(row["content"]) for row in tracks[track].data.corpus}
        for track in TRACKS
    }
    manifest = {
        "schema_version": "1",
        "benchmark_version": BENCHMARK_VERSION,
        "source": {
            "dataset": HF_DATASET,
            "revision": HF_REVISION,
            "trust_remote_code": False,
        },
        "protocol": {
            "tracks": list(TRACKS),
            "independent_candidate_pools": True,
            "cross_track_or_biology_comparison_allowed": False,
            "text_protocol": "content.strip() for documents; query.strip() for queries",
            "qrels": "source gold_ids become grade-1 positives",
            "gold_ids_long": "restricted provenance only",
            "excluded_ids": "filter before ranking and metrics; never negative qrels",
            "ranking_depth": 100,
            "metrics": list(METRIC_NAMES),
        },
        "selection": {
            "salt": SELECTION_SALT,
            "target_documents_per_track": TARGET_DOCUMENTS,
            "model_score_used": False,
            "exact_content_deduplication": True,
            "policy": (
                "Keep all queries and positive-qrel documents, then select one representative per remaining exact "
                "content SHA256 group in fixed salted track-specific order."
            ),
        },
        "aggregate": {
            "tracks": 2,
            "documents": 15_000,
            "queries": 204,
            "positive_qrels": 1_492,
            "cross_track_shared_exact_content_values": len(content_sets[TRACKS[0]] & content_sets[TRACKS[1]]),
        },
        "review_plan": {
            **review_entry,
            "path": "review-plan-120.jsonl",
            "rows": len(review_rows),
            "allocation": {track: REVIEW_PER_TRACK for track in TRACKS},
            "fixed_before_model_runs": True,
            "model_score_used": False,
            "selection_salt": REVIEW_SALT,
        },
        "long_dense": long_dense,
        "publication": {
            "classification": "research_only",
            "public_export_allowed": False,
            "gate": PUBLICATION_GATE,
            "restricted_artifacts": [
                "corpus",
                "queries",
                "qrels",
                "rankings",
                "review_plan",
                "review_pack",
                "failure_cases",
            ],
        },
        "tracks": manifest_tracks,
    }
    (staging / "manifest.json").write_bytes(_json_bytes(manifest))
    validate_materialization(staging, source_root=source_root)
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    return manifest


def _validate_long_dense_manifest(value: dict[str, Any]) -> None:
    if value["method"] != "long_dense":
        raise BrightPilotError("invalid long-dense method contract")
    expected_model = {
        **LONG_DENSE_SNAPSHOT,
        "trust_remote_code": False,
        "representation": "SentenceTransformer dense representation",
        "declared_max_sequence_length": LONG_DENSE_DECLARED_MAX_SEQUENCE_LENGTH,
    }
    if value["model"] != expected_model:
        raise BrightPilotError("long-dense model identity mismatch")
    selection = value["cap_selection"]
    if selection["candidate_caps"] != list(LONG_DENSE_CAPS):
        raise BrightPilotError("long-dense cap candidates mismatch")
    if selection["selected_cap"] != LONG_DENSE_SELECTED_CAP:
        raise BrightPilotError("long-dense selected cap mismatch")
    if selection["score_used"] or not selection["fixed_before_model_scoring"]:
        raise BrightPilotError("long-dense cap selection used model scores")
    totals = {
        "economics": {"queries": 103, "candidates": 7_500, "gold_documents": 800},
        "psychology": {"queries": 101, "candidates": 7_500, "gold_documents": 688},
    }
    for track in TRACKS:
        for role, total in totals[track].items():
            evidence = selection["tracks"][track][role]
            if evidence["total"] != total:
                raise BrightPilotError("long-dense truncation denominator mismatch")
            for cap in LONG_DENSE_CAPS:
                actual = evidence["by_cap"][str(cap)]
                truncated = LONG_DENSE_EXPECTED_TRUNCATION[track][role][cap]
                if actual != {"truncated": truncated, "rate": truncated / total}:
                    raise BrightPilotError("long-dense truncation evidence mismatch")
    selected_rates = [
        selection["tracks"][track]["gold_documents"]["by_cap"][str(LONG_DENSE_SELECTED_CAP)]["rate"]
        for track in TRACKS
    ]
    prior_rates = [
        selection["tracks"][track]["gold_documents"]["by_cap"][str(LONG_DENSE_CAPS[0])]["rate"]
        for track in TRACKS
    ]
    if any(rate > LONG_DENSE_GOLD_TRUNCATION_THRESHOLD for rate in selected_rates):
        raise BrightPilotError("selected long-dense cap violates gold truncation threshold")
    if all(rate <= LONG_DENSE_GOLD_TRUNCATION_THRESHOLD for rate in prior_rates):
        raise BrightPilotError("selected long-dense cap is not minimal")
    if _sha256_bytes(_json_bytes(value)) != LONG_DENSE_CONTRACT_SHA256:
        raise BrightPilotError("long-dense contract identity mismatch")


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BrightPilotError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise BrightPilotError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return tuple(rows)


def _load_track_from_files(root: Path, track: str) -> PilotTrack:
    track_root = root / track
    corpus = _read_jsonl(track_root / "corpus.jsonl")
    queries = _read_jsonl(track_root / "queries.jsonl")
    qrel_rows = _read_jsonl(track_root / "qrels.jsonl")
    selection = _read_jsonl(track_root / "selection.jsonl")
    qrels: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for row in qrel_rows:
        if set(row) != {"query_id", "document_id", "grade"} or row["grade"] != 1:
            raise BrightPilotError(f"invalid qrel row for {track}")
        if row["document_id"] in qrels[row["query_id"]]:
            raise BrightPilotError(f"duplicate qrel row for {track}")
        qrels[row["query_id"]][row["document_id"]] = row["grade"]
    data = TrackData(track, corpus, queries, dict(qrels))
    try:
        validate_track_data(data)
    except ValueError as error:
        raise BrightPilotError(str(error)) from error
    source_examples = tuple(
        {
            "id": row["id"],
            "query": row["text"],
            "gold_ids": sorted(data.qrels[row["id"]]),
            "excluded_ids": row["excluded_ids"],
            "gold_ids_long": row["gold_ids_long"],
        }
        for row in queries
    )
    return PilotTrack(data, source_examples, selection)


def validate_materialization(root: str | Path, *, source_root: str | Path | None = None) -> dict[str, Any]:
    """Validate schema, file identities, source-derived content, and qrel invariants."""
    output = Path(root)
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BrightPilotError("missing or invalid pilot manifest") from error
    try:
        Draft202012Validator(_manifest_schema()).validate(manifest)
    except Exception as error:
        raise BrightPilotError(f"invalid pilot manifest schema: {error}") from error
    if _sha256_file(manifest_path) != CANONICAL_MANIFEST_SHA256:
        raise BrightPilotError("canonical materialization manifest identity mismatch")
    _validate_long_dense_manifest(manifest["long_dense"])
    expected_paths = {"manifest.json", "review-plan-120.jsonl"}
    for track in TRACKS:
        expected_paths.update(f"{track}/{name}" for name in MATERIALIZED_FILES)
    actual_paths = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise BrightPilotError("materialized file tree does not match the fixed contract")
    loaded = {}
    for track in TRACKS:
        track_manifest = manifest["tracks"][track]
        if set(track_manifest["files"]) != set(MATERIALIZED_FILES):
            raise BrightPilotError(f"materialized file set mismatch for {track}")
        canonical = CANONICAL_TRACK_FILES.get(track)
        for name, entry in track_manifest["files"].items():
            if entry["path"] != name:
                raise BrightPilotError(f"invalid materialized path for {track}/{name}")
            path = output / track / name
            if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256_file(path) != entry["sha256"]:
                raise BrightPilotError(f"materialized identity mismatch for {track}/{name}")
            if canonical and entry != canonical[name]:
                raise BrightPilotError(f"canonical identity mismatch for {track}/{name}")
        pilot_track = _load_track_from_files(output, track)
        if len(pilot_track.data.corpus) != TARGET_DOCUMENTS:
            raise BrightPilotError(f"document count mismatch for {track}")
        if len(pilot_track.data.queries) != TARGET_QUERIES[track]:
            raise BrightPilotError(f"query count mismatch for {track}")
        if sum(map(len, pilot_track.data.qrels.values())) != TARGET_QRELS[track]:
            raise BrightPilotError(f"qrel count mismatch for {track}")
        if _ids_sha256(row["id"] for row in pilot_track.data.corpus) != SELECTION_ID_SHA256[track]:
            raise BrightPilotError(f"selection identity mismatch for {track}")
        audit, evidence = audit_track(pilot_track)
        if json.loads((output / track / "audit.json").read_text(encoding="utf-8")) != audit:
            raise BrightPilotError(f"audit mismatch for {track}")
        if _read_jsonl(output / track / "sensitive-evidence.jsonl") != evidence:
            raise BrightPilotError(f"sensitive evidence mismatch for {track}")
        loaded[track] = pilot_track
    review_entry = manifest["review_plan"]
    review_path = output / review_entry["path"]
    if review_path.stat().st_size != review_entry["bytes"] or _sha256_file(review_path) != review_entry["sha256"]:
        raise BrightPilotError("review plan identity mismatch")
    if CANONICAL_REVIEW_PLAN and review_entry != CANONICAL_REVIEW_PLAN:
        raise BrightPilotError("canonical review plan identity mismatch")
    if _read_jsonl(review_path) != review_plan_rows(loaded):
        raise BrightPilotError("review plan content mismatch")
    if source_root is not None:
        for track in TRACKS:
            expected = select_track(load_track(source_root, track))
            actual = loaded[track]
            if actual.data != expected.data or actual.selection != expected.selection:
                raise BrightPilotError(f"source-derived content mismatch for {track}")
    return manifest


def load_materialized(root: str | Path, track: str) -> TrackData:
    """Load one validated independent candidate pool."""
    if track not in TRACKS:
        raise BrightPilotError(f"unsupported pilot track: {track}")
    validate_materialization(root)
    return _load_track_from_files(Path(root), track).data


def verify_long_dense_preflight(root: str | Path) -> dict[str, Any]:
    """Recompute tokenizer-only cap evidence and match the frozen manifest contract."""
    data_path = Path(root)
    manifest = validate_materialization(data_path)
    tracks = {track: _load_track_from_files(data_path, track) for track in TRACKS}
    actual = _long_dense_preflight_for_tracks(tracks)
    if actual != manifest["long_dense"]:
        raise BrightPilotError("long-dense preflight does not match the materialization manifest")
    return actual


def _ranking_rows(rankings: dict[str, list[tuple[str, float]]]) -> Iterable[dict[str, Any]]:
    for query_id in sorted(rankings):
        for rank, (document_id, score) in enumerate(rankings[query_id], 1):
            yield {
                "query_id": query_id,
                "rank": rank,
                "document_id": document_id,
                "score": score,
            }


def _metric_rows(per_query: dict[str, dict[str, float]]) -> Iterable[dict[str, Any]]:
    for query_id in sorted(per_query):
        yield {"query_id": query_id, "metrics": per_query[query_id]}


def _data_identity(manifest: dict[str, Any], manifest_path: Path, track: str) -> dict[str, Any]:
    return {
        "materialization_manifest_sha256": _sha256_file(manifest_path),
        "source_revision": manifest["source"]["revision"],
        "selection_ids_sha256": manifest["tracks"][track]["selection_ids_sha256"],
        "corpus_sha256": manifest["tracks"][track]["files"]["corpus.jsonl"]["sha256"],
        "queries_sha256": manifest["tracks"][track]["files"]["queries.jsonl"]["sha256"],
        "qrels_sha256": manifest["tracks"][track]["files"]["qrels.jsonl"]["sha256"],
        "selection_sha256": manifest["tracks"][track]["files"]["selection.jsonl"]["sha256"],
        "candidate_policy": "independent fixed 7,500-document track pool with pre-ranking excluded_ids filtering",
    }


def run_baseline(data_root: str | Path, output_root: str | Path, method: str) -> dict[str, Any]:
    """Run one complete baseline over both fixed independent tracks."""
    if method not in {"bm25", "dense", "long_dense"}:
        raise BrightPilotError(f"unsupported baseline method: {method}")
    data_path = Path(data_root)
    manifest = validate_materialization(data_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    if method == "long_dense":
        verify_long_dense_preflight(data_path)
        model_identity = snapshot_identity(LONG_DENSE_MODEL_ID, LONG_DENSE_MODEL_REVISION)
    elif method == "dense":
        model_identity = snapshot_identity(MODEL_ID, MODEL_REVISION)
    else:
        model_identity = None
    result_entries = {}
    for track in TRACKS:
        data = _load_track_from_files(data_path, track).data
        started = time.perf_counter()
        if method == "bm25":
            rankings, execution = bm25_rank(data)
        elif method == "dense":
            rankings, execution = dense_rank(data, MODEL_ID, MODEL_REVISION)
            execution["model_snapshot"] = model_identity
        else:
            rankings, execution = dense_rank(
                data,
                LONG_DENSE_MODEL_ID,
                LONG_DENSE_MODEL_REVISION,
                batch_size=LONG_DENSE_BATCH_SIZE,
                model_path=LONG_DENSE_SNAPSHOT_PATH,
                max_sequence_length=LONG_DENSE_SELECTED_CAP,
            )
            execution["cap_selection"] = {
                "selected_cap": LONG_DENSE_SELECTED_CAP,
                "gold_truncation_threshold": LONG_DENSE_GOLD_TRUNCATION_THRESHOLD,
                "fixed_before_model_scoring": True,
                "score_used": False,
                "materialization_contract_sha256": LONG_DENSE_CONTRACT_SHA256,
            }
            execution["model_snapshot"] = model_identity
        wall_time = time.perf_counter() - started
        per_query = query_metrics(data, rankings)
        ranking_name = f"{method}-{track}.rankings.jsonl"
        metric_name = f"{method}-{track}.per-query.jsonl"
        ranking_entry = write_jsonl(output / ranking_name, _ranking_rows(rankings))
        metric_entry = write_jsonl(output / metric_name, _metric_rows(per_query))
        execution = {
            **execution,
            "wall_time_s": wall_time,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        result = {
            "schema_version": "1",
            "benchmark_version": BENCHMARK_VERSION,
            "track": track,
            "method": method,
            "baseline_role": {
                "bm25": "formal_lexical",
                "dense": "diagnostic_truncation_degraded",
                "long_dense": "formal_long_context_dense",
            }[method],
            "queries_evaluated": len(data.queries),
            "documents_searched": len(data.corpus),
            "positive_qrels": sum(map(len, data.qrels.values())),
            "metrics": {
                metric: sum(values[metric] for values in per_query.values()) / len(per_query)
                for metric in METRIC_NAMES
            },
            "confidence_intervals": bootstrap_confidence_intervals(per_query),
            "execution": execution,
            "rankings": {**ranking_entry, "path": ranking_name, "depth": 100},
            "per_query_metrics": {**metric_entry, "path": metric_name},
            "data_identity": _data_identity(manifest, data_path / "manifest.json", track),
            "training_overlap": {
                "status": "unknown_not_zero_shot_verified",
                "zero_shot_claim_allowed": False,
            },
            "publication": {
                "classification": "research_only",
                "public_export_allowed": False,
                "gate": PUBLICATION_GATE,
            },
            "failure_counts": {
                "zero_recall_at_10": sum(values["recall@10"] == 0.0 for values in per_query.values()),
                "zero_recall_at_100": sum(values["recall@100"] == 0.0 for values in per_query.values()),
                "positive_recall_at_10": sum(values["recall@10"] > 0.0 for values in per_query.values()),
            },
        }
        try:
            Draft202012Validator(_result_schema()).validate(result)
        except Exception as error:
            raise BrightPilotError(f"invalid {method}/{track} result: {error}") from error
        result_path = output / f"{method}-{track}.json"
        result_path.write_bytes(_json_bytes(result))
        result_entries[track] = {
            "path": result_path.name,
            "bytes": result_path.stat().st_size,
            "sha256": _sha256_file(result_path),
            "rankings": result["rankings"],
            "per_query_metrics": result["per_query_metrics"],
        }
    run_manifest = {
        "schema_version": "1",
        "benchmark_version": BENCHMARK_VERSION,
        "method": method,
        "tracks": result_entries,
        "publication": {
            "classification": "research_only",
            "public_export_allowed": False,
            "gate": PUBLICATION_GATE,
        },
    }
    run_path = output / f"{method}-run-manifest.json"
    run_path.write_bytes(_json_bytes(run_manifest))
    return run_manifest


def _validate_ranking_rows(
    data: TrackData,
    rows: tuple[dict[str, Any], ...],
    *,
    depth: int,
) -> dict[str, list[tuple[str, float]]]:
    """Validate exact ranking coverage and order against a materialized track."""
    validate_track_data(data)
    query_ids = [row["id"] for row in data.queries]
    document_ids = [row["id"] for row in data.corpus]
    document_id_set = set(document_ids)
    document_positions = {document_id: position for position, document_id in enumerate(document_ids)}
    excluded_by_query = {row["id"]: set(row.get("excluded_ids", ())) for row in data.queries}
    if len(rows) != len(query_ids) * depth:
        raise BrightPilotError("ranking row count does not match query count and depth")
    rankings: dict[str, list[tuple[str, float]]] = {query_id: [] for query_id in query_ids}
    expected_file_order = []
    for query_id in sorted(query_ids):
        expected_file_order.extend((query_id, rank) for rank in range(1, depth + 1))
    actual_file_order = []
    for row in rows:
        if set(row) != {"query_id", "rank", "document_id", "score"}:
            raise BrightPilotError("ranking row fields do not match the fixed contract")
        query_id = row["query_id"]
        rank = row["rank"]
        document_id = row["document_id"]
        score = row["score"]
        if query_id not in rankings or type(rank) is not int or not isinstance(document_id, str):
            raise BrightPilotError("ranking row contains an invalid query, rank, or document")
        if type(score) not in {int, float} or not math.isfinite(score):
            raise BrightPilotError("ranking score must be finite")
        if document_id not in document_id_set or document_id in excluded_by_query[query_id]:
            raise BrightPilotError("ranking document is outside the query candidate pool")
        actual_file_order.append((query_id, rank))
        rankings[query_id].append((document_id, float(score)))
    if actual_file_order != expected_file_order:
        raise BrightPilotError("ranking rows do not have complete sorted query/rank coverage")
    for query_id, values in rankings.items():
        ranked_ids = [document_id for document_id, _ in values]
        if len(ranked_ids) != len(set(ranked_ids)):
            raise BrightPilotError(f"ranking contains duplicate documents for query {query_id!r}")
        for previous, current in zip(values, values[1:]):
            previous_id, previous_score = previous
            current_id, current_score = current
            if current_score > previous_score:
                raise BrightPilotError(f"ranking scores are not descending for query {query_id!r}")
            if current_score == previous_score and document_positions[current_id] < document_positions[previous_id]:
                raise BrightPilotError(f"ranking tie order is invalid for query {query_id!r}")
    return rankings


def _validate_per_query_rows(
    data: TrackData,
    rows: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, float]]:
    """Validate strict per-query metric rows and complete query coverage."""
    expected_query_ids = sorted(row["id"] for row in data.queries)
    if len(rows) != len(expected_query_ids):
        raise BrightPilotError("per-query metric row count mismatch")
    actual_query_ids = []
    output = {}
    for row in rows:
        if set(row) != {"query_id", "metrics"} or not isinstance(row["metrics"], dict):
            raise BrightPilotError("per-query metric row fields do not match the fixed contract")
        query_id = row["query_id"]
        metrics = row["metrics"]
        if set(metrics) != set(METRIC_NAMES):
            raise BrightPilotError("per-query metric names do not match the fixed protocol")
        if any(
            type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1
            for value in metrics.values()
        ):
            raise BrightPilotError("per-query metric value is invalid")
        actual_query_ids.append(query_id)
        output[query_id] = {metric: float(metrics[metric]) for metric in METRIC_NAMES}
    if actual_query_ids != expected_query_ids or len(output) != len(expected_query_ids):
        raise BrightPilotError("per-query metrics do not cover the sorted query set exactly")
    return output


def _load_result_bound(
    results_root: Path,
    method: str,
    track: str,
    *,
    data: TrackData,
    materialization_manifest: dict[str, Any],
    materialization_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, list[tuple[str, float]]], dict[str, dict[str, float]]]:
    """Load and replay one result against an already validated materialization."""
    if track not in TRACKS or method not in BASELINE_METHODS or data.name != track:
        raise BrightPilotError("result track or method is not bound to the materialized track")
    path = results_root / f"{method}-{track}.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator(_result_schema()).validate(result)
    except Exception as error:
        raise BrightPilotError(f"invalid stored result: {method}/{track}: {error}") from error
    expected_identity = _data_identity(materialization_manifest, materialization_manifest_path, track)
    if result["data_identity"] != expected_identity or result["track"] != track or result["method"] != method:
        raise BrightPilotError("result identity does not match the materialized track")
    ranking_path = results_root / result["rankings"]["path"]
    metric_path = results_root / result["per_query_metrics"]["path"]
    canonical_artifacts = CANONICAL_RESULT_ARTIFACTS[(method, track)]
    expected_files = {path.name, ranking_path.name, metric_path.name}
    actual_files = {candidate.name for candidate in results_root.glob(f"{method}-{track}*") if candidate.is_file()}
    if actual_files != expected_files:
        raise BrightPilotError("result artifact file set does not match the fixed contract")
    for artifact_path, entry in (
        (ranking_path, result["rankings"]),
        (metric_path, result["per_query_metrics"]),
    ):
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != entry["bytes"]
            or _sha256_file(artifact_path) != entry["sha256"]
        ):
            raise BrightPilotError(f"artifact identity mismatch: {artifact_path.name}")
    ranking_rows = _read_jsonl(ranking_path)
    metric_rows = _read_jsonl(metric_path)
    if len(ranking_rows) != result["rankings"]["rows"] or len(metric_rows) != result["per_query_metrics"]["rows"]:
        raise BrightPilotError("artifact row count does not match the result contract")
    rankings = _validate_ranking_rows(data, ranking_rows, depth=result["rankings"]["depth"])
    stored_per_query = _validate_per_query_rows(data, metric_rows)
    replayed_per_query = query_metrics(data, rankings)
    if stored_per_query != replayed_per_query:
        raise BrightPilotError("stored per-query metrics do not match replayed rankings and qrels")
    if result["metrics"] != aggregate_metrics(replayed_per_query):
        raise BrightPilotError("aggregate metrics do not match replayed per-query metrics")
    if result["confidence_intervals"] != bootstrap_confidence_intervals(replayed_per_query):
        raise BrightPilotError("confidence intervals do not match the fixed bootstrap replay")
    expected_failures = {
        "zero_recall_at_10": sum(values["recall@10"] == 0.0 for values in replayed_per_query.values()),
        "zero_recall_at_100": sum(values["recall@100"] == 0.0 for values in replayed_per_query.values()),
        "positive_recall_at_10": sum(values["recall@10"] > 0.0 for values in replayed_per_query.values()),
    }
    if result["failure_counts"] != expected_failures:
        raise BrightPilotError("failure counts do not match replayed per-query metrics")
    if (
        result["rankings"]["sha256"] != canonical_artifacts["rankings"]
        or result["per_query_metrics"]["sha256"] != canonical_artifacts["per_query_metrics"]
    ):
        raise BrightPilotError("result artifact hashes do not match the fixed pilot run")
    return result, rankings, replayed_per_query


def _load_result(
    results_root: Path,
    data_root: Path,
    method: str,
    track: str,
) -> tuple[dict[str, Any], dict[str, list[tuple[str, float]]], dict[str, dict[str, float]]]:
    """Load one result with a freshly validated materialization trust root."""
    manifest = validate_materialization(data_root)
    data = _load_track_from_files(data_root, track).data
    return _load_result_bound(
        results_root,
        method,
        track,
        data=data,
        materialization_manifest=manifest,
        materialization_manifest_path=data_root / "manifest.json",
    )


def _review_candidate_rows(
    data: TrackData,
    query_id: str,
    rankings_by_method: dict[str, dict[str, list[tuple[str, float]]]],
) -> list[dict[str, Any]]:
    documents = {row["id"]: row["content"] for row in data.corpus}
    positions: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for method in ("bm25", "long_dense", "dense"):
        if method not in rankings_by_method:
            continue
        for rank, (document_id, score) in enumerate(rankings_by_method[method][query_id][:10], 1):
            positions[document_id][method] = {"rank": rank, "score": score}
    rows = []
    for document_id in sorted(
        positions,
        key=lambda value: (
            min(item["rank"] for item in positions[value].values()),
            value,
        ),
    ):
        grade = data.qrels[query_id].get(document_id)
        rows.append(
            {
                "document_id": document_id,
                "content": documents[document_id],
                "baseline_positions": positions[document_id],
                "label_status": "positive_qrel" if grade is not None else "unjudged_candidate",
                "judged_grade": grade,
                "review": {
                    "relevance": None,
                    "likely_hard_negative": None,
                    "suspected_missing_positive": None,
                    "sensitive_information": None,
                    "notes": "",
                },
            }
        )
    return rows


def build_review_pack(data_root: str | Path, results_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Attach both baseline top-10 unions to the pre-model 120-query plan."""
    data_path = Path(data_root)
    manifest = validate_materialization(data_path)
    result_path = Path(results_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    plan = _read_jsonl(data_path / manifest["review_plan"]["path"])
    loaded_results = {}
    rankings = defaultdict(dict)
    per_query = defaultdict(dict)
    data_by_track = {}
    for track in TRACKS:
        data_by_track[track] = _load_track_from_files(data_path, track).data
        for method in BASELINE_METHODS:
            result, method_rankings, method_metrics = _load_result_bound(
                result_path,
                method,
                track,
                data=data_by_track[track],
                materialization_manifest=manifest,
                materialization_manifest_path=data_path / "manifest.json",
            )
            loaded_results[(track, method)] = result
            rankings[track][method] = method_rankings
            per_query[track][method] = method_metrics
    pack_rows = []
    for plan_row in plan:
        track = plan_row["track"]
        query_id = plan_row["query_id"]
        data = data_by_track[track]
        documents = {row["id"]: row["content"] for row in data.corpus}
        pack_rows.append(
            {
                **plan_row,
                "gold_documents": [
                    {
                        "document_id": document_id,
                        "content": documents[document_id],
                        "grade": data.qrels[query_id][document_id],
                        "review": {"supports_query": None, "ambiguous": None, "notes": ""},
                    }
                    for document_id in sorted(data.qrels[query_id])
                ],
                "baseline_top10_union": _review_candidate_rows(data, query_id, rankings[track]),
                "review_summary": {
                    "qrels_complete": None,
                    "gold_support_failures": [],
                    "suspected_missing_positive_ids": [],
                    "likely_hard_negative_ids": [],
                    "notes": "",
                },
            }
        )
    pack_entry = write_jsonl(output / "audit-pack-120.jsonl", pack_rows)
    failure_rows = []
    safe_cases = []
    failure_category_counts = {}
    for track in TRACKS:
        data = data_by_track[track]
        query_by_id = {row["id"]: row for row in data.queries}
        categories = defaultdict(list)
        for query_id in sorted(query_by_id):
            bm25_hit = per_query[track]["bm25"][query_id]["recall@10"] > 0
            dense_hit = per_query[track]["long_dense"][query_id]["recall@10"] > 0
            category = (
                "both_hit_top10"
                if bm25_hit and dense_hit
                else "bm25_only_top10"
                if bm25_hit
                else "long_dense_only_top10"
                if dense_hit
                else "both_miss_top10"
            )
            categories[category].append(query_id)
        failure_category_counts[track] = {category: len(values) for category, values in sorted(categories.items())}
        for category in sorted(categories):
            selected = sorted(
                categories[category],
                key=lambda value: (_sha256_bytes(f"{BENCHMARK_VERSION}\0failure\0{track}\0{value}".encode()), value),
            )[:2]
            for query_id in selected:
                query = query_by_id[query_id]
                case_hash = _sha256_bytes(f"{BENCHMARK_VERSION}\0case\0{track}\0{query_id}".encode())
                failure_rows.append(
                    {
                        "case_sha256": case_hash,
                        "track": track,
                        "category": category,
                        "query_id": query_id,
                        "query": query["text"],
                        "gold_ids": sorted(data.qrels[query_id]),
                        "bm25_top10": _review_candidate_rows(data, query_id, {"bm25": rankings[track]["bm25"]}),
                        "long_dense_top10": _review_candidate_rows(
                            data, query_id, {"long_dense": rankings[track]["long_dense"]}
                        ),
                        "minilm_diagnostic_top10": _review_candidate_rows(
                            data, query_id, {"dense": rankings[track]["dense"]}
                        ),
                        "metrics": {
                            method: per_query[track][method][query_id]
                            for method in BASELINE_METHODS
                        },
                    }
                )
                safe_cases.append(
                    {
                        "case_sha256": case_hash,
                        "track": track,
                        "category": category,
                        "query_tokens": len(tokens(query["text"])),
                        "positive_qrels": len(data.qrels[query_id]),
                        "metrics": {
                            method: per_query[track][method][query_id]
                            for method in BASELINE_METHODS
                        },
                        "restricted_text_included": False,
                    }
                )
    failure_entry = write_jsonl(output / "failure-cases.jsonl", failure_rows)
    safe_failure_entry = write_jsonl(output / "failure-cases-safe.jsonl", safe_cases)
    audit_manifest = {
        "schema_version": "1",
        "benchmark_version": BENCHMARK_VERSION,
        "review_plan_sha256": manifest["review_plan"]["sha256"],
        "selection_fixed_before_model_runs": True,
        "unjudged_candidates_are_negatives": False,
        "official_qrels_modified": False,
        "formal_baselines": list(FORMAL_METHODS),
        "diagnostic_baselines": ["dense"],
        "audit_pack": {**pack_entry, "path": "audit-pack-120.jsonl"},
        "failure_cases": {**failure_entry, "path": "failure-cases.jsonl"},
        "safe_failure_cases": {**safe_failure_entry, "path": "failure-cases-safe.jsonl"},
        "failure_category_counts": failure_category_counts,
        "review_protocol": {
            "queries": 120,
            "allocation": {track: 60 for track in TRACKS},
            "double_review_queries": 24,
            "agreement_metric": "Cohen kappa over binary relevance",
            "minimum_agreement": 0.70,
            "stop_if_gold_unsupported_or_ambiguous_fraction_exceeds": 0.05,
            "stop_if_queries_with_credible_missing_positive_fraction_exceeds": 0.10,
        },
        "publication": {
            "classification": "research_only",
            "public_export_allowed": False,
            "gate": PUBLICATION_GATE,
        },
    }
    (output / "audit-manifest.json").write_bytes(_json_bytes(audit_manifest))
    return audit_manifest


def _safe_execution(method: str, execution: dict[str, Any]) -> dict[str, Any]:
    safe = {key: value for key, value in execution.items() if key != "model_snapshot"}
    if method in {"dense", "long_dense"}:
        snapshot = execution["model_snapshot"]
        safe["model_snapshot"] = {
            key: snapshot[key]
            for key in ("repo_id", "revision", "files", "bytes", "aggregate_sha256")
        }
    return safe


def write_safe_summary(
    data_root: str | Path,
    results_root: str | Path,
    audit_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Write a commit-safe aggregate summary with no restricted rows or identifiers."""
    data_path = Path(data_root)
    results_path = Path(results_root)
    audit_path = Path(audit_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = validate_materialization(data_path)
    audit_manifest = json.loads((audit_path / "audit-manifest.json").read_text(encoding="utf-8"))
    if audit_manifest["review_plan_sha256"] != manifest["review_plan"]["sha256"]:
        raise BrightPilotError("audit pack does not match the materialization review plan")
    tracks = {}
    macro = {method: {metric: 0.0 for metric in METRIC_NAMES} for method in FORMAL_METHODS}
    diagnostic_macro = {metric: 0.0 for metric in METRIC_NAMES}
    artifact_identities = {}
    for track in TRACKS:
        audit = json.loads((data_path / track / "audit.json").read_text(encoding="utf-8"))
        data = _load_track_from_files(data_path, track).data
        track_summary = {
            "documents": TARGET_DOCUMENTS,
            "queries": TARGET_QUERIES[track],
            "positive_qrels": TARGET_QRELS[track],
            "selection_ids_sha256": SELECTION_ID_SHA256[track],
            "governance_audit": audit,
        }
        method_metrics = {}
        for method in BASELINE_METHODS:
            result, _, per_query = _load_result_bound(
                results_path,
                method,
                track,
                data=data,
                materialization_manifest=manifest,
                materialization_manifest_path=data_path / "manifest.json",
            )
            method_metrics[method] = per_query
            summary_key = "minilm_diagnostic" if method == "dense" else method
            track_summary[summary_key] = {
                "baseline_role": result["baseline_role"],
                "metrics": result["metrics"],
                "confidence_intervals": result["confidence_intervals"],
                "failure_counts": result["failure_counts"],
                "execution": _safe_execution(method, result["execution"]),
            }
            for metric in METRIC_NAMES:
                if method in FORMAL_METHODS:
                    macro[method][metric] += result["metrics"][metric] / len(TRACKS)
                else:
                    diagnostic_macro[metric] += result["metrics"][metric] / len(TRACKS)
            artifact_identities[f"{method}-{track}"] = {
                "result_sha256": _sha256_file(results_path / f"{method}-{track}.json"),
                "rankings_sha256": result["rankings"]["sha256"],
                "per_query_metrics_sha256": result["per_query_metrics"]["sha256"],
            }
        track_summary["long_dense_minus_bm25_paired_bootstrap"] = paired_bootstrap_deltas(
            method_metrics["long_dense"],
            method_metrics["bm25"],
        )
        tracks[track] = track_summary
    safe_failure_rows = _read_jsonl(audit_path / audit_manifest["safe_failure_cases"]["path"])
    summary = {
        "schema_version": "1",
        "benchmark_version": BENCHMARK_VERSION,
        "scope": {
            "tracks": list(TRACKS),
            "independent_candidate_pools": True,
            "documents": 15_000,
            "queries": 204,
            "positive_qrels": 1_492,
            "comparison_to_full_corpus_biology_or_official_bright_allowed": False,
        },
        "source": manifest["source"],
        "selection": manifest["selection"],
        "publication": manifest["publication"],
        "review_plan": {
            "rows": manifest["review_plan"]["rows"],
            "sha256": manifest["review_plan"]["sha256"],
            "fixed_before_model_runs": True,
            "model_score_used": False,
        },
        "long_dense_contract": manifest["long_dense"],
        "tracks": tracks,
        "unweighted_macro": macro,
        "minilm_diagnostic_unweighted_macro": diagnostic_macro,
        "failure_cases": list(safe_failure_rows),
        "failure_category_counts": audit_manifest["failure_category_counts"],
        "artifact_identities": {
            "materialization_manifest_sha256": _sha256_file(data_path / "manifest.json"),
            "review_pack_sha256": audit_manifest["audit_pack"]["sha256"],
            "failure_cases_sha256": audit_manifest["failure_cases"]["sha256"],
            **artifact_identities,
        },
        "claim_boundaries": [
            "Results apply only to the fixed 7,500-document per-track pilot slices.",
            "Unjudged documents are not confirmed negatives and official qrels were not modified.",
            "Formal dense comparisons use BGE-M3 long_dense; MiniLM is truncation-degraded diagnostic evidence only.",
            "Training overlap is unknown and no verified zero-shot claim is allowed.",
            "Restricted source text, queries, qrels, rankings, and review artifacts are not included.",
        ],
        "restricted_text_or_identifiers_included": False,
    }
    summary_path = output / "summary.json"
    summary_path.write_bytes(_json_bytes(summary))
    safe_manifest = {
        "schema_version": "1",
        "benchmark_version": BENCHMARK_VERSION,
        "summary": {
            "path": "summary.json",
            "bytes": summary_path.stat().st_size,
            "sha256": _sha256_file(summary_path),
        },
        "restricted_text_or_identifiers_included": False,
        "publication": manifest["publication"],
    }
    (output / "manifest.json").write_bytes(_json_bytes(safe_manifest))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--source-root", required=True)
    materialize_parser.add_argument("--output", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data", required=True)
    validate_parser.add_argument("--source-root")
    preflight_parser = subparsers.add_parser("long-dense-preflight")
    preflight_parser.add_argument("--data", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--data", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--method", choices=BASELINE_METHODS, required=True)
    review_parser = subparsers.add_parser("review-pack")
    review_parser.add_argument("--data", required=True)
    review_parser.add_argument("--results", required=True)
    review_parser.add_argument("--output", required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--data", required=True)
    summary_parser.add_argument("--results", required=True)
    summary_parser.add_argument("--audit", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "materialize":
        result = materialize(args.source_root, args.output)
    elif args.command == "validate":
        result = validate_materialization(args.data, source_root=args.source_root)
    elif args.command == "long-dense-preflight":
        result = verify_long_dense_preflight(args.data)
    elif args.command == "run":
        result = run_baseline(args.data, args.output, args.method)
    elif args.command == "review-pack":
        result = build_review_pack(args.data, args.results, args.output)
    else:
        result = write_safe_summary(args.data, args.results, args.audit, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
