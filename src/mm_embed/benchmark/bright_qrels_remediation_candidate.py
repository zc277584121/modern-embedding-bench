"""Build and validate the full restricted BRIGHT remediation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mm_embed.benchmark import bright_qrels_remediation as control

CANDIDATE_ID = "bright-qrels-remediation-candidate-v0.1"
REPAIR_ROUND = 2
RESTRICTED_ROOT = Path("results/bright-qrels-remediation-v0.1/restricted")
TRACKED_ROOT = Path("benchmark/artifacts/bright-qrels-remediation-v0.1")
REPORT_PATH = Path("benchmark/research/bright_qrels_remediation_candidate_20260906.md")
REVIEW_SESSION = "ccu-codex-modern-embedding-bench-20260906181130-2009367"
REVIEW_TIME = "2026-09-06T18:30:00Z"
REVIEW_ROLE = "remediation_executor_primary"
CONTROL_REPAIR_TIME = "2026-09-06T20:30:00Z"
PRE_REPAIR_CONTROL_SHA256 = "d05de5d23f5cdff6a5dd12a38f6e856fecb634239641ef0f689e39b62e3dd9c2"
PRE_REPAIR_PROPOSAL_SCHEMA_SHA256 = "de3f42104e4325b86ff628ae5928a36114d0271f6ce19bce7aae20e5d9e8b389"
SCHEMA_NAMES = (
    "bright-qrels-remediation-diff-v01.schema.json",
    "bright-qrels-remediation-evidence-ledger-v01.schema.json",
    "bright-qrels-remediation-query-risk-v01.schema.json",
    "bright-qrels-remediation-restricted-manifest-v01.schema.json",
    "bright-qrels-remediation-safe-aggregate-v01.schema.json",
)
RESTRICTED_BUNDLE_NAMES = (
    "proposal.json",
    "proposal.json.sha256",
    "original-vs-proposed.jsonl",
    "original-vs-proposed.jsonl.sha256",
    "evidence-ledger.jsonl",
    "evidence-ledger.jsonl.sha256",
    "query-risk.jsonl",
    "query-risk.jsonl.sha256",
    "manifest.json",
    "manifest.json.sha256",
)
PROTECTED_SURFACE_IDENTITIES = {
    "configs/default.yaml": "55bb93a824bae87dd248aa14449276c2b604128e06b7f5854b2577f0417f9846",
    "benchmark/tasks/core.yaml": "f517fba1947f1e8b9454762c424d449f750d2308b25aee1f969cb0234c7ea78c",
    "src/mm_embed/benchmark/registry.py": "be51679ee411bedbfbcabf578d9e472949712123316ec1b4e24d907e32dbde2c",
    "src/mm_embed/benchmark/runner.py": "43b1578690b691188ffbe305ec1f0ab3b4dfb289041c3f0d3b93bbb36db0a4b2",
    "src/mm_embed/benchmark/leaderboard.py": "0437923ae3d265cb6f8ff04032b1d83ea378608617e36cdd0e68dc58cbfd724f",
    "src/mm_embed/hf_publish/export.py": "4c5b12e75c67a827ab4b00e12e2563cbe75df77b10343e58595498e43b78a3fc",
}
PROTECTED_SURFACES_SHA256 = "8d4b79c1b5ab17f7e0922009fa3ecd0139f311ac5a40b7d785d47950ad85154c"
TRACKED_CANDIDATE_PATHS = tuple(
    sorted(
        (
            "benchmark/artifacts/bright-qrels-remediation-v0.1/candidate-aggregate.json",
            "benchmark/artifacts/bright-qrels-remediation-v0.1/candidate-aggregate.json.sha256",
            "benchmark/artifacts/bright-qrels-remediation-v0.1/control-freeze.json",
            "benchmark/artifacts/bright-qrels-remediation-v0.1/control-freeze.json.sha256",
            "benchmark/research/bright_qrels_remediation_candidate_20260906.md",
            "benchmark/research/bright_qrels_remediation_candidate_20260906.md.sha256",
            "benchmark/research/bright_qrels_remediation_control_20260906.md",
            "schemas/bright-qrels-remediation-control-v01.schema.json",
            "schemas/bright-qrels-remediation-diff-v01.schema.json",
            "schemas/bright-qrels-remediation-evidence-ledger-v01.schema.json",
            "schemas/bright-qrels-remediation-proposal-v01.schema.json",
            "schemas/bright-qrels-remediation-query-risk-v01.schema.json",
            "schemas/bright-qrels-remediation-restricted-manifest-v01.schema.json",
            "schemas/bright-qrels-remediation-safe-aggregate-v01.schema.json",
            "scripts/bright_qrels_remediation.py",
            "scripts/build_bright_qrels_remediation_candidate.py",
            "src/mm_embed/benchmark/bright_qrels_remediation.py",
            "src/mm_embed/benchmark/bright_qrels_remediation_candidate.py",
            "tests/test_bright_qrels_remediation.py",
            "tests/test_bright_qrels_remediation_candidate.py",
        )
    )
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def candidate_identities(repository: str | Path | None = None) -> dict[str, Any]:
    """Return exact ordered path and path-bound content identities."""
    repo = Path(repository) if repository is not None else control._root()
    path_payload = "".join(f"{path}\n" for path in TRACKED_CANDIDATE_PATHS).encode("utf-8")
    content_payload = "".join(
        f"{path}\0{control._sha256_file(repo / path)}\n" for path in TRACKED_CANDIDATE_PATHS
    ).encode("utf-8")
    return {
        "paths": TRACKED_CANDIDATE_PATHS,
        "count": len(TRACKED_CANDIDATE_PATHS),
        "ordered_path_identity_sha256": hashlib.sha256(path_payload).hexdigest(),
        "ordered_content_identity_sha256": hashlib.sha256(content_payload).hexdigest(),
    }


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _write_bytes(path: Path, value: bytes) -> dict[str, Any]:
    control._assert_no_symlink_chain(path.parent)
    try:
        target_mode = os.lstat(path).st_mode
    except FileNotFoundError:
        target_mode = None
    if target_mode is not None and stat.S_ISLNK(target_mode):
        raise control.BrightQrelsRemediationError(f"refusing to replace symlink output: {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        directory_fd = os.open(path.parent, directory_flags)
        try:
            file_fd = os.open(temporary.name, flags, 0o600, dir_fd=directory_fd)
            with os.fdopen(file_fd, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary.name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise control.BrightQrelsRemediationError(f"cannot safely write output: {path.name}") from error
    return {"rows": value.count(b"\n"), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    result = _write_bytes(path, _json_bytes(value))
    result["rows"] = 1
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _write_bytes(path, _jsonl_bytes(rows))


def _write_sidecar(path: Path, sha256: str) -> None:
    _write_bytes(path.with_suffix(path.suffix + ".sha256"), (sha256 + "\n").encode("ascii"))


def _row_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _validate_protected_surfaces(repository: Path, overrides: dict[str, Path] | None = None) -> None:
    records = []
    for relative_path, expected_sha256 in PROTECTED_SURFACE_IDENTITIES.items():
        path = overrides.get(relative_path, repository / relative_path) if overrides else repository / relative_path
        actual_sha256 = control._sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise control.BrightQrelsRemediationError(f"protected publication surface identity drift: {relative_path}")
        records.append(f"{actual_sha256}  {relative_path}\n")
    if hashlib.sha256("".join(records).encode("utf-8")).hexdigest() != PROTECTED_SURFACES_SHA256:
        raise control.BrightQrelsRemediationError("protected publication surface aggregate identity drift")


def _restricted_bundle_root(
    repository: Path,
    control_value: dict[str, Any],
    *,
    create: bool,
    require_files: bool,
) -> Path:
    declared = control_value["proposal_constraints"]["restricted_root"]
    if declared != RESTRICTED_ROOT.as_posix():
        raise control.BrightQrelsRemediationError("candidate restricted root differs from the frozen lexical root")
    root = control._restricted_root_path(repository, declared, create=create)
    for name in RESTRICTED_BUNDLE_NAMES:
        control._assert_restricted_file(
            repository,
            declared,
            root / name,
            expected_name=name,
            require_exists=require_files,
        )
    return root


def _effective_units(
    pack: tuple[dict[str, Any], ...],
    primary: tuple[dict[str, Any], ...],
    adjudications: tuple[dict[str, Any], ...],
) -> dict[tuple[str, str, str, str | None], dict[str, Any]]:
    primary_by_query = {(row["track"], row["query_id"]): row for row in primary}
    overrides = {
        (row["unit_type"], row["track"], row["query_id"], row.get("document_id")): row for row in adjudications
    }
    result = {}
    for source_row in pack:
        track = source_row["track"]
        query_id = source_row["query_id"]
        annotation = primary_by_query[(track, query_id)]
        units = [("query", None, annotation["query_review"])]
        units.extend(("gold", item["document_id"], item) for item in annotation["gold_reviews"])
        units.extend(("candidate", item["document_id"], item) for item in annotation["candidate_reviews"])
        for unit_type, document_id, primary_judgment in units:
            key = (unit_type, track, query_id, document_id)
            adjudication = overrides.get(key)
            result[key] = {
                "primary_judgment": primary_judgment,
                "adjudicated_judgment": adjudication["adjudicated_judgment"] if adjudication else None,
                "effective_judgment": (adjudication["adjudicated_judgment"] if adjudication else primary_judgment),
                "input_row_sha256": annotation["input_row_sha256"],
                "adjudicated": adjudication is not None,
            }
    return result


def _provenance(
    input_row_sha256: str,
    source_annotation_sha256: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "judgment_origin": "agent_judged",
        "agent_role": REVIEW_ROLE,
        "session_id": REVIEW_SESSION,
        "round": 1,
        "judged_at": REVIEW_TIME,
        "rubric_sha256": audit["rubric_sha256"],
        "input_artifact_sha256": audit["audit_pack"]["sha256"],
        "input_row_sha256": input_row_sha256,
        "source_annotation_sha256": source_annotation_sha256,
    }


def _remediation_for_unit(
    key: tuple[str, str, str, str | None],
    source: dict[str, str],
    accepted: dict[str, Any],
    official: dict[tuple[str, str, str], int],
    audit: dict[str, Any],
) -> dict[str, Any]:
    unit_type, track, query_id, document_id = key
    status = source["source_judgment_status"]

    if unit_type == "query":
        taxonomy = ["task_semantics_mismatch"]
        if accepted.get("credible_missing_positive") == "yes":
            taxonomy.append("credible_missing_positive")
        elif accepted.get("qrels_completeness") == "incomplete":
            taxonomy.append("candidate_pool_coverage_gap")
        if status == "uncertain" or accepted.get("qrels_completeness") == "uncertain":
            taxonomy.append("insufficient_existing_evidence")
        disposition = "stop_query_task_semantics_mismatch"
        evidence_level = "conflicting"
        rationale = (
            "The local source task labels author-curated linked passages, while the accepted audit evaluates direct "
            "answer support; the frozen local contracts do not define a valid mapping between those semantics."
        )
    elif unit_type == "gold":
        taxonomy = []
        if accepted.get("ambiguity") == "ambiguous":
            taxonomy.append("gold_ambiguity")
        if accepted.get("support") in {"partially_supports", "does_not_support"}:
            taxonomy.append("gold_support_defect")
        if status == "uncertain":
            taxonomy.append("insufficient_existing_evidence")
        if not taxonomy:
            taxonomy = ["no_qrels_defect"]
        if status in {"ambiguous", "uncertain"}:
            disposition = "downgrade_to_uncertain"
            rationale = (
                "The accepted source judgment is not determinate, and unresolved task semantics prohibit a positive "
                "membership decision or a synthesized negative."
            )
        else:
            disposition = "quarantine_official_positive"
            rationale = (
                "The item is quarantined only from this inactive reviewed-scope proposal because a direct-support "
                "audit cannot negate linked-source relevance under unresolved task semantics."
            )
        if accepted.get("ambiguity") == "ambiguous":
            evidence_level = "conflicting"
        elif accepted.get("support") == "supports":
            evidence_level = "document_level_support"
        elif accepted.get("support") == "partially_supports":
            evidence_level = "contextual_or_topical_only"
        else:
            evidence_level = "insufficient"
    else:
        suspected = accepted.get("suspected_missing_positive")
        taxonomy = ["credible_missing_positive"] if suspected == "credible" else ["no_qrels_defect"]
        official_key = (track, query_id, document_id)
        if official_key in official:
            disposition = "not_applicable"
            rationale = (
                "The candidate is an alternate view of an official positive; its accepted candidate assessment is "
                "preserved without creating a duplicate membership decision."
            )
        elif status == "unjudged":
            disposition = "no_change_unjudged"
            rationale = (
                "The accepted audit assessment is preserved as evidence, but the candidate remains unjudged for "
                "official qrels membership and is not converted to either a positive or a negative."
            )
        else:
            disposition = "downgrade_to_uncertain"
            taxonomy = ["insufficient_existing_evidence"]
            rationale = "Accepted ambiguity or uncertainty is preserved without creating a qrels membership label."
        evidence_level = (
            "document_level_support"
            if suspected == "credible" or accepted.get("relevance") == "relevant"
            else "contextual_or_topical_only"
        )

    return {
        "unit_type": unit_type,
        "track": track,
        "query_id": query_id,
        "document_id": document_id,
        "source_judgment_status": status,
        "taxonomy": taxonomy,
        "disposition": disposition,
        "evidence_level": evidence_level,
        "supporting_spans": [],
        "rationale": rationale,
        "provenance": _provenance(
            source["input_row_sha256"],
            source["source_annotation_sha256"],
            audit,
        ),
    }


def _query_state(items: list[dict[str, Any]]) -> str:
    if any(
        "task_semantics_mismatch" in item["taxonomy"] or item["disposition"] == "stop_query_task_semantics_mismatch"
        for item in items
    ):
        return "task_semantics_mismatch"
    if any(
        "candidate_pool_coverage_gap" in item["taxonomy"]
        or "insufficient_existing_evidence" in item["taxonomy"]
        or item["source_judgment_status"] in {"ambiguous", "uncertain"}
        or item["evidence_level"] in {"conflicting", "insufficient"}
        or item["disposition"]
        in {"stop_query_insufficient_existing_evidence", "downgrade_to_uncertain", "abstain_uncertain"}
        for item in items
    ):
        return "insufficient_existing_evidence"
    return "remediable_with_existing_data"


def _query_risks(
    dispositions: list[dict[str, Any]],
    diff_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dispositions:
        grouped[(row["track"], row["query_id"])].append(row)
    diff_by_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in diff_rows:
        diff_by_query[(row["track"], row["query_id"])].append(row)
    rows = []
    for (track, query_id), items in sorted(grouped.items()):
        units = Counter(item["unit_type"] for item in items)
        source_status = Counter(item["source_judgment_status"] for item in items)
        disposition_counts = Counter(item["disposition"] for item in items)
        query_diff = diff_by_query[(track, query_id)]
        rows.append(
            {
                "schema_version": "1",
                "proposal_id": control.PROPOSAL_ID,
                "track": track,
                "query_id": query_id,
                "coverage": {
                    "query": units["query"],
                    "gold": units["gold"],
                    "candidate": units["candidate"],
                    "complete": units["query"] == 1,
                },
                "source_judgment_status": dict(sorted(source_status.items())),
                "dispositions": dict(sorted(disposition_counts.items())),
                "membership_diff": {
                    "retained_positive": sum(row["operation"] == "retain" for row in query_diff),
                    "added_positive": sum(row["operation"] == "add" for row in query_diff),
                    "quarantined_or_uncertain_positive": sum(row["operation"] == "quarantine" for row in query_diff),
                    "negative_synthesized": sum(row["negative_synthesized"] for row in query_diff),
                },
                "verdict": _query_state(items),
            }
        )
    return rows


def _diff_rows(dispositions: list[dict[str, Any]], official: dict[tuple[str, str, str], int]) -> list[dict[str, Any]]:
    rows = []
    for item in dispositions:
        key = (item["track"], item["query_id"], item["document_id"])
        if item["unit_type"] == "gold":
            retained = item["disposition"] == "retain_official_positive"
            original_grade: int | None = official[key]
            proposed_grade: int | None = official[key] if retained else None
            operation = "retain" if retained else "quarantine"
        elif item["unit_type"] == "candidate" and item["disposition"] == "propose_add_positive":
            original_grade = None
            proposed_grade = 1
            operation = "add"
        else:
            continue
        rows.append(
            {
                "schema_version": "1",
                "proposal_id": control.PROPOSAL_ID,
                "track": item["track"],
                "query_id": item["query_id"],
                "document_id": item["document_id"],
                "original_grade": original_grade,
                "proposed_grade": proposed_grade,
                "operation": operation,
                "disposition": item["disposition"],
                "negative_synthesized": False,
            }
        )
    return sorted(rows, key=lambda row: (row["track"], row["query_id"], row["document_id"]))


def _evidence_ledger(
    dispositions: list[dict[str, Any]],
    effective: dict[tuple[str, str, str, str | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in dispositions:
        key = control._unit_key(item)
        source = effective[key]
        rows.append(
            {
                "schema_version": "1",
                "proposal_id": control.PROPOSAL_ID,
                "unit_type": item["unit_type"],
                "track": item["track"],
                "query_id": item["query_id"],
                "document_id": item["document_id"],
                "primary_judgment": source["primary_judgment"],
                "adjudicated_judgment": source["adjudicated_judgment"],
                "effective_judgment": source["effective_judgment"],
                "primary_judgment_sha256": _row_sha256(source["primary_judgment"]),
                "effective_judgment_sha256": _row_sha256(source["effective_judgment"]),
                "review_basis": "frozen_task_contract_and_accepted_audit_only",
                "remediation": item,
            }
        )
    return rows


def _count_nested(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def _taxonomy_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(value for row in rows for value in row["taxonomy"]).items()))


def _accepted_evidence(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = {
        unit_type: [row for row in ledger if row["unit_type"] == unit_type]
        for unit_type in ("query", "gold", "candidate")
    }

    def counts(unit_type: str, field: str) -> dict[str, int]:
        return dict(
            sorted(Counter(row["effective_judgment"].get(field, "missing") for row in by_type[unit_type]).items())
        )

    def counts_by_track(unit_type: str, field: str) -> dict[str, dict[str, int]]:
        return {
            track: dict(
                sorted(
                    Counter(
                        row["effective_judgment"].get(field, "missing")
                        for row in by_type[unit_type]
                        if row["track"] == track
                    ).items()
                )
            )
            for track in control.TRACKS
        }

    unresolved_by_track = Counter(
        row["track"]
        for row in ledger
        if row["unit_type"] == "query" and row["effective_judgment"].get("decision_status") != "decided"
    )
    credible_queries_by_track = Counter(
        row["track"]
        for row in ledger
        if row["unit_type"] == "query" and row["effective_judgment"].get("credible_missing_positive") == "yes"
    )
    credible_candidates_by_track = Counter(
        row["track"]
        for row in ledger
        if row["unit_type"] == "candidate" and row["effective_judgment"].get("suspected_missing_positive") == "credible"
    )
    return {
        "queries": {
            "decision_status": counts("query", "decision_status"),
            "qrels_completeness": counts("query", "qrels_completeness"),
            "credible_missing_positive": counts("query", "credible_missing_positive"),
            "credible_missing_positive_queries_by_track": {
                track: credible_queries_by_track[track] for track in control.TRACKS
            },
        },
        "gold": {
            "support": counts("gold", "support"),
            "support_by_track": counts_by_track("gold", "support"),
            "ambiguity": counts("gold", "ambiguity"),
            "ambiguity_by_track": counts_by_track("gold", "ambiguity"),
        },
        "candidates": {
            "relevance": counts("candidate", "relevance"),
            "relevance_by_track": counts_by_track("candidate", "relevance"),
            "suspected_missing_positive": counts("candidate", "suspected_missing_positive"),
            "suspected_missing_positive_by_track": counts_by_track("candidate", "suspected_missing_positive"),
            "credible_missing_positive_by_track": {
                track: credible_candidates_by_track[track] for track in control.TRACKS
            },
        },
        "unresolved_queries": {
            "total": sum(unresolved_by_track.values()),
            "by_track": {track: unresolved_by_track[track] for track in control.TRACKS},
        },
        "adjudicated_units": sum(row["adjudicated_judgment"] is not None for row in ledger),
    }


def _safe_aggregate(
    proposal_sha256: str,
    manifest_sha256: str,
    artifacts: dict[str, dict[str, Any]],
    proposal: dict[str, Any],
    ledger: list[dict[str, Any]],
    query_risks: list[dict[str, Any]],
    diff_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    dispositions = proposal["dispositions"]
    conclusion = proposal["conclusion"]
    tracks = {}
    for track in control.TRACKS:
        track_dispositions = [row for row in dispositions if row["track"] == track]
        track_risks = [row for row in query_risks if row["track"] == track]
        unit_counts = Counter(row["unit_type"] for row in track_dispositions)
        stopped = sum(
            row["unit_type"] == "query" and row["disposition"].startswith("stop_query_") for row in track_dispositions
        )
        verdict = conclusion["tracks"][track]
        tracks[track] = {
            "coverage": {
                "queries": unit_counts["query"],
                "gold_items": unit_counts["gold"],
                "candidate_items": unit_counts["candidate"],
            },
            "dispositions": _count_nested(track_dispositions, "disposition"),
            "taxonomy": _taxonomy_counts(track_dispositions),
            "reviewed_queries_eligible": len(track_risks) - stopped,
            "reviewed_queries_stopped": stopped,
            "verdict": verdict,
            "retainable_scope": (
                "none_of_the_reviewed_queries_until_task_semantics_are_resolved"
                if verdict == "task_semantics_mismatch"
                else "only_queries_without_a_frozen_stop_condition"
            ),
            "unsupported_claims": [
                "corrected_relevance_gold",
                "full_track_repair",
                "model_quality_recomparison",
                "public_leaderboard",
            ],
            "existing_evidence_gap": (
                "No frozen local contract determines whether linked-source relevance or direct answer support governs "
                "qrels membership."
            ),
            "next_minimum_action": (
                "Freeze one task-semantic definition, then independently review the same 60 frozen queries under that "
                "definition before any qrels activation."
            ),
        }
    return {
        "schema_version": "1",
        "candidate_id": CANDIDATE_ID,
        "repair_round": REPAIR_ROUND,
        "story_id": "S-20260906-001",
        "control_sha256": proposal["control_sha256"],
        "control_revision": {
            "changed_at": CONTROL_REPAIR_TIME,
            "reason": "Repair round 2 adds explicit supporting-span offsets without changing business rules.",
            "previous_control_sha256": PRE_REPAIR_CONTROL_SHA256,
            "current_control_sha256": proposal["control_sha256"],
            "previous_proposal_schema_sha256": PRE_REPAIR_PROPOSAL_SCHEMA_SHA256,
            "current_proposal_schema_sha256": control._sha256_file(
                control._schema_path("bright-qrels-remediation-proposal-v01.schema.json")
            ),
            "rules_sha256": control._sha256_json(control.FROZEN_RULES),
        },
        "rules_sha256": "4dacff181fff81caa87aa7ee40c7c165fe05e227436d53f97410bb6837bbb52c",
        "protected_surfaces_sha256": PROTECTED_SURFACES_SHA256,
        "restricted_artifacts": {
            "tracked": False,
            "proposal_sha256": proposal_sha256,
            "manifest_sha256": manifest_sha256,
            "files": {
                name: {key: value[key] for key in ("rows", "bytes", "sha256")} for name, value in artifacts.items()
            },
        },
        "coverage": {
            "queries": sum(row["coverage"]["query"] for row in query_risks),
            "gold_items": sum(row["coverage"]["gold"] for row in query_risks),
            "candidate_items": sum(row["coverage"]["candidate"] for row in query_risks),
            "tracks": {track: sum(row["track"] == track for row in query_risks) for track in control.TRACKS},
        },
        "dispositions": {
            "total": len(dispositions),
            "by_unit_type": _count_nested(dispositions, "unit_type"),
            "by_disposition": _count_nested(dispositions, "disposition"),
            "by_evidence_level": _count_nested(dispositions, "evidence_level"),
            "by_taxonomy": _taxonomy_counts(dispositions),
        },
        "diff": {
            "original_positive_rows_in_reviewed_scope": sum(row["original_grade"] is not None for row in diff_rows),
            "proposed_positive_rows": sum(row["proposed_grade"] is not None for row in diff_rows),
            "retained_positive_rows": sum(row["operation"] == "retain" for row in diff_rows),
            "added_positive_rows": sum(row["operation"] == "add" for row in diff_rows),
            "quarantined_or_uncertain_positive_rows": sum(row["operation"] == "quarantine" for row in diff_rows),
            "negative_rows": sum(row["negative_synthesized"] for row in diff_rows),
        },
        "accepted_evidence": _accepted_evidence(ledger),
        "tracks": tracks,
        "sensitivity_boundaries": [
            {
                "condition": "current_frozen_local_contracts",
                "verdict": "task_semantics_mismatch",
                "reason": "linked-source relevance and direct answer support are not locally mapped",
            },
            {
                "condition": "linked_source_relevance_is_explicitly_adopted",
                "verdict": "insufficient_existing_evidence",
                "reason": "the accepted support audit cannot validate or negate linked-source relevance",
            },
            {
                "condition": "direct_answer_support_is_explicitly_adopted",
                "verdict": "insufficient_existing_evidence",
                "reason": "membership changes still require independent item review and direct-span evidence",
            },
        ],
        "score_independence": {
            "model_scores_consumed": False,
            "rank_positions_consumed": False,
            "method_outcomes_consumed": False,
            "rankings_used_for_candidate_location": False,
        },
        "verdict": {**conclusion["tracks"], "overall": conclusion["overall"]},
        "s007_candidate_input": {
            "eligible": conclusion["overall"] == "insufficient_existing_evidence",
            "reason": (
                "The derived overall verdict is insufficient_existing_evidence."
                if conclusion["overall"] == "insufficient_existing_evidence"
                else "The derived overall verdict is not insufficient_existing_evidence."
            ),
            "story_modified_or_promoted": False,
        },
        "publication": {
            "status": "inactive",
            "classification": "restricted_no_publish",
            "default_activation": False,
            "registry_integration": False,
            "runner_integration": False,
            "public_export_allowed": False,
            "leaderboard_allowed": False,
            "publish_allowed": False,
            "official_qrels_modified": False,
            "accepted_evidence_modified": False,
        },
        "safe_content": {
            "contains_query_or_document_text": False,
            "contains_canonical_ids": False,
            "contains_rankings": False,
            "contains_supporting_spans": False,
            "contains_proposed_qrels_rows": False,
            "contains_reviewer_identity": False,
            "contains_absolute_private_paths": False,
            "contains_reversible_mapping": False,
        },
    }


def _report(aggregate: dict[str, Any]) -> str:
    economics = aggregate["tracks"]["economics"]
    psychology = aggregate["tracks"]["psychology"]
    evidence = aggregate["accepted_evidence"]
    gold_support = evidence["gold"]["support"]
    gold_ambiguity = evidence["gold"]["ambiguity"]
    credible_candidates = evidence["candidates"]["credible_missing_positive_by_track"]
    credible_queries = evidence["queries"]["credible_missing_positive_queries_by_track"]
    return f"""# BRIGHT Qrels Remediation Candidate v0.1

## Status

This is a research-only, restricted, inactive, default-deny candidate for Story
`S-20260906-001`, repair round `{aggregate["repair_round"]}`. It does not modify official qrels, accepted evidence,
registries, runners, exporters, leaderboards, or publication configuration.

The restricted proposal is bound by SHA-256 `{aggregate["restricted_artifacts"]["proposal_sha256"]}`.
It contains complete row-level evidence and provenance but remains gitignored
and no-publish. This tracked report contains no source text, canonical identity,
ranking, supporting span, proposed qrels row, reviewer identity, private path,
or reversible mapping.

## Control contract repair history

Repair round 2 changed only the supporting-span structural contract. At
`{aggregate["control_revision"]["changed_at"]}`, the proposal schema identity
changed from `{aggregate["control_revision"]["previous_proposal_schema_sha256"]}`
to `{aggregate["control_revision"]["current_proposal_schema_sha256"]}` by adding
mandatory `start_offset` and `end_offset`. The control identity consequently
changed from `{aggregate["control_revision"]["previous_control_sha256"]}` to
`{aggregate["control_revision"]["current_control_sha256"]}`. Business rules
remain byte-identical under rules identity
`{aggregate["control_revision"]["rules_sha256"]}`.

## Coverage and disposition

- Queries: {aggregate["coverage"]["queries"]} / 120
- Gold items: {aggregate["coverage"]["gold_items"]} / 1,235
- Frozen candidates: {aggregate["coverage"]["candidate_items"]} / 2,804
- Total dispositions: {aggregate["dispositions"]["total"]}
- Original positive rows in reviewed scope: {aggregate["diff"]["original_positive_rows_in_reviewed_scope"]}
- Proposed positive rows: {aggregate["diff"]["proposed_positive_rows"]}
- Quarantined or uncertain positive rows: {aggregate["diff"]["quarantined_or_uncertain_positive_rows"]}
- Added positives: {aggregate["diff"]["added_positive_rows"]}
- Synthesized negatives: {aggregate["diff"]["negative_rows"]}

Accepted item-level evidence is preserved independently from the contract-level
semantic stop:

- Gold support: {gold_support.get("supports", 0)} supports,
  {gold_support.get("partially_supports", 0)} partially supports, and
  {gold_support.get("does_not_support", 0)} does not support.
- Economics gold support: {evidence["gold"]["support_by_track"]["economics"].get("supports", 0)} /
  {evidence["gold"]["support_by_track"]["economics"].get("partially_supports", 0)} /
  {evidence["gold"]["support_by_track"]["economics"].get("does_not_support", 0)}.
- Psychology gold support: {evidence["gold"]["support_by_track"]["psychology"].get("supports", 0)} /
  {evidence["gold"]["support_by_track"]["psychology"].get("partially_supports", 0)} /
  {evidence["gold"]["support_by_track"]["psychology"].get("does_not_support", 0)}.
- Gold ambiguity: {gold_ambiguity.get("ambiguous", 0)} ambiguous and
  {gold_ambiguity.get("unambiguous", 0)} unambiguous.
- Credible missing-positive candidates: economics {credible_candidates["economics"]}
  across {credible_queries["economics"]} queries; psychology {credible_candidates["psychology"]}
  across {credible_queries["psychology"]} queries.
- Unresolved accepted query judgments: {evidence["unresolved_queries"]["total"]}
  total, economics {evidence["unresolved_queries"]["by_track"]["economics"]},
  psychology {evidence["unresolved_queries"]["by_track"]["psychology"]}.
- Accepted adjudications represented in the evidence ledger: {evidence["adjudicated_units"]}.

Quarantine means omission from this inactive reviewed-scope candidate. It is not
a relevance-zero judgment and does not alter official qrels.

## Three-state verdict

- Economics: `{economics["verdict"]}`; {economics["reviewed_queries_stopped"]} reviewed queries stopped.
- Psychology: `{psychology["verdict"]}`; {psychology["reviewed_queries_stopped"]} reviewed queries stopped.
- Overall: `{aggregate["verdict"]["overall"]}`.

The source task defines positives as author-curated linked passages associated
with naturally occurring questions. The accepted audit assessed whether a
passage directly supports the central information need or answer. Frozen local
materials do not specify how those standards map. Therefore support failure is
not converted into non-relevance, and accepted candidate relevance is not
converted into a positive qrels row.

No reviewed query is eligible for a bounded repaired track until the task
semantics are resolved. The candidate cannot support corrected relevance gold,
a full-track repair, model-quality recomparison, or a public leaderboard.

## Score-independent sensitivity boundary

- Under the current contracts, the result remains `task_semantics_mismatch`.
- If linked-source relevance is explicitly adopted, existing support judgments
  are insufficient to validate or negate those labels.
- If direct answer support is explicitly adopted, independent item review and
  direct-span evidence are still required for membership changes.

No model score, rank position, method outcome, or ranking file was consumed by
candidate generation.

## Next minimum action

Freeze one benchmark task definition: linked-source/citation relevance, direct
answer-support relevance, or explicitly separate tracks. Then independently
review the same frozen 60-query sample per track under that definition before
activating any qrels. This is a product/research semantics decision because it
changes benchmark identity and supported claims.

The result is not eligible as an input gate for `S-20260814-007`: the frozen
overall state is not `insufficient_existing_evidence`, and that Story was not
modified or promoted.
"""


def _validate_safe_output_privacy(
    aggregate: dict[str, Any],
    report: str,
    pack: tuple[dict[str, Any], ...],
) -> None:
    payload = _json_bytes(aggregate).decode("utf-8") + report
    if "/data" in payload or "/home/" in payload or REVIEW_SESSION in payload:
        raise control.BrightQrelsRemediationError("tracked candidate contains a private path or reviewer identity")
    identifiers = set()
    source_texts = set()
    for row in pack:
        identifiers.add(row["query_id"])
        source_texts.add(row["query"])
        for item in row["gold_documents"]:
            identifiers.add(item["document_id"])
            source_texts.add(item["content"])
        for item in row["baseline_top10_union"]:
            identifiers.add(item["document_id"])
            source_texts.add(item["content"])
    if any(identifier in payload for identifier in identifiers if len(identifier) >= 8):
        raise control.BrightQrelsRemediationError("tracked candidate contains a canonical identity")
    if any(text in payload for text in source_texts if len(text) >= 64):
        raise control.BrightQrelsRemediationError("tracked candidate contains restricted source text")


def _validate_restricted_components(
    proposal: dict[str, Any],
    diff_rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    query_risks: list[dict[str, Any]],
) -> None:
    for row in diff_rows:
        control._validate_schema(row, "bright-qrels-remediation-diff-v01.schema.json", "remediation diff row")
    for row in ledger:
        control._validate_schema(
            row,
            "bright-qrels-remediation-evidence-ledger-v01.schema.json",
            "remediation evidence-ledger row",
        )
    for row in query_risks:
        control._validate_schema(row, "bright-qrels-remediation-query-risk-v01.schema.json", "query-risk row")
    if len(proposal["dispositions"]) != len(ledger):
        raise control.BrightQrelsRemediationError("evidence-ledger coverage mismatch")
    proposal_by_key = {control._unit_key(row): row for row in proposal["dispositions"]}
    ledger_by_key = {control._unit_key(row["remediation"]): row for row in ledger}
    if set(proposal_by_key) != set(ledger_by_key):
        raise control.BrightQrelsRemediationError("evidence-ledger identity mismatch")
    if any(ledger_by_key[key]["remediation"] != row for key, row in proposal_by_key.items()):
        raise control.BrightQrelsRemediationError("evidence-ledger disposition mismatch")
    expected_diff_rows = sum(row["unit_type"] == "gold" for row in proposal["dispositions"]) + sum(
        row["unit_type"] == "candidate" and row["disposition"] == "propose_add_positive"
        for row in proposal["dispositions"]
    )
    if len(diff_rows) != expected_diff_rows:
        raise control.BrightQrelsRemediationError("original-vs-proposed diff coverage mismatch")
    if any(row["negative_synthesized"] for row in diff_rows):
        raise control.BrightQrelsRemediationError("diff contains a synthesized negative grade")
    if len(query_risks) != control.EXPECTED_COVERAGE["queries"]:
        raise control.BrightQrelsRemediationError("query-risk coverage mismatch")
    risk_keys = [(row["track"], row["query_id"]) for row in query_risks]
    expected_query_keys = {
        (row["track"], row["query_id"]) for row in proposal["dispositions"] if row["unit_type"] == "query"
    }
    if len(risk_keys) != len(set(risk_keys)) or set(risk_keys) != expected_query_keys:
        raise control.BrightQrelsRemediationError("query-risk identity set is not exact and unique")


def _expected_proposal(
    control_summary: dict[str, Any],
    control_value: dict[str, Any],
    inventory: dict[tuple[str, str, str, str | None], dict[str, str]],
    effective: dict[tuple[str, str, str, str | None], dict[str, Any]],
    official: dict[tuple[str, str, str], int],
    canonical_documents: dict[tuple[str, str, str], str],
) -> dict[str, Any]:
    dispositions = [
        _remediation_for_unit(
            key,
            source,
            effective[key]["effective_judgment"],
            official,
            control_value["source_identities"]["audit"],
        )
        for key, source in inventory.items()
    ]
    proposal = {
        "schema_version": "1",
        "proposal_id": control.PROPOSAL_ID,
        "version": "0.1",
        "control_sha256": control_summary["control_sha256"],
        "status": "inactive",
        "activation": {
            "default_activation": False,
            "registry_integration": False,
            "runner_integration": False,
        },
        "publication": {
            "classification": "restricted_no_publish",
            "public_export_allowed": False,
            "leaderboard_allowed": False,
            "publish_allowed": False,
        },
        "source_identities": control_value["source_identities"],
        "dispositions": dispositions,
        "proposed_qrels_rows": [],
        "conclusion": control._derive_conclusion(dispositions),
    }
    control._validate_proposal_payload(
        proposal,
        control_value,
        inventory,
        official,
        canonical_documents,
    )
    return proposal


def _expected_manifest(
    control_summary: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    conclusion: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "candidate_id": CANDIDATE_ID,
        "repair_round": REPAIR_ROUND,
        "story_id": "S-20260906-001",
        "control_sha256": control_summary["control_sha256"],
        "rules_sha256": control_summary["rules_sha256"],
        "status": "inactive",
        "classification": "restricted_no_publish",
        "artifacts": artifacts,
        "schema_identities": {name: control._sha256_file(control._schema_path(name)) for name in SCHEMA_NAMES},
        "coverage": control.EXPECTED_COVERAGE,
        "conclusion": conclusion,
        "official_qrels_modified": False,
        "accepted_evidence_modified": False,
        "default_activation": False,
        "registry_integration": False,
        "runner_integration": False,
        "public_export_allowed": False,
        "leaderboard_allowed": False,
        "publish_allowed": False,
    }


def build_candidate(repository: str | Path | None = None) -> dict[str, Any]:
    """Build the full candidate from frozen local evidence only."""
    repo = Path(repository) if repository is not None else control._root()
    control_summary = control.validate_control(repository=repo)
    _validate_protected_surfaces(repo)
    control_value = control._read_object(repo / control.DEFAULT_PATHS["control"], "remediation control")
    paths = control._resolve_paths(repo)
    pack = control._read_jsonl(paths["audit_pack"], "frozen audit pack")
    primary = control._read_jsonl(paths["primary_annotations"], "accepted primary annotations")
    adjudications = control._read_jsonl(paths["adjudications"], "accepted adjudications")
    official = control._official_qrels(control._read_jsonl(paths["economics_qrels"], "economics qrels"), "economics")
    official.update(
        control._official_qrels(control._read_jsonl(paths["psychology_qrels"], "psychology qrels"), "psychology")
    )
    inventory = control._source_inventory(
        pack,
        primary,
        adjudications,
        official,
        control_value["source_identities"]["audit"],
    )
    effective = _effective_units(pack, primary, adjudications)
    if set(inventory) != set(effective):
        raise control.BrightQrelsRemediationError("effective audit inventory mismatch")
    canonical_documents = control._canonical_documents(pack)
    proposal = _expected_proposal(
        control_summary,
        control_value,
        inventory,
        effective,
        official,
        canonical_documents,
    )
    dispositions = proposal["dispositions"]
    diff_rows = _diff_rows(dispositions, official)
    ledger = _evidence_ledger(dispositions, effective)
    query_risks = _query_risks(dispositions, diff_rows)
    _validate_restricted_components(proposal, diff_rows, ledger, query_risks)

    restricted = _restricted_bundle_root(
        repo,
        control_value,
        create=True,
        require_files=False,
    )
    artifacts = {
        "proposal.json": _write_json(restricted / "proposal.json", proposal),
        "original-vs-proposed.jsonl": _write_jsonl(restricted / "original-vs-proposed.jsonl", diff_rows),
        "evidence-ledger.jsonl": _write_jsonl(restricted / "evidence-ledger.jsonl", ledger),
        "query-risk.jsonl": _write_jsonl(restricted / "query-risk.jsonl", query_risks),
    }
    for name, identity in artifacts.items():
        _write_sidecar(restricted / name, identity["sha256"])
    manifest = _expected_manifest(control_summary, artifacts, proposal["conclusion"])
    control._validate_schema(
        manifest,
        "bright-qrels-remediation-restricted-manifest-v01.schema.json",
        "restricted remediation manifest",
    )
    manifest_identity = _write_json(restricted / "manifest.json", manifest)
    _write_sidecar(restricted / "manifest.json", manifest_identity["sha256"])

    aggregate = _safe_aggregate(
        artifacts["proposal.json"]["sha256"],
        manifest_identity["sha256"],
        artifacts,
        proposal,
        ledger,
        query_risks,
        diff_rows,
    )
    report = _report(aggregate)
    _validate_safe_output_privacy(aggregate, report, pack)
    control._validate_schema(
        aggregate,
        "bright-qrels-remediation-safe-aggregate-v01.schema.json",
        "remediation safe aggregate",
    )
    aggregate_path = repo / TRACKED_ROOT / "candidate-aggregate.json"
    aggregate_identity = _write_json(aggregate_path, aggregate)
    _write_sidecar(aggregate_path, aggregate_identity["sha256"])
    report_path = repo / REPORT_PATH
    report_identity = _write_bytes(report_path, report.encode("utf-8"))
    _write_sidecar(report_path, report_identity["sha256"])
    return {
        "candidate_id": CANDIDATE_ID,
        "proposal_sha256": artifacts["proposal.json"]["sha256"],
        "manifest_sha256": manifest_identity["sha256"],
        "aggregate_sha256": aggregate_identity["sha256"],
        "report_sha256": report_identity["sha256"],
        "coverage": control.EXPECTED_COVERAGE,
        "dispositions": len(dispositions),
        "diff_rows": len(diff_rows),
        "verdict": proposal["conclusion"],
    }


def validate_candidate(repository: str | Path | None = None) -> dict[str, Any]:
    """Recompute the candidate from immutable inputs and compare every artifact byte."""
    repo = Path(repository) if repository is not None else control._root()
    _validate_protected_surfaces(repo)
    control_summary = control.validate_control(repository=repo)
    control_value = control._read_object(repo / control.DEFAULT_PATHS["control"], "remediation control")
    paths = control._resolve_paths(repo)
    pack = control._read_jsonl(paths["audit_pack"], "frozen audit pack")
    primary = control._read_jsonl(paths["primary_annotations"], "accepted primary annotations")
    adjudications = control._read_jsonl(paths["adjudications"], "accepted adjudications")
    official = control._official_qrels(
        control._read_jsonl(paths["economics_qrels"], "economics qrels"),
        "economics",
    )
    official.update(
        control._official_qrels(
            control._read_jsonl(paths["psychology_qrels"], "psychology qrels"),
            "psychology",
        )
    )
    inventory = control._source_inventory(
        pack,
        primary,
        adjudications,
        official,
        control_value["source_identities"]["audit"],
    )
    effective = _effective_units(pack, primary, adjudications)
    if set(inventory) != set(effective):
        raise control.BrightQrelsRemediationError("effective audit inventory mismatch")
    canonical_documents = control._canonical_documents(pack)

    restricted = _restricted_bundle_root(
        repo,
        control_value,
        create=False,
        require_files=True,
    )
    proposal_path = restricted / "proposal.json"
    proposal_result = control.validate_proposal(proposal_path, repository=repo)
    proposal = control._read_object(proposal_path, "restricted remediation proposal")
    expected_proposal = _expected_proposal(
        control_summary,
        control_value,
        inventory,
        effective,
        official,
        canonical_documents,
    )
    if proposal_path.read_bytes() != _json_bytes(expected_proposal):
        raise control.BrightQrelsRemediationError("proposal deterministic derivation mismatch")

    expected_diff = _diff_rows(proposal["dispositions"], official)
    expected_ledger = _evidence_ledger(proposal["dispositions"], effective)
    expected_query_risks = _query_risks(proposal["dispositions"], expected_diff)
    expected_payloads = {
        "proposal.json": _json_bytes(expected_proposal),
        "original-vs-proposed.jsonl": _jsonl_bytes(expected_diff),
        "evidence-ledger.jsonl": _jsonl_bytes(expected_ledger),
        "query-risk.jsonl": _jsonl_bytes(expected_query_risks),
    }
    for name, expected_bytes in expected_payloads.items():
        path = restricted / name
        if path.read_bytes() != expected_bytes:
            raise control.BrightQrelsRemediationError(f"{name} deterministic recomputation mismatch")

    diff_rows = list(control._read_jsonl(restricted / "original-vs-proposed.jsonl", "remediation diff"))
    ledger = list(control._read_jsonl(restricted / "evidence-ledger.jsonl", "evidence ledger"))
    query_risks = list(control._read_jsonl(restricted / "query-risk.jsonl", "query risk"))
    _validate_restricted_components(proposal, diff_rows, ledger, query_risks)

    artifacts = {}
    for name in expected_payloads:
        path = restricted / name
        actual = {
            "rows": len(control._read_jsonl(path, name)) if name.endswith(".jsonl") else 1,
            "bytes": path.stat().st_size,
            "sha256": control._sha256_file(path),
        }
        if path.with_suffix(path.suffix + ".sha256").read_text(encoding="utf-8").strip() != actual["sha256"]:
            raise control.BrightQrelsRemediationError(f"restricted artifact sidecar drift: {name}")
        artifacts[name] = actual

    manifest_path = restricted / "manifest.json"
    expected_manifest = _expected_manifest(control_summary, artifacts, proposal["conclusion"])
    if manifest_path.read_bytes() != _json_bytes(expected_manifest):
        raise control.BrightQrelsRemediationError("restricted manifest deterministic recomputation mismatch")
    manifest = control._read_object(manifest_path, "restricted remediation manifest")
    control._validate_schema(
        manifest,
        "bright-qrels-remediation-restricted-manifest-v01.schema.json",
        "restricted remediation manifest",
    )
    manifest_sha256 = control._sha256_file(restricted / "manifest.json")
    if (restricted / "manifest.json.sha256").read_text(encoding="utf-8").strip() != manifest_sha256:
        raise control.BrightQrelsRemediationError("restricted manifest sidecar drift")

    aggregate_path = repo / TRACKED_ROOT / "candidate-aggregate.json"
    aggregate = control._read_object(aggregate_path, "remediation safe aggregate")
    expected_aggregate = _safe_aggregate(
        artifacts["proposal.json"]["sha256"],
        manifest_sha256,
        artifacts,
        proposal,
        ledger,
        query_risks,
        diff_rows,
    )
    if aggregate_path.read_bytes() != _json_bytes(expected_aggregate):
        raise control.BrightQrelsRemediationError("safe aggregate recomputation mismatch")
    control._validate_schema(
        aggregate,
        "bright-qrels-remediation-safe-aggregate-v01.schema.json",
        "remediation safe aggregate",
    )
    if aggregate_path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip() != control._sha256_file(
        aggregate_path
    ):
        raise control.BrightQrelsRemediationError("safe aggregate sidecar drift")
    report_path = repo / REPORT_PATH
    expected_report = _report(aggregate)
    if report_path.read_text(encoding="utf-8") != expected_report:
        raise control.BrightQrelsRemediationError("tracked report recomputation mismatch")
    _validate_safe_output_privacy(aggregate, expected_report, pack)
    if report_path.with_suffix(".md.sha256").read_text(encoding="utf-8").strip() != control._sha256_file(report_path):
        raise control.BrightQrelsRemediationError("tracked report sidecar drift")
    return {
        "candidate_id": CANDIDATE_ID,
        "status": "inactive",
        "coverage": proposal_result["coverage"],
        "dispositions": len(proposal["dispositions"]),
        "diff_rows": len(diff_rows),
        "verdict": proposal_result["conclusion"],
        "publication": "closed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Build the full restricted and safe candidate")
    subparsers.add_parser("validate", help="Validate and deterministically recompute the candidate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_candidate() if args.command == "build" else validate_candidate()
    except control.BrightQrelsRemediationError as error:
        raise SystemExit(f"fail_closed: {error}") from error
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
