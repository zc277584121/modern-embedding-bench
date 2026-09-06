"""Fail-closed controls for the restricted BRIGHT qrels remediation proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

CONTROL_ID = "bright-qrels-remediation-control-v0.1"
PROPOSAL_ID = "bright-qrels-remediation-proposal-v0.1"
TRACKS = ("economics", "psychology")
EXPECTED_COVERAGE = {"queries": 120, "gold_items": 1_235, "candidate_items": 2_804}

TAXONOMY = [
    "task_semantics_mismatch",
    "gold_support_defect",
    "gold_ambiguity",
    "credible_missing_positive",
    "candidate_pool_coverage_gap",
    "insufficient_existing_evidence",
    "no_qrels_defect",
]
DISPOSITIONS = {
    "query": [
        "retain_query",
        "stop_query_task_semantics_mismatch",
        "stop_query_insufficient_existing_evidence",
        "not_applicable",
    ],
    "gold": [
        "retain_official_positive",
        "quarantine_official_positive",
        "downgrade_to_uncertain",
        "abstain_uncertain",
        "not_applicable",
    ],
    "candidate": [
        "propose_add_positive",
        "no_change_unjudged",
        "downgrade_to_uncertain",
        "abstain_uncertain",
        "not_applicable",
    ],
}
EVIDENCE_LEVELS = [
    "direct_supporting_span",
    "document_level_support",
    "contextual_or_topical_only",
    "conflicting",
    "insufficient",
]
REQUIRED_PROVENANCE = [
    "judgment_origin",
    "agent_role",
    "session_id",
    "round",
    "judged_at",
    "rubric_sha256",
    "input_artifact_sha256",
    "input_row_sha256",
    "source_annotation_sha256",
]
SEMANTIC_BOUNDARIES = [
    "support_false_does_not_imply_relevance_false",
    "partial_support_does_not_imply_invalid_positive",
    "unjudged_ambiguous_or_uncertain_never_implies_negative",
    "model_score_rank_or_method_win_never_selects_remediation",
    "task_semantics_mismatch_is_not_silently_rewritten_as_qrels_defect",
]
STOP_THRESHOLDS = {
    "missing_or_invalid_provenance": 0,
    "unknown_identity": 0,
    "duplicate_or_conflicting_disposition": 0,
    "source_identity_drift": 0,
    "default_or_runtime_activation": 0,
    "publish_export_or_leaderboard_enablement": 0,
    "negative_grade_or_automatic_negative_conversion": 0,
    "unresolved_membership_change_per_query": 0,
    "stopped_or_incomplete_queries_per_remediable_track": 0,
}
CONCLUSION_STATES = [
    "remediable_with_existing_data",
    "insufficient_existing_evidence",
    "task_semantics_mismatch",
]
DERIVATION_PRECEDENCE = [
    "task_semantics_mismatch",
    "insufficient_existing_evidence",
    "remediable_with_existing_data",
]
FROZEN_RULES = {
    "taxonomy": TAXONOMY,
    "dispositions": DISPOSITIONS,
    "evidence_levels": EVIDENCE_LEVELS,
    "required_provenance": REQUIRED_PROVENANCE,
    "semantic_boundaries": SEMANTIC_BOUNDARIES,
    "stop_thresholds": STOP_THRESHOLDS,
    "conclusion_states": CONCLUSION_STATES,
    "derivation_precedence": DERIVATION_PRECEDENCE,
}

DEFAULT_PATHS = {
    "control": Path("benchmark/artifacts/bright-qrels-remediation-v0.1/control-freeze.json"),
    "review_plan": Path("data/bright-nontechnical-pilot-v0.2/review-plan-120.jsonl"),
    "audit_pack": Path("results/bright-nontechnical-pilot-v0.2/audit/audit-pack-120.jsonl"),
    "primary_annotations": Path("results/bright-label-quality-audit-v0.1/restricted/primary/annotations.jsonl"),
    "validator_core_annotations": Path(
        "results/bright-label-quality-audit-v0.1/restricted/validator-blind/core-annotations.jsonl"
    ),
    "validator_supplement_annotations": Path(
        "results/bright-label-quality-audit-v0.1/restricted/validator-blind/supplement-annotations.jsonl"
    ),
    "adjudications": Path("results/bright-label-quality-audit-v0.1/restricted/adjudication/decisions.jsonl"),
    "frozen_manifest": Path("results/bright-label-quality-audit-v0.1/restricted/frozen-input-manifest.json"),
    "accepted_aggregate": Path("benchmark/artifacts/bright-label-quality-audit-v0.1/aggregate.json"),
    "accepted_predeclaration": Path("benchmark/artifacts/bright-label-quality-audit-v0.1/predeclaration.json"),
    "economics_qrels": Path("data/bright-nontechnical-pilot-v0.2/economics/qrels.jsonl"),
    "psychology_qrels": Path("data/bright-nontechnical-pilot-v0.2/psychology/qrels.jsonl"),
    "restricted_results": Path("results/bright-cross-paradigm-v0.1"),
    "accepted_verification": Path("results/bright-cross-paradigm-v0.1/accepted-verification"),
    "tracked_results": Path("benchmark/artifacts/bright-cross-paradigm-results-v0.1"),
}


class BrightQrelsRemediationError(ValueError):
    """A stable fail-closed remediation validation error."""


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return _sha256_bytes(encoded.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BrightQrelsRemediationError(f"missing immutable input: {path.name}") from error
    return digest.hexdigest()


def _read_object(path: Path, subject: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BrightQrelsRemediationError(f"missing or invalid {subject}") from error
    if not isinstance(value, dict):
        raise BrightQrelsRemediationError(f"{subject} must be a JSON object")
    return value


def _read_jsonl(path: Path, subject: str) -> tuple[dict[str, Any], ...]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BrightQrelsRemediationError(f"non-object {subject} row at line {line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise BrightQrelsRemediationError(f"missing or invalid {subject}") from error
    return tuple(rows)


def _schema_path(name: str) -> Path:
    return _root() / "schemas" / name


@cache
def _schema(name: str) -> dict[str, Any]:
    value = _read_object(_schema_path(name), f"schema {name}")
    Draft202012Validator.check_schema(value)
    return value


def _validate_schema(value: Any, name: str, subject: str) -> None:
    errors = sorted(
        Draft202012Validator(_schema(name), format_checker=FormatChecker()).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.path) or "<root>"
        raise BrightQrelsRemediationError(f"invalid {subject} at {location}: {first.message}")


def _tree_identity(path: Path) -> dict[str, Any]:
    try:
        files = sorted(item for item in path.rglob("*") if item.is_file())
    except OSError as error:
        raise BrightQrelsRemediationError(f"cannot enumerate immutable tree: {path.name}") from error
    if not files:
        raise BrightQrelsRemediationError(f"immutable tree is empty: {path.name}")
    records = "".join(f"{_sha256_file(item)}  ./{item.relative_to(path).as_posix()}\n" for item in files)
    return {"files": len(files), "sha256": _sha256_bytes(records.encode("utf-8"))}


def _file_identity(path: Path) -> dict[str, Any]:
    return {"rows": len(_read_jsonl(path, path.name)), "sha256": _sha256_file(path)}


def _combined_qrels_identity(paths: Iterable[tuple[str, Path]]) -> str:
    records = "".join(f"{_sha256_file(path)}  {logical_path}\n" for logical_path, path in paths)
    return _sha256_bytes(records.encode("utf-8"))


def _git_value(repository: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BrightQrelsRemediationError(f"accepted git identity is unavailable: {revision}")
    return result.stdout.strip()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_chain(path: Path) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        except OSError as error:
            raise BrightQrelsRemediationError("cannot lstat restricted path") from error
        if stat.S_ISLNK(mode):
            raise BrightQrelsRemediationError(f"restricted path contains a symlink: {current.name}")


def _restricted_root_path(
    repository: Path,
    restricted_root: str,
    *,
    create: bool,
) -> Path:
    repo = _lexical_absolute(repository)
    relative = Path(restricted_root)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise BrightQrelsRemediationError("frozen restricted root is not a canonical repository-relative path")
    expected_root = _lexical_absolute(repo / relative)
    if os.path.commonpath((os.fspath(repo), os.fspath(expected_root))) != os.fspath(repo):
        raise BrightQrelsRemediationError("frozen restricted root escapes the repository lexically")
    _assert_no_symlink_chain(repo)
    _assert_no_symlink_chain(expected_root)
    if create:
        current = repo
        for part in relative.parts:
            current /= part
            try:
                mode = os.lstat(current).st_mode
            except FileNotFoundError:
                try:
                    os.mkdir(current)
                except OSError as error:
                    raise BrightQrelsRemediationError("cannot create frozen restricted root") from error
                mode = os.lstat(current).st_mode
            if stat.S_ISLNK(mode):
                raise BrightQrelsRemediationError(f"restricted path contains a symlink: {current.name}")
            if not stat.S_ISDIR(mode):
                raise BrightQrelsRemediationError("restricted root parent is not a directory")
    elif not expected_root.is_dir():
        raise BrightQrelsRemediationError("frozen restricted root is missing or not a directory")
    _assert_no_symlink_chain(expected_root)
    real_repo = os.path.realpath(repo)
    real_root = os.path.realpath(expected_root)
    if os.path.commonpath((real_repo, real_root)) != real_repo:
        raise BrightQrelsRemediationError("frozen restricted root escapes repository realpath containment")
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(expected_root)],
        cwd=repository,
        check=False,
    )
    if ignored.returncode != 0:
        raise BrightQrelsRemediationError("frozen restricted root is not gitignored")
    return expected_root


def _assert_restricted_file(
    repository: Path,
    restricted_root: str,
    path: Path,
    *,
    expected_name: str,
    require_exists: bool,
) -> Path:
    root = _restricted_root_path(repository, restricted_root, create=False)
    candidate = _lexical_absolute(path if path.is_absolute() else _lexical_absolute(repository) / path)
    expected = root / expected_name
    if candidate != expected:
        raise BrightQrelsRemediationError("restricted file is outside its exact frozen lexical path")
    _assert_no_symlink_chain(candidate)
    try:
        mode = os.lstat(candidate).st_mode
    except FileNotFoundError:
        if require_exists:
            raise BrightQrelsRemediationError(f"restricted file is missing: {expected_name}") from None
    except OSError as error:
        raise BrightQrelsRemediationError("cannot lstat restricted file") from error
    else:
        if stat.S_ISLNK(mode):
            raise BrightQrelsRemediationError(f"restricted file is a symlink: {expected_name}")
        if not stat.S_ISREG(mode):
            raise BrightQrelsRemediationError(f"restricted file is not regular: {expected_name}")
    return candidate


def _assert_restricted_proposal_path(repository: Path, proposal_path: Path, restricted_root: str) -> None:
    _assert_restricted_file(
        repository,
        restricted_root,
        proposal_path,
        expected_name="proposal.json",
        require_exists=True,
    )


def _resolve_paths(repository: Path, overrides: dict[str, Path] | None = None) -> dict[str, Path]:
    paths = {key: repository / value for key, value in DEFAULT_PATHS.items()}
    if overrides:
        paths.update(overrides)
    return paths


def validate_control(
    control_path: str | Path = DEFAULT_PATHS["control"],
    *,
    repository: str | Path | None = None,
    path_overrides: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Validate the frozen control and every immutable local baseline."""
    repo = Path(repository) if repository is not None else _root()
    path = Path(control_path)
    if not path.is_absolute():
        path = repo / path
    control = _read_object(path, "remediation control")
    _validate_schema(control, "bright-qrels-remediation-control-v01.schema.json", "remediation control")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        expected_control_sha256 = sidecar.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise BrightQrelsRemediationError("missing remediation control identity sidecar") from error
    if expected_control_sha256 != _sha256_file(path):
        raise BrightQrelsRemediationError("remediation control identity mismatch")
    if control["rules"] != FROZEN_RULES:
        raise BrightQrelsRemediationError("frozen remediation rules drifted")
    if control["rules_sha256"] != _sha256_json(control["rules"]):
        raise BrightQrelsRemediationError("frozen remediation rules identity mismatch")
    for schema_name, expected_sha256 in control["schema_identities"].items():
        if _sha256_file(_schema_path(schema_name)) != expected_sha256:
            raise BrightQrelsRemediationError(f"remediation schema identity drift: {schema_name}")

    baseline = control["accepted_baseline"]
    if _git_value(repo, baseline["commit"] + "^{tree}") != baseline["tree"]:
        raise BrightQrelsRemediationError("accepted commit tree identity mismatch")
    if _git_value(repo, baseline["commit"] + "^") != baseline["parent"]:
        raise BrightQrelsRemediationError("accepted commit parent identity mismatch")

    paths = _resolve_paths(repo, path_overrides)
    source = control["source_identities"]
    audit = source["audit"]
    file_checks = {
        "review_plan": paths["review_plan"],
        "audit_pack": paths["audit_pack"],
        "primary_annotations": paths["primary_annotations"],
        "validator_core_annotations": paths["validator_core_annotations"],
        "validator_supplement_annotations": paths["validator_supplement_annotations"],
        "adjudications": paths["adjudications"],
    }
    for name, input_path in file_checks.items():
        if _file_identity(input_path) != audit[name]:
            raise BrightQrelsRemediationError(f"{name} identity drift")
    scalar_checks = {
        "frozen_input_manifest_sha256": paths["frozen_manifest"],
        "accepted_aggregate_sha256": paths["accepted_aggregate"],
        "accepted_predeclaration_sha256": paths["accepted_predeclaration"],
    }
    for name, input_path in scalar_checks.items():
        if _sha256_file(input_path) != audit[name]:
            raise BrightQrelsRemediationError(f"{name} identity drift")

    aggregate = _read_object(paths["accepted_aggregate"], "accepted audit aggregate")
    coverage = aggregate.get("coverage")
    if coverage != {
        "queries": 120,
        "gold_items": 1_235,
        "candidate_items": 2_804,
        "tracks": {"economics": 60, "psychology": 60},
    }:
        raise BrightQrelsRemediationError("accepted aggregate coverage drift")
    if aggregate.get("outcome") != "fail_closed" or aggregate.get("official_qrels_modified") is not False:
        raise BrightQrelsRemediationError("accepted aggregate gate drift")

    qrels = source["official_qrels"]
    economics = _file_identity(paths["economics_qrels"])
    psychology = _file_identity(paths["psychology_qrels"])
    if economics != qrels["economics"] or psychology != qrels["psychology"]:
        raise BrightQrelsRemediationError("official qrels identity drift")
    combined = _combined_qrels_identity(
        (
            ("economics/qrels.jsonl", paths["economics_qrels"]),
            ("psychology/qrels.jsonl", paths["psychology_qrels"]),
        )
    )
    if combined != qrels["combined_sha256"]:
        raise BrightQrelsRemediationError("official qrels combined identity drift")

    results = source["accepted_results"]
    tree_checks = {
        "restricted_tree": paths["restricted_results"],
        "accepted_verification_tree": paths["accepted_verification"],
        "tracked_safe_package_tree": paths["tracked_results"],
    }
    for name, input_path in tree_checks.items():
        if _tree_identity(input_path) != results[name]:
            raise BrightQrelsRemediationError(f"accepted results {name} identity drift")

    return {
        "control_id": CONTROL_ID,
        "control_sha256": expected_control_sha256,
        "rules_sha256": control["rules_sha256"],
        "coverage": audit["coverage"],
        "official_qrels_unchanged": True,
        "accepted_results_unchanged": True,
        "proposal_status": "not_created",
        "publication": "closed",
    }


def _unit_key(row: dict[str, Any]) -> tuple[str, str, str, str | None]:
    return row["unit_type"], row["track"], row["query_id"], row["document_id"]


def _source_inventory(
    pack: tuple[dict[str, Any], ...],
    primary: tuple[dict[str, Any], ...],
    adjudications: tuple[dict[str, Any], ...],
    official_qrels: dict[tuple[str, str, str], int],
    audit: dict[str, Any],
) -> dict[tuple[str, str, str, str | None], dict[str, str]]:
    primary_by_query = {(row["track"], row["query_id"]): row for row in primary}
    overrides = {
        (row["unit_type"], row["track"], row["query_id"], row.get("document_id")): row["adjudicated_judgment"]
        for row in adjudications
    }
    inventory: dict[tuple[str, str, str, str | None], dict[str, str]] = {}
    for source_row in pack:
        track = source_row["track"]
        query_id = source_row["query_id"]
        annotation = primary_by_query.get((track, query_id))
        if annotation is None:
            raise BrightQrelsRemediationError("accepted annotation query identity is missing")
        input_row_sha256 = annotation["input_row_sha256"]
        units = [("query", None, annotation["query_review"])]
        units.extend(("gold", item["document_id"], item) for item in annotation["gold_reviews"])
        units.extend(("candidate", item["document_id"], item) for item in annotation["candidate_reviews"])
        for unit_type, document_id, primary_judgment in units:
            key = (unit_type, track, query_id, document_id)
            if key in inventory:
                raise BrightQrelsRemediationError("accepted source contains duplicate unit identity")
            judgment = overrides.get(key, primary_judgment)
            if unit_type == "candidate" and (track, query_id, document_id) not in official_qrels:
                status = "unjudged"
            elif judgment.get("decision_status") in {"uncertain", "abstain"} or "uncertain" in {
                judgment.get("support"),
                judgment.get("ambiguity"),
                judgment.get("relevance"),
            }:
                status = "uncertain"
            elif judgment.get("ambiguity") == "ambiguous":
                status = "ambiguous"
            else:
                status = "clear"
            inventory[key] = {
                "source_judgment_status": status,
                "input_row_sha256": input_row_sha256,
                "source_annotation_sha256": (
                    audit["adjudications"]["sha256"] if key in overrides else audit["primary_annotations"]["sha256"]
                ),
            }
    return inventory


def _canonical_documents(
    pack: tuple[dict[str, Any], ...],
) -> dict[tuple[str, str, str], str]:
    """Return canonical document content bound to each frozen query identity."""
    documents: dict[tuple[str, str, str], str] = {}
    global_content: dict[str, str] = {}
    for source_row in pack:
        track = source_row["track"]
        query_id = source_row["query_id"]
        for item in (*source_row["gold_documents"], *source_row["baseline_top10_union"]):
            document_id = item["document_id"]
            content = item["content"]
            key = (track, query_id, document_id)
            existing = documents.get(key)
            if existing is not None and existing != content:
                raise BrightQrelsRemediationError("canonical query document content conflicts")
            global_existing = global_content.get(document_id)
            if global_existing is not None and global_existing != content:
                raise BrightQrelsRemediationError("canonical document identity maps to conflicting content")
            documents[key] = content
            global_content[document_id] = content
    return documents


def _official_qrels(rows: Iterable[dict[str, Any]], track: str) -> dict[tuple[str, str, str], int]:
    result: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (track, row.get("query_id"), row.get("document_id"))
        grade = row.get("grade")
        if key in result or not isinstance(grade, int) or isinstance(grade, bool) or grade < 1:
            raise BrightQrelsRemediationError("official qrels contain invalid or duplicate positive identity")
        result[key] = grade
    return result


def _validate_disposition_semantics(
    row: dict[str, Any],
    expected: dict[str, str],
    control: dict[str, Any],
    canonical_documents: dict[tuple[str, str, str], str] | None = None,
) -> None:
    unit_type = row["unit_type"]
    disposition = row["disposition"]
    if disposition not in DISPOSITIONS[unit_type]:
        raise BrightQrelsRemediationError("disposition is not allowed for its unit type")
    if row["source_judgment_status"] != expected["source_judgment_status"]:
        raise BrightQrelsRemediationError("source judgment status drift")
    taxonomy = set(row["taxonomy"])
    if "no_qrels_defect" in taxonomy and len(taxonomy) != 1:
        raise BrightQrelsRemediationError("no_qrels_defect conflicts with defect taxonomy")
    if row["source_judgment_status"] in {"ambiguous", "uncertain"} and disposition not in {
        "downgrade_to_uncertain",
        "abstain_uncertain",
        "stop_query_task_semantics_mismatch",
        "stop_query_insufficient_existing_evidence",
    }:
        raise BrightQrelsRemediationError("ambiguous or uncertain evidence was converted to a decided label")
    if row["source_judgment_status"] == "unjudged" and disposition not in {
        "no_change_unjudged",
        "propose_add_positive",
        "downgrade_to_uncertain",
        "abstain_uncertain",
    }:
        raise BrightQrelsRemediationError("unjudged evidence was converted to an unsupported label")
    spans = row["supporting_spans"]
    if disposition == "propose_add_positive" and (row["evidence_level"] != "direct_supporting_span" or not spans):
        raise BrightQrelsRemediationError("positive addition lacks direct existing supporting evidence")
    for span in spans:
        if span["document_id"] != row["document_id"]:
            raise BrightQrelsRemediationError("supporting span document identity mismatch")
        if _sha256_bytes(span["text"].encode("utf-8")) != span["span_sha256"]:
            raise BrightQrelsRemediationError("supporting span identity mismatch")
        if canonical_documents is not None:
            document_key = (row["track"], row["query_id"], span["document_id"])
            content = canonical_documents.get(document_key)
            if content is None:
                raise BrightQrelsRemediationError("supporting span uses an unknown canonical document identity")
            start_offset = span["start_offset"]
            end_offset = span["end_offset"]
            if not 0 <= start_offset < end_offset <= len(content):
                raise BrightQrelsRemediationError("supporting span offsets are outside canonical document bounds")
            if content[start_offset:end_offset] != span["text"]:
                raise BrightQrelsRemediationError("supporting span text does not match its declared canonical offsets")
    provenance = row["provenance"]
    audit = control["source_identities"]["audit"]
    if provenance["rubric_sha256"] != audit["rubric_sha256"]:
        raise BrightQrelsRemediationError("disposition rubric identity mismatch")
    if provenance["input_artifact_sha256"] != audit["audit_pack"]["sha256"]:
        raise BrightQrelsRemediationError("disposition input artifact identity mismatch")
    if provenance["input_row_sha256"] != expected["input_row_sha256"]:
        raise BrightQrelsRemediationError("disposition input row identity mismatch")
    if provenance["source_annotation_sha256"] != expected["source_annotation_sha256"]:
        raise BrightQrelsRemediationError("disposition source annotation identity mismatch")


def _derive_conclusion(dispositions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in dispositions:
        by_query.setdefault((row["track"], row["query_id"]), []).append(row)
    track_states: dict[str, str] = {}
    for track in TRACKS:
        states = []
        for (row_track, _query_id), rows in by_query.items():
            if row_track != track:
                continue
            if any(
                "task_semantics_mismatch" in row["taxonomy"]
                or row["disposition"] == "stop_query_task_semantics_mismatch"
                for row in rows
            ):
                states.append("task_semantics_mismatch")
            elif any(
                "candidate_pool_coverage_gap" in row["taxonomy"]
                or "insufficient_existing_evidence" in row["taxonomy"]
                or row["source_judgment_status"] in {"ambiguous", "uncertain"}
                or row["evidence_level"] in {"conflicting", "insufficient"}
                or row["disposition"]
                in {"stop_query_insufficient_existing_evidence", "downgrade_to_uncertain", "abstain_uncertain"}
                for row in rows
            ):
                states.append("insufficient_existing_evidence")
            else:
                states.append("remediable_with_existing_data")
        track_states[track] = next(state for state in DERIVATION_PRECEDENCE if state in states)
    overall = next(state for state in DERIVATION_PRECEDENCE if state in track_states.values())
    return {"tracks": track_states, "overall": overall}


def _validate_proposal_payload(
    proposal: dict[str, Any],
    control: dict[str, Any],
    inventory: dict[tuple[str, str, str, str | None], dict[str, str]],
    official: dict[tuple[str, str, str], int],
    canonical_documents: dict[tuple[str, str, str], str] | None = None,
) -> dict[str, Any]:
    _validate_schema(proposal, "bright-qrels-remediation-proposal-v01.schema.json", "restricted remediation proposal")
    if proposal["source_identities"] != control["source_identities"]:
        raise BrightQrelsRemediationError("proposal source identities drifted")

    seen: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
    for row in proposal["dispositions"]:
        key = _unit_key(row)
        if key in seen:
            raise BrightQrelsRemediationError("duplicate or conflicting disposition")
        expected = inventory.get(key)
        if expected is None:
            raise BrightQrelsRemediationError("unknown disposition identity")
        _validate_disposition_semantics(row, expected, control, canonical_documents)
        seen[key] = row
    if set(seen) != set(inventory):
        raise BrightQrelsRemediationError("proposal disposition coverage is incomplete")

    expected_rows: dict[tuple[str, str, str], tuple[str, int]] = {}
    for row in proposal["dispositions"]:
        if row["unit_type"] == "gold" and row["disposition"] == "retain_official_positive":
            key = (row["track"], row["query_id"], row["document_id"])
            expected_rows[key] = ("retain", official[key])
        elif row["unit_type"] == "candidate" and row["disposition"] == "propose_add_positive":
            key = (row["track"], row["query_id"], row["document_id"])
            if key in official:
                raise BrightQrelsRemediationError("positive addition duplicates an official qrels row")
            expected_rows[key] = ("add", 1)
    actual_rows: dict[tuple[str, str, str], tuple[str, int]] = {}
    for row in proposal["proposed_qrels_rows"]:
        key = (row["track"], row["query_id"], row["document_id"])
        if key in actual_rows:
            raise BrightQrelsRemediationError("duplicate proposed qrels row")
        actual_rows[key] = (row["operation"], row["grade"])
    if actual_rows != expected_rows:
        raise BrightQrelsRemediationError("proposed qrels rows do not exactly match positive dispositions")

    derived = _derive_conclusion(proposal["dispositions"])
    if proposal["conclusion"] != derived:
        raise BrightQrelsRemediationError("proposal conclusion does not match frozen three-state derivation")
    return derived


def validate_proposal(
    proposal_path: str | Path,
    control_path: str | Path = DEFAULT_PATHS["control"],
    *,
    repository: str | Path | None = None,
    path_overrides: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Validate a complete restricted proposal without activating or exporting it."""
    repo = Path(repository) if repository is not None else _root()
    control_summary = validate_control(control_path, repository=repo, path_overrides=path_overrides)
    control_file = Path(control_path)
    if not control_file.is_absolute():
        control_file = repo / control_file
    control = _read_object(control_file, "remediation control")
    proposal_file = Path(proposal_path)
    if not proposal_file.is_absolute():
        proposal_file = repo / proposal_file
    _assert_restricted_proposal_path(repo, proposal_file, control["proposal_constraints"]["restricted_root"])
    proposal = _read_object(proposal_file, "restricted remediation proposal")
    _validate_schema(proposal, "bright-qrels-remediation-proposal-v01.schema.json", "restricted remediation proposal")
    if proposal["control_sha256"] != control_summary["control_sha256"]:
        raise BrightQrelsRemediationError("proposal control identity mismatch")

    paths = _resolve_paths(repo, path_overrides)
    official = _official_qrels(_read_jsonl(paths["economics_qrels"], "economics qrels"), "economics")
    official.update(_official_qrels(_read_jsonl(paths["psychology_qrels"], "psychology qrels"), "psychology"))
    pack = _read_jsonl(paths["audit_pack"], "frozen audit pack")
    primary = _read_jsonl(paths["primary_annotations"], "accepted primary annotations")
    adjudications = _read_jsonl(paths["adjudications"], "accepted adjudications")
    inventory = _source_inventory(pack, primary, adjudications, official, control["source_identities"]["audit"])
    canonical_documents = _canonical_documents(pack)
    derived = _validate_proposal_payload(proposal, control, inventory, official, canonical_documents)
    return {
        "proposal_id": PROPOSAL_ID,
        "status": "inactive",
        "coverage": EXPECTED_COVERAGE,
        "conclusion": derived,
        "publication": "closed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    control_parser = subparsers.add_parser("validate-control", help="Validate frozen controls and baselines")
    control_parser.add_argument("--control", type=Path, default=DEFAULT_PATHS["control"])
    proposal_parser = subparsers.add_parser("validate-proposal", help="Validate one complete inactive proposal")
    proposal_parser.add_argument("proposal", type=Path)
    proposal_parser.add_argument("--control", type=Path, default=DEFAULT_PATHS["control"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-control":
            result = validate_control(args.control)
        else:
            result = validate_proposal(args.proposal, args.control)
    except BrightQrelsRemediationError as error:
        raise SystemExit(f"fail_closed: {error}") from error
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
