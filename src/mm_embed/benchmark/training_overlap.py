"""Deterministic training-overlap and strict zero-shot assessment.

The contract uses only exact registry-owned identifiers and approved typed
relationships. It performs no network access and intentionally fails closed.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from mm_embed.benchmark.registry import BenchmarkCatalog, ModelSpec, TaskSpec


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELATIONSHIP_REGISTRY_PATH = REPO_ROOT / "benchmark" / "training_overlap_relationships.yaml"
TRAINING_OVERLAP_SCHEMA_VERSION = "1"
TRAINING_OVERLAP_CONTRACT_VERSION = "1"
LEGACY_RELATIONSHIP_DIGEST = "0" * 64

DATA_OVERLAP_STATUSES = {"exact", "adapted", "declared_none", "unknown"}
TASK_TRAINING_STATUSES = {"same_task", "similar_task", "declared_none", "unknown"}
ZERO_SHOT_STATUSES = {"no", "reviewed_yes", "unknown"}
REVIEW_STATES = {"pending", "approved", "rejected"}
REASON_CODES = {
    "exact_source_match",
    "approved_alias_match",
    "approved_adapted_source_match",
    "same_task_exposure",
    "similar_task_exposure",
    "complete_reviewed_non_overlap",
    "model_training_disclosure_unknown",
    "task_source_unknown",
    "unresolved_model_lineage",
    "ambiguous_source_mapping",
    "stale_relationship",
    "stale_model_evidence",
    "stale_task_evidence",
    "conflicting_claims",
    "legacy_missing_contract",
}
REASON_ORDER = {reason: index for index, reason in enumerate(sorted(REASON_CODES))}

RELATIONSHIP_EFFECTS = {
    "alias_of": ("exact", None, True),
    "same_examples_as": ("exact", "same_task", True),
    "subset_of": ("adapted", "same_task", True),
    "sampled_from": ("adapted", "same_task", True),
    "translated_from": ("adapted", "same_task", True),
    "reformatted_from": ("adapted", "same_task", True),
    "same_task_as": (None, "same_task", False),
    "similar_task_to": (None, "similar_task", False),
}
EXACT_PREDICATES = {"alias_of", "same_examples_as"}
MATERIAL_PREDICATES = {"subset_of", "sampled_from", "translated_from", "reformatted_from"}


@dataclass(frozen=True)
class SourceRelationshipSpec:
    id: str
    canonical: bool
    locator: dict[str, Any]
    public_provenance: dict[str, Any]
    review: dict[str, Any]

    @property
    def revision(self) -> str | None:
        for key in ("revision", "dataset_version", "contract_revision"):
            value = self.locator.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return None


@dataclass(frozen=True)
class RelationshipSpec:
    id: str
    subject: str
    predicate: str
    object: str
    effect: dict[str, Any]
    applies_to: dict[str, Any]
    public_provenance: dict[str, Any]
    review: dict[str, Any]


@dataclass(frozen=True)
class RelationshipRegistry:
    schema_version: str
    revision: str
    sha256: str
    sources: dict[str, SourceRelationshipSpec]
    relationships: tuple[RelationshipSpec, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, sha256: str | None = None) -> "RelationshipRegistry":
        if not isinstance(data, dict):
            raise ValueError("Training-overlap relationship registry must be a mapping")
        canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        digest = sha256 or hashlib.sha256(canonical).hexdigest()
        sources: dict[str, SourceRelationshipSpec] = {}
        for row in data.get("sources") or []:
            source_id = str(row.get("id") or "")
            if not source_id:
                raise ValueError("Training-overlap source is missing id")
            if source_id in sources:
                raise ValueError(f"Duplicate training-overlap source id '{source_id}'")
            sources[source_id] = SourceRelationshipSpec(
                id=source_id,
                canonical=bool(row.get("canonical", True)),
                locator=dict(row.get("locator") or {}),
                public_provenance=dict(row.get("public_provenance") or {}),
                review=dict(row.get("review") or {}),
            )
        relationships = []
        relationship_ids: set[str] = set()
        for row in data.get("relationships") or []:
            relationship_id = str(row.get("id") or "")
            if not relationship_id:
                raise ValueError("Training-overlap relationship is missing id")
            if relationship_id in relationship_ids:
                raise ValueError(f"Duplicate training-overlap relationship id '{relationship_id}'")
            relationship_ids.add(relationship_id)
            relationships.append(
                RelationshipSpec(
                    id=relationship_id,
                    subject=str(row.get("subject") or ""),
                    predicate=str(row.get("predicate") or ""),
                    object=str(row.get("object") or ""),
                    effect=dict(row.get("effect") or {}),
                    applies_to=dict(row.get("applies_to") or {}),
                    public_provenance=dict(row.get("public_provenance") or {}),
                    review=dict(row.get("review") or {}),
                )
            )
        registry = cls(
            schema_version=str(data.get("schema_version") or ""),
            revision=str(data.get("revision") or ""),
            sha256=digest,
            sources=sources,
            relationships=tuple(relationships),
        )
        validate_relationship_registry(registry)
        return registry


@dataclass
class _Match:
    kind: str
    relationship_ids: list[str] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)


def load_relationship_registry(path: str | Path | None = None) -> RelationshipRegistry:
    """Load and validate the local relationship registry without network access."""
    registry_path = Path(path) if path else DEFAULT_RELATIONSHIP_REGISTRY_PATH
    raw = registry_path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8")) or {}
    return RelationshipRegistry.from_dict(data, sha256=hashlib.sha256(raw).hexdigest())


def validate_relationship_registry(registry: RelationshipRegistry) -> None:
    """Validate source identity, typed effects, review gates, and alias uniqueness."""
    if registry.schema_version != TRAINING_OVERLAP_SCHEMA_VERSION:
        raise ValueError(f"Unsupported training-overlap relationship schema: {registry.schema_version}")
    if not registry.revision:
        raise ValueError("Training-overlap relationship registry is missing revision")
    for source in registry.sources.values():
        review_state = source.review.get("state")
        if review_state not in REVIEW_STATES:
            raise ValueError(f"Source '{source.id}' has unsupported review state '{review_state}'")
        if review_state == "approved":
            _require_public_review(source.public_provenance, f"source '{source.id}'")
            if not source.revision:
                raise ValueError(f"Approved source '{source.id}' is missing a pinned locator revision")

    for relationship in registry.relationships:
        review_state = relationship.review.get("state")
        if review_state not in REVIEW_STATES:
            raise ValueError(
                f"Relationship '{relationship.id}' has unsupported review state '{review_state}'"
            )
        if relationship.predicate not in RELATIONSHIP_EFFECTS:
            raise ValueError(f"Unsupported relationship predicate '{relationship.predicate}'")
        if relationship.subject not in registry.sources or relationship.object not in registry.sources:
            raise ValueError(f"Relationship '{relationship.id}' references an unknown source")
        expected_data, expected_task, expected_transitive = RELATIONSHIP_EFFECTS[relationship.predicate]
        actual = (
            relationship.effect.get("data_overlap"),
            relationship.effect.get("task_training"),
            relationship.effect.get("transitive"),
        )
        if actual != (expected_data, expected_task, expected_transitive):
            raise ValueError(f"Relationship '{relationship.id}' has invalid effect semantics")
        if review_state != "approved":
            continue
        _require_public_review(relationship.public_provenance, f"relationship '{relationship.id}'")
        subject_revision = registry.sources[relationship.subject].revision
        object_revision = registry.sources[relationship.object].revision
        if relationship.applies_to.get("subject_revision") != subject_revision:
            raise ValueError(f"Relationship '{relationship.id}' has a stale subject revision")
        if relationship.applies_to.get("object_revision") != object_revision:
            raise ValueError(f"Relationship '{relationship.id}' has a stale object revision")

    alias_graph = _relationship_graph(registry, EXACT_PREDICATES, current_only=False)
    visited: set[str] = set()
    for source_id in sorted(registry.sources):
        if source_id in visited:
            continue
        component = _component(source_id, alias_graph)
        visited.update(component)
        canonical_ids = [value for value in component if registry.sources[value].canonical]
        if len(component) > 1 and len(canonical_ids) != 1:
            raise ValueError(
                "Ambiguous alias component has {} canonical representatives: {}".format(
                    len(canonical_ids), ", ".join(sorted(canonical_ids)) or "none"
                )
            )


def validate_catalog_contract(catalog: "BenchmarkCatalog", registry: RelationshipRegistry) -> None:
    """Validate structured declarations against exact source and model IDs."""
    validate_relationship_registry(registry)
    _validate_catalog_semantics(catalog)
    for model in catalog.models.values():
        training = model.training_data
        for parent_id in training.adapted_from:
            if parent_id not in catalog.models:
                raise ValueError(f"Model '{model.id}' references unknown parent '{parent_id}'")
        for claim in [*training.source_claims, *training.negative_claims]:
            if claim.source_id not in registry.sources:
                raise ValueError(f"Model '{model.id}' references unknown source '{claim.source_id}'")
            _validate_claim_revision(claim.source_id, claim.source_revision, registry, f"model '{model.id}'")
        if training.disclosure == "complete" and not training.source_claims:
            if not _reviewed_training_declaration(model):
                raise ValueError(f"Model '{model.id}' has an unreviewed complete empty training declaration")
        _validate_conflicting_claims(model, registry)
    _validate_lineage_cycles(catalog)

    for task in catalog.tasks.values():
        evaluation = task.evaluation_sources
        for claim in evaluation.sources:
            if claim.source_id not in registry.sources:
                raise ValueError(f"Task '{task.id}' references unknown source '{claim.source_id}'")
            _validate_claim_revision(claim.source_id, claim.source_revision, registry, f"task '{task.id}'")
        if evaluation.disclosure == "complete" and (not evaluation.sources or not _reviewed_task_declaration(task)):
            raise ValueError(f"Task '{task.id}' has an incomplete reviewed evaluation-source declaration")


def assess_training_overlap(
    *,
    model: "ModelSpec",
    task: "TaskSpec",
    catalog: "BenchmarkCatalog",
    relationship_registry: RelationshipRegistry,
    assessed_at: str | None,
) -> dict[str, Any]:
    """Assess one model/task pair using only frozen local declarations."""
    validate_relationship_registry(relationship_registry)
    _validate_catalog_semantics(catalog)
    reasons: set[str] = set()
    task_claims = _usable_task_claims(task, relationship_registry, reasons)
    lineage, lineage_complete = _resolved_lineage(model, catalog)
    if not lineage_complete:
        reasons.add("unresolved_model_lineage")

    data_status = "unknown"
    task_status = "unknown"
    matched_models: set[str] = set()
    matched_training_sources: set[str] = set()
    matched_evaluation_sources: set[str] = set()
    relationship_ids: set[str] = set()

    for lineage_model in lineage:
        training = lineage_model.training_data
        if not _approved_public_provenance(training.review.state, training.public_provenance):
            reasons.add("model_training_disclosure_unknown")
            continue
        for claim in training.source_claims:
            if not _claim_is_current(claim.source_id, claim.source_revision, relationship_registry):
                reasons.add("stale_model_evidence")
                continue
            for evaluation_claim in task_claims:
                exact_graph = _relationship_graph(
                    relationship_registry,
                    EXACT_PREDICATES,
                    current_only=True,
                    model=lineage_model,
                    task=task,
                )
                material_graph = _relationship_graph(
                    relationship_registry,
                    EXACT_PREDICATES | MATERIAL_PREDICATES,
                    current_only=True,
                    model=lineage_model,
                    task=task,
                )
                exact = _graph_match(claim.source_id, evaluation_claim.source_id, exact_graph, "exact")
                adapted = _graph_match(claim.source_id, evaluation_claim.source_id, material_graph, "adapted")
                task_match = _task_relationship_match(
                    claim.source_id,
                    evaluation_claim.source_id,
                    relationship_registry,
                    model=lineage_model,
                    task=task,
                )
                selected_data = exact or adapted
                if selected_data is None:
                    permissive_graph = _relationship_graph(
                        relationship_registry,
                        EXACT_PREDICATES | MATERIAL_PREDICATES,
                        current_only=False,
                    )
                    stale_match = _graph_match(
                        claim.source_id,
                        evaluation_claim.source_id,
                        permissive_graph,
                        "adapted",
                    )
                    if stale_match:
                        stale_relationships = {
                            relationship.id: relationship for relationship in relationship_registry.relationships
                        }
                        for relationship_id in stale_match.relationship_ids:
                            relationship = stale_relationships[relationship_id]
                            if not _relationship_is_current(
                                relationship,
                                relationship_registry,
                                model=lineage_model,
                                task=task,
                            ):
                                reasons.add("stale_relationship")
                                if relationship.applies_to.get("model_evidence_revision") not in (
                                    None,
                                    lineage_model.training_data.public_provenance.evidence_revision,
                                ):
                                    reasons.add("stale_model_evidence")
                                if relationship.applies_to.get("task_evidence_revision") not in (
                                    None,
                                    task.evaluation_sources.public_provenance.evidence_revision,
                                ) or relationship.applies_to.get("task_dataset_version") not in (
                                    None,
                                    task.dataset_version,
                                ):
                                    reasons.add("stale_task_evidence")
                if selected_data:
                    if selected_data.kind == "exact":
                        data_status = "exact"
                        if not selected_data.relationship_ids:
                            reasons.add("exact_source_match")
                        elif "alias_of" in selected_data.predicates:
                            reasons.add("approved_alias_match")
                        else:
                            reasons.add("exact_source_match")
                    elif data_status != "exact":
                        data_status = "adapted"
                        reasons.add("approved_adapted_source_match")
                    matched_models.add(lineage_model.id)
                    matched_training_sources.add(claim.source_id)
                    matched_evaluation_sources.add(evaluation_claim.source_id)
                    relationship_ids.update(selected_data.relationship_ids)
                if claim.source_id == evaluation_claim.source_id:
                    task_match = _Match("same_task")
                if task_match:
                    if task_match.kind == "same_task":
                        task_status = "same_task"
                        reasons.add("same_task_exposure")
                    elif task_status != "same_task":
                        task_status = "similar_task"
                        reasons.add("similar_task_exposure")
                    matched_models.add(lineage_model.id)
                    matched_training_sources.add(claim.source_id)
                    matched_evaluation_sources.add(evaluation_claim.source_id)
                    relationship_ids.update(task_match.relationship_ids)

    can_conclude_none = bool(task_claims) and lineage_complete and _lineage_supports_negative(
        lineage, task_claims, relationship_registry
    )
    if data_status == "unknown" and can_conclude_none:
        data_status = "declared_none"
    if task_status == "unknown" and can_conclude_none:
        task_status = "declared_none"
    if data_status == "declared_none" and task_status == "declared_none":
        reasons.add("complete_reviewed_non_overlap")

    zero_shot_status = _derived_zero_shot_status(data_status, task_status)

    if not task_claims:
        reasons.add("task_source_unknown")
    if all(item.training_data.disclosure == "unknown" for item in lineage):
        reasons.add("model_training_disclosure_unknown")

    return {
        "schema_version": TRAINING_OVERLAP_SCHEMA_VERSION,
        "relationship_registry_revision": relationship_registry.revision,
        "relationship_registry_sha256": relationship_registry.sha256,
        "model_revision": model.training_data.model_revision,
        "model_training_evidence_revision": model.training_data.public_provenance.evidence_revision,
        "task_dataset_version": task.dataset_version,
        "task_source_evidence_revision": task.evaluation_sources.public_provenance.evidence_revision,
        "data_overlap_status": data_status,
        "task_training_status": task_status,
        "zero_shot_status": zero_shot_status,
        "matched_model_ids": sorted(matched_models),
        "matched_training_source_ids": sorted(matched_training_sources),
        "matched_evaluation_source_ids": sorted(matched_evaluation_sources),
        "relationship_ids": sorted(relationship_ids),
        "reason_codes": _ordered_reasons(reasons),
        "assessed_at": assessed_at,
    }


def validate_assessment_snapshot(snapshot: dict[str, Any]) -> None:
    """Validate a frozen public snapshot without consulting today's catalog."""
    if snapshot.get("schema_version") != TRAINING_OVERLAP_SCHEMA_VERSION:
        raise ValueError("Unsupported training-overlap assessment schema")
    if snapshot.get("data_overlap_status") not in DATA_OVERLAP_STATUSES:
        raise ValueError("Unsupported data_overlap_status")
    if snapshot.get("task_training_status") not in TASK_TRAINING_STATUSES:
        raise ValueError("Unsupported task_training_status")
    if snapshot.get("zero_shot_status") not in ZERO_SHOT_STATUSES:
        raise ValueError("Unsupported zero_shot_status")
    expected_zero_shot_status = _derived_zero_shot_status(
        str(snapshot["data_overlap_status"]),
        str(snapshot["task_training_status"]),
    )
    if snapshot.get("zero_shot_status") != expected_zero_shot_status:
        raise ValueError(
            "zero_shot_status is inconsistent with data_overlap_status and task_training_status"
        )
    digest = snapshot.get("relationship_registry_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Invalid training-overlap relationship digest")
    unknown_reasons = set(snapshot.get("reason_codes") or []) - REASON_CODES
    if unknown_reasons:
        raise ValueError(f"Unsupported training-overlap reason codes: {sorted(unknown_reasons)}")


def validate_assessment_registry_binding(
    snapshot: dict[str, Any],
    relationship_registry: RelationshipRegistry,
) -> None:
    """Reject a same-revision snapshot whose digest no longer matches."""
    validate_assessment_snapshot(snapshot)
    if (
        snapshot.get("relationship_registry_revision") == relationship_registry.revision
        and snapshot.get("relationship_registry_sha256") != relationship_registry.sha256
    ):
        raise ValueError("Training-overlap assessment relationship digest is stale")


def legacy_unknown_assessment() -> dict[str, Any]:
    """Return the explicit frozen public interpretation for pre-contract rows."""
    return {
        "schema_version": TRAINING_OVERLAP_SCHEMA_VERSION,
        "relationship_registry_revision": "legacy",
        "relationship_registry_sha256": LEGACY_RELATIONSHIP_DIGEST,
        "model_revision": None,
        "model_training_evidence_revision": None,
        "task_dataset_version": None,
        "task_source_evidence_revision": None,
        "data_overlap_status": "unknown",
        "task_training_status": "unknown",
        "zero_shot_status": "unknown",
        "matched_model_ids": [],
        "matched_training_source_ids": [],
        "matched_evaluation_source_ids": [],
        "relationship_ids": [],
        "reason_codes": ["legacy_missing_contract"],
        "assessed_at": None,
    }


def public_assessment_for_record(record: dict[str, Any]) -> dict[str, Any]:
    """Use the historical snapshot or explicitly normalize a legacy row."""
    snapshot = record.get("training_overlap")
    if snapshot is None:
        if record.get("training_overlap_contract_version") == TRAINING_OVERLAP_CONTRACT_VERSION:
            raise ValueError("Post-contract public result is missing training_overlap")
        return legacy_unknown_assessment()
    if not isinstance(snapshot, dict):
        raise ValueError("training_overlap must be an object")
    validate_assessment_snapshot(snapshot)
    return dict(snapshot)


def public_model_training_projection(model: "ModelSpec") -> dict[str, Any]:
    """Project only explicitly public model training evidence."""
    training = model.training_data
    return {
        "disclosure": training.disclosure,
        "lineage_disclosure": training.lineage_disclosure,
        "model_revision": training.model_revision,
        "source_ids": sorted(claim.source_id for claim in training.source_claims),
        "adapted_from": sorted(training.adapted_from),
        "evidence_revision": training.public_provenance.evidence_revision,
        "evidence_urls": list(training.public_provenance.urls),
        "review_state": training.review.state,
    }


def public_task_source_projection(task: "TaskSpec") -> dict[str, Any]:
    """Project only explicitly public task source evidence."""
    evaluation = task.evaluation_sources
    return {
        "disclosure": evaluation.disclosure,
        "sources": [
            {
                "source_id": claim.source_id,
                "source_revision": claim.source_revision,
                "config": claim.config,
                "split": claim.split,
                "transformation_id": claim.transformation_id,
            }
            for claim in evaluation.sources
        ],
        "evidence_revision": evaluation.public_provenance.evidence_revision,
        "evidence_urls": list(evaluation.public_provenance.urls),
        "review_state": evaluation.review.state,
    }


def _require_public_review(provenance: dict[str, Any], owner: str) -> None:
    if not provenance.get("urls") or not provenance.get("reviewed_at") or not provenance.get("reviewed_by"):
        raise ValueError(f"Approved {owner} is missing public review provenance")


def _validate_catalog_semantics(catalog: "BenchmarkCatalog") -> None:
    for model in catalog.models.values():
        training = model.training_data
        if training.disclosure not in {"unknown", "partial", "complete"}:
            raise ValueError(f"Model '{model.id}' has unsupported training disclosure")
        if training.lineage_disclosure not in {"unknown", "complete"}:
            raise ValueError(f"Model '{model.id}' has unsupported lineage disclosure")
        if training.review.state not in REVIEW_STATES:
            raise ValueError(f"Model '{model.id}' has unsupported review state '{training.review.state}'")
        for claim in training.source_claims:
            if claim.relation != "trained_on":
                raise ValueError(f"Model '{model.id}' has unsupported training relation '{claim.relation}'")
            if claim.scope != "material_samples":
                raise ValueError(f"Model '{model.id}' has unsupported training scope '{claim.scope}'")
        for claim in training.negative_claims:
            if claim.relation != "not_trained_on":
                raise ValueError(f"Model '{model.id}' has unsupported negative relation '{claim.relation}'")
            if claim.scope != "material_samples":
                raise ValueError(f"Model '{model.id}' has unsupported negative scope '{claim.scope}'")

    for task in catalog.tasks.values():
        evaluation = task.evaluation_sources
        if evaluation.disclosure not in {"unknown", "complete"}:
            raise ValueError(f"Task '{task.id}' has unsupported evaluation source disclosure")
        if evaluation.review.state not in REVIEW_STATES:
            raise ValueError(f"Task '{task.id}' has unsupported review state '{evaluation.review.state}'")
        for claim in evaluation.sources:
            if claim.usage != "evaluation":
                raise ValueError(f"Task '{task.id}' has unsupported evaluation usage '{claim.usage}'")


def _derived_zero_shot_status(data_status: str, task_status: str) -> str:
    if data_status in {"exact", "adapted"} or task_status in {"same_task", "similar_task"}:
        return "no"
    if data_status == "declared_none" and task_status == "declared_none":
        return "reviewed_yes"
    return "unknown"


def _validate_claim_revision(
    source_id: str,
    claim_revision: str | None,
    registry: RelationshipRegistry,
    owner: str,
) -> None:
    if claim_revision != registry.sources[source_id].revision:
        raise ValueError(f"{owner} has a stale source revision for '{source_id}'")


def _approved_public_provenance(state: str, provenance: Any) -> bool:
    return (
        state == "approved"
        and bool(provenance.urls)
        and bool(provenance.evidence_revision)
        and bool(provenance.reviewed_at)
        and bool(provenance.reviewed_by)
    )


def _reviewed_training_declaration(model: "ModelSpec") -> bool:
    training = model.training_data
    return (
        training.lineage_disclosure == "complete"
        and bool(training.model_revision)
        and _approved_public_provenance(training.review.state, training.public_provenance)
    )


def _reviewed_task_declaration(task: "TaskSpec") -> bool:
    evaluation = task.evaluation_sources
    return _approved_public_provenance(evaluation.review.state, evaluation.public_provenance)


def _validate_lineage_cycles(catalog: "BenchmarkCatalog") -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(model_id: str) -> None:
        if model_id in visiting:
            raise ValueError(f"Model lineage cycle includes '{model_id}'")
        if model_id in visited:
            return
        visiting.add(model_id)
        for parent_id in catalog.models[model_id].training_data.adapted_from:
            visit(parent_id)
        visiting.remove(model_id)
        visited.add(model_id)

    for model_id in catalog.models:
        visit(model_id)


def _validate_conflicting_claims(model: "ModelSpec", registry: RelationshipRegistry) -> None:
    exact_graph = _relationship_graph(registry, EXACT_PREDICATES, current_only=True)
    positives = {claim.source_id for claim in model.training_data.source_claims}
    for negative in model.training_data.negative_claims:
        if any(_graph_match(positive, negative.source_id, exact_graph, "exact") for positive in positives):
            raise ValueError(f"Model '{model.id}' has conflicting positive and negative source claims")


def _usable_task_claims(task: "TaskSpec", registry: RelationshipRegistry, reasons: set[str]) -> list[Any]:
    evaluation = task.evaluation_sources
    if evaluation.disclosure != "complete" or not _reviewed_task_declaration(task) or not evaluation.sources:
        return []
    current = []
    for claim in evaluation.sources:
        if claim.source_id not in registry.sources:
            reasons.add("task_source_unknown")
        elif not _claim_is_current(claim.source_id, claim.source_revision, registry):
            reasons.add("stale_task_evidence")
        else:
            current.append(claim)
    return current


def _claim_is_current(source_id: str, revision: str | None, registry: RelationshipRegistry) -> bool:
    source = registry.sources.get(source_id)
    return bool(source and source.review.get("state") == "approved" and revision == source.revision)


def _resolved_lineage(model: "ModelSpec", catalog: "BenchmarkCatalog") -> tuple[list["ModelSpec"], bool]:
    resolved: dict[str, "ModelSpec"] = {}
    visiting: set[str] = set()
    complete = True

    def visit(current: "ModelSpec") -> None:
        nonlocal complete
        if current.id in visiting:
            complete = False
            return
        if current.id in resolved:
            return
        resolved[current.id] = current
        visiting.add(current.id)
        for parent_id in current.training_data.adapted_from:
            parent = catalog.models.get(parent_id)
            if parent is None:
                complete = False
                continue
            visit(parent)
        visiting.remove(current.id)

    visit(model)
    return [resolved[key] for key in sorted(resolved)], complete


def _lineage_supports_negative(
    lineage: list["ModelSpec"],
    task_claims: list[Any],
    registry: RelationshipRegistry,
) -> bool:
    exact_graph = _relationship_graph(registry, EXACT_PREDICATES, current_only=True)
    for model in lineage:
        training = model.training_data
        if not _reviewed_training_declaration(model):
            return False
        if training.lineage_disclosure != "complete":
            return False
        for task_claim in task_claims:
            if training.disclosure == "complete":
                continue
            matching_negative = any(
                _claim_is_current(claim.source_id, claim.source_revision, registry)
                and _graph_match(claim.source_id, task_claim.source_id, exact_graph, "exact")
                for claim in training.negative_claims
            )
            if not matching_negative:
                return False
    return True


def _relationship_is_current(
    relationship: RelationshipSpec,
    registry: RelationshipRegistry,
    *,
    model: "ModelSpec" | None = None,
    task: "TaskSpec" | None = None,
) -> bool:
    if relationship.review.get("state") != "approved":
        return False
    subject = registry.sources[relationship.subject]
    object_ = registry.sources[relationship.object]
    applies = relationship.applies_to
    if applies.get("subject_revision") != subject.revision or applies.get("object_revision") != object_.revision:
        return False
    if model is not None and applies.get("model_evidence_revision") not in (
        None,
        model.training_data.public_provenance.evidence_revision,
    ):
        return False
    if task is not None and applies.get("task_evidence_revision") not in (
        None,
        task.evaluation_sources.public_provenance.evidence_revision,
    ):
        return False
    if task is not None and applies.get("task_dataset_version") not in (None, task.dataset_version):
        return False
    return True


def _relationship_graph(
    registry: RelationshipRegistry,
    predicates: set[str],
    *,
    current_only: bool,
    model: "ModelSpec" | None = None,
    task: "TaskSpec" | None = None,
) -> dict[str, list[tuple[str, RelationshipSpec]]]:
    graph: dict[str, list[tuple[str, RelationshipSpec]]] = defaultdict(list)
    for relationship in registry.relationships:
        if relationship.predicate not in predicates:
            continue
        if relationship.review.get("state") != "approved":
            continue
        if current_only and not _relationship_is_current(relationship, registry, model=model, task=task):
            continue
        graph[relationship.subject].append((relationship.object, relationship))
        graph[relationship.object].append((relationship.subject, relationship))
    for edges in graph.values():
        edges.sort(key=lambda item: (item[0], item[1].id))
    return graph


def _graph_match(
    source_id: str,
    target_id: str,
    graph: dict[str, list[tuple[str, RelationshipSpec]]],
    requested_kind: str,
) -> _Match | None:
    if source_id == target_id:
        return _Match("exact")
    queue = deque([(source_id, [], [])])
    visited = {source_id}
    while queue:
        current, relationship_ids, predicates = queue.popleft()
        for next_id, relationship in graph.get(current, []):
            if next_id in visited:
                continue
            next_relationships = [*relationship_ids, relationship.id]
            next_predicates = [*predicates, relationship.predicate]
            if next_id == target_id:
                kind = "adapted" if any(value in MATERIAL_PREDICATES for value in next_predicates) else "exact"
                if requested_kind == "exact" and kind != "exact":
                    return None
                return _Match(kind, next_relationships, next_predicates)
            visited.add(next_id)
            queue.append((next_id, next_relationships, next_predicates))
    return None


def _task_relationship_match(
    source_id: str,
    target_id: str,
    registry: RelationshipRegistry,
    *,
    model: "ModelSpec",
    task: "TaskSpec",
) -> _Match | None:
    exact_graph = _relationship_graph(
        registry,
        EXACT_PREDICATES,
        current_only=True,
        model=model,
        task=task,
    )
    source_component = _component(source_id, exact_graph)
    target_component = _component(target_id, exact_graph)
    same_match: _Match | None = None
    similar_match: _Match | None = None
    for relationship in registry.relationships:
        if relationship.predicate not in {"same_examples_as", *MATERIAL_PREDICATES, "same_task_as", "similar_task_to"}:
            continue
        if not _relationship_is_current(relationship, registry, model=model, task=task):
            continue
        bridges_components = (
            relationship.subject in source_component and relationship.object in target_component
        ) or (
            relationship.object in source_component and relationship.subject in target_component
        )
        if not bridges_components or relationship.subject == relationship.object:
            continue
        if relationship.predicate == "similar_task_to":
            similar_match = _Match("similar_task", [relationship.id], [relationship.predicate])
        else:
            same_match = _Match("same_task", [relationship.id], [relationship.predicate])
    return same_match or similar_match


def _component(source_id: str, graph: dict[str, list[tuple[str, RelationshipSpec]]]) -> set[str]:
    component = {source_id}
    queue = deque([source_id])
    while queue:
        current = queue.popleft()
        for next_id, _relationship in graph.get(current, []):
            if next_id not in component:
                component.add(next_id)
                queue.append(next_id)
    return component


def _ordered_reasons(reasons: Iterable[str]) -> list[str]:
    known = [reason for reason in set(reasons) if reason in REASON_CODES]
    return sorted(known, key=lambda reason: (REASON_ORDER[reason], reason))
