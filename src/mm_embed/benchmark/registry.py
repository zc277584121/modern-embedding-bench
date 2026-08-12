"""Data-backed benchmark registry.

The v2 benchmark layer keeps model/task/run definitions in YAML so new models
can be reviewed and added without editing Python code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "benchmark"
EVIDENCE_TIER_ALIASES = {
    "benchmark": "benchmark",
    "fixture": "fixture",
    "fixture_only": "fixture",
    "legacy": "legacy",
    "smoke": "smoke",
    "standard": "benchmark",
    "unknown": "unknown",
}
REVIEW_STATES = {"pending", "approved", "rejected"}


@dataclass(frozen=True)
class PublicProvenanceSpec:
    """Public, pinned evidence metadata safe to project into artifacts."""

    urls: list[str] = field(default_factory=list)
    evidence_revision: str | None = None
    reviewed_at: str | None = None
    reviewed_by: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PublicProvenanceSpec":
        values = dict(data or {})
        return cls(
            urls=[str(value) for value in values.get("urls") or []],
            evidence_revision=_optional_string(values.get("evidence_revision")),
            reviewed_at=_optional_string(values.get("reviewed_at")),
            reviewed_by=_optional_string(values.get("reviewed_by")),
        )


@dataclass(frozen=True)
class ReviewSpec:
    """Review state; private notes are retained locally and never projected."""

    state: str = "pending"
    private_notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ReviewSpec":
        values = dict(data or {})
        state = str(values.get("state", "pending"))
        if state not in REVIEW_STATES:
            raise ValueError(f"Unsupported review state '{state}'")
        return cls(
            state=state,
            private_notes=_optional_string(values.get("private_notes")),
        )


@dataclass(frozen=True)
class TrainingSourceClaim:
    """A model training-source declaration using an exact registry-owned ID."""

    source_id: str
    relation: str = "trained_on"
    scope: str = "material_samples"
    source_revision: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingSourceClaim":
        if not data.get("source_id"):
            raise ValueError("Training source claim is missing source_id")
        relation = str(data.get("relation", "trained_on"))
        scope = str(data.get("scope", "material_samples"))
        if relation != "trained_on":
            raise ValueError(f"Unsupported training source relation '{relation}'")
        if scope != "material_samples":
            raise ValueError(f"Unsupported training source scope '{scope}'")
        return cls(
            source_id=str(data["source_id"]),
            relation=relation,
            scope=scope,
            source_revision=_optional_string(data.get("source_revision")),
        )


@dataclass(frozen=True)
class NegativeTrainingClaim:
    """A reviewed source-specific negative training assertion."""

    source_id: str
    relation: str = "not_trained_on"
    scope: str = "material_samples"
    source_revision: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NegativeTrainingClaim":
        if not data.get("source_id"):
            raise ValueError("Negative training claim is missing source_id")
        if str(data.get("source_id")) == "*":
            raise ValueError("Wildcard negative training claims are not supported")
        relation = str(data.get("relation", "not_trained_on"))
        scope = str(data.get("scope", "material_samples"))
        if relation != "not_trained_on":
            raise ValueError(f"Unsupported negative training relation '{relation}'")
        if scope != "material_samples":
            raise ValueError(f"Unsupported negative training scope '{scope}'")
        return cls(
            source_id=str(data["source_id"]),
            relation=relation,
            scope=scope,
            source_revision=_optional_string(data.get("source_revision")),
        )


@dataclass(frozen=True)
class TrainingDataSpec:
    """Structured model training disclosure normalized to fail-closed unknown."""

    disclosure: str = "unknown"
    source_claims: list[TrainingSourceClaim] = field(default_factory=list)
    negative_claims: list[NegativeTrainingClaim] = field(default_factory=list)
    adapted_from: list[str] = field(default_factory=list)
    lineage_disclosure: str = "unknown"
    model_revision: str | None = None
    public_provenance: PublicProvenanceSpec = field(default_factory=PublicProvenanceSpec)
    review: ReviewSpec = field(default_factory=ReviewSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainingDataSpec":
        values = dict(data or {})
        return cls(
            disclosure=str(values.get("disclosure", "unknown")),
            source_claims=[TrainingSourceClaim.from_dict(item) for item in values.get("source_claims") or []],
            negative_claims=[NegativeTrainingClaim.from_dict(item) for item in values.get("negative_claims") or []],
            adapted_from=[str(value) for value in values.get("adapted_from") or []],
            lineage_disclosure=str(values.get("lineage_disclosure", "unknown")),
            model_revision=_optional_string(values.get("model_revision")),
            public_provenance=PublicProvenanceSpec.from_dict(values.get("public_provenance")),
            review=ReviewSpec.from_dict(values.get("review")),
        )


@dataclass(frozen=True)
class EvaluationSourceClaim:
    """An exact, pinned evaluation-source declaration for a task slice."""

    source_id: str
    usage: str = "evaluation"
    config: str | None = None
    split: str | None = None
    transformation_id: str | None = None
    source_revision: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationSourceClaim":
        if not data.get("source_id"):
            raise ValueError("Evaluation source claim is missing source_id")
        usage = str(data.get("usage", "evaluation"))
        if usage != "evaluation":
            raise ValueError(f"Unsupported evaluation source usage '{usage}'")
        return cls(
            source_id=str(data["source_id"]),
            usage=usage,
            config=_optional_string(data.get("config")),
            split=_optional_string(data.get("split")),
            transformation_id=_optional_string(data.get("transformation_id")),
            source_revision=_optional_string(data.get("source_revision")),
        )


@dataclass(frozen=True)
class EvaluationSourcesSpec:
    """Structured task source disclosure normalized to fail-closed unknown."""

    disclosure: str = "unknown"
    sources: list[EvaluationSourceClaim] = field(default_factory=list)
    public_provenance: PublicProvenanceSpec = field(default_factory=PublicProvenanceSpec)
    review: ReviewSpec = field(default_factory=ReviewSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvaluationSourcesSpec":
        values = dict(data or {})
        return cls(
            disclosure=str(values.get("disclosure", "unknown")),
            sources=[EvaluationSourceClaim.from_dict(item) for item in values.get("sources") or []],
            public_provenance=PublicProvenanceSpec.from_dict(values.get("public_provenance")),
            review=ReviewSpec.from_dict(values.get("review")),
        )


@dataclass(frozen=True)
class MaterializationSpec:
    """Registry-owned expectations for a task materialization transform."""

    transformation_parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MaterializationSpec":
        values = dict(data or {})
        return cls(transformation_parameters=dict(values.get("transformation_parameters") or {}))


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def normalize_evidence_tier(value: Any, *, default: str = "unknown") -> str:
    """Normalize manifest and historical evidence labels for publication."""
    if value is None or not str(value).strip():
        return default
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return EVIDENCE_TIER_ALIASES.get(key, default)


@dataclass(frozen=True)
class ModelSpec:
    """A reviewable model entry from benchmark/models/*.yaml."""

    id: str
    display_name: str
    provider: str
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
    representation_kind: str = "dense_vector"
    model_revision: str | None = None
    vocabulary_id: str | None = None
    representation_id: str | None = None
    query_route: str | None = None
    document_route: str | None = None
    modalities: list[str] = field(default_factory=list)
    dimensions: int | None = None
    max_text_length: int | None = None
    supports_mrl: bool = False
    access: str = "unknown"
    api_key_env: str | None = None
    status: str = "active"
    publish: bool = True
    priority: int = 100
    tags: list[str] = field(default_factory=list)
    source: str | None = None
    notes: str | None = None
    training_data: TrainingDataSpec = field(default_factory=TrainingDataSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_file: Path) -> "ModelSpec":
        required = ("id", "display_name", "provider")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Model entry in {source_file} is missing: {', '.join(missing)}")
        representation_kind = str(data.get("representation_kind", "dense_vector"))
        if representation_kind not in {"dense_vector", "sparse_csr", "multi_vector"}:
            raise ValueError(f"Unsupported model representation_kind '{representation_kind}'")
        dimensions = data.get("dimensions")
        publish = bool(data.get("publish", True))
        sparse_required = ("model_revision", "vocabulary_id", "representation_id", "query_route", "document_route")
        if representation_kind == "sparse_csr":
            missing_sparse = [key for key in sparse_required if not data.get(key)]
            if dimensions is None:
                missing_sparse.append("dimensions")
            if missing_sparse:
                raise ValueError(f"Sparse model entry in {source_file} is missing: {', '.join(missing_sparse)}")
        if data.get("provider") == "deterministic_sparse_fixture" and publish:
            raise ValueError("Deterministic sparse fixture models must disable publication")
        return cls(
            id=str(data["id"]),
            display_name=str(data["display_name"]),
            provider=str(data["provider"]),
            provider_kwargs=dict(data.get("provider_kwargs") or {}),
            representation_kind=representation_kind,
            model_revision=_optional_string(data.get("model_revision")),
            vocabulary_id=_optional_string(data.get("vocabulary_id")),
            representation_id=_optional_string(data.get("representation_id")),
            query_route=_optional_string(data.get("query_route")),
            document_route=_optional_string(data.get("document_route")),
            modalities=list(data.get("modalities") or []),
            dimensions=dimensions,
            max_text_length=data.get("max_text_length"),
            supports_mrl=bool(data.get("supports_mrl", False)),
            access=str(data.get("access", "unknown")),
            api_key_env=data.get("api_key_env"),
            status=str(data.get("status", "active")),
            publish=publish,
            priority=int(data.get("priority", 100)),
            tags=list(data.get("tags") or []),
            source=data.get("source"),
            notes=data.get("notes"),
            training_data=TrainingDataSpec.from_dict(data.get("training_data")),
        )


@dataclass(frozen=True)
class TaskSpec:
    """A reviewable task entry from benchmark/tasks/*.yaml."""

    id: str
    display_name: str
    task: str
    description: str
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    execution_kind: str = "dense"
    fixture_only: bool = False
    score_validity: str | None = None
    required_modalities: list[str] = field(default_factory=list)
    primary_metric: str | None = None
    metric_direction: str = "higher"
    dataset_version: str = "unknown"
    publish: bool = True
    leaderboard_publish: bool = True
    tags: list[str] = field(default_factory=list)
    evaluation_sources: EvaluationSourcesSpec = field(default_factory=EvaluationSourcesSpec)
    materialization: MaterializationSpec = field(default_factory=MaterializationSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_file: Path) -> "TaskSpec":
        required = ("id", "display_name", "task", "description")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Task entry in {source_file} is missing: {', '.join(missing)}")
        execution_kind = str(data.get("execution_kind", "dense"))
        if execution_kind not in {"dense", "sparse_exact", "multi_vector_exact"}:
            raise ValueError(f"Unsupported task execution_kind '{execution_kind}'")
        fixture_only = bool(data.get("fixture_only", False))
        publish = bool(data.get("publish", True))
        leaderboard_publish = bool(data.get("leaderboard_publish", publish))
        if fixture_only and (publish or leaderboard_publish):
            raise ValueError("Fixture-only tasks must disable public and leaderboard publication")
        if execution_kind in {"sparse_exact", "multi_vector_exact"} and not fixture_only:
            raise ValueError("The current exact retrieval task surfaces are contract-fixture only")
        return cls(
            id=str(data["id"]),
            display_name=str(data["display_name"]),
            task=str(data["task"]),
            description=str(data["description"]),
            default_kwargs=dict(data.get("default_kwargs") or {}),
            execution_kind=execution_kind,
            fixture_only=fixture_only,
            score_validity=_optional_string(data.get("score_validity")),
            required_modalities=list(data.get("required_modalities") or []),
            primary_metric=data.get("primary_metric"),
            metric_direction=str(data.get("metric_direction", "higher")),
            dataset_version=str(data.get("dataset_version", "unknown")),
            publish=publish,
            leaderboard_publish=leaderboard_publish,
            tags=list(data.get("tags") or []),
            evaluation_sources=EvaluationSourcesSpec.from_dict(data.get("evaluation_sources")),
            materialization=MaterializationSpec.from_dict(data.get("materialization")),
        )


@dataclass(frozen=True)
class RunTask:
    """A task selected by a run manifest."""

    id: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | dict[str, Any]) -> "RunTask":
        if isinstance(value, str):
            return cls(id=value)
        if isinstance(value, dict):
            task_id = value.get("id") or value.get("task")
            if not task_id:
                raise ValueError(f"Run task entry is missing id/task: {value}")
            return cls(id=str(task_id), kwargs=dict(value.get("kwargs") or {}))
        raise ValueError(f"Unsupported run task entry: {value!r}")


@dataclass(frozen=True)
class RunManifest:
    """A concrete benchmark run plan."""

    id: str
    description: str = ""
    model_ids: list[str] = field(default_factory=list)
    tasks: list[RunTask] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    publish: bool = True
    evidence_tier: str = "benchmark"


@dataclass(frozen=True)
class BenchmarkCatalog:
    """Loaded model and task registries."""

    root: Path
    models: dict[str, ModelSpec]
    tasks: dict[str, TaskSpec]

    def require_model(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.models))
            raise KeyError(f"Unknown model spec '{model_id}'. Available: {available}") from exc

    def require_task(self, task_id: str) -> TaskSpec:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.tasks))
            raise KeyError(f"Unknown task spec '{task_id}'. Available: {available}") from exc


def load_catalog(root: str | Path | None = None) -> BenchmarkCatalog:
    """Load all model and task specs under a benchmark root."""
    benchmark_root = Path(root) if root else DEFAULT_BENCHMARK_ROOT
    model_files = sorted((benchmark_root / "models").glob("*.yaml"))
    task_files = sorted((benchmark_root / "tasks").glob("*.yaml"))

    models: dict[str, ModelSpec] = {}
    for path in model_files:
        for row in _read_yaml(path).get("models", []):
            spec = ModelSpec.from_dict(row, path)
            if spec.id in models:
                raise ValueError(f"Duplicate model id '{spec.id}' in {path}")
            models[spec.id] = spec

    tasks: dict[str, TaskSpec] = {}
    for path in task_files:
        for row in _read_yaml(path).get("tasks", []):
            spec = TaskSpec.from_dict(row, path)
            if spec.id in tasks:
                raise ValueError(f"Duplicate task id '{spec.id}' in {path}")
            tasks[spec.id] = spec

    catalog = BenchmarkCatalog(root=benchmark_root, models=models, tasks=tasks)
    relationship_path = benchmark_root / "training_overlap_relationships.yaml"
    if relationship_path.exists():
        from mm_embed.benchmark.training_overlap import load_relationship_registry, validate_catalog_contract

        validate_catalog_contract(catalog, load_relationship_registry(relationship_path))
    elif any(model.training_data.disclosure != "unknown" for model in models.values()) or any(
        task.evaluation_sources.disclosure != "unknown" for task in tasks.values()
    ):
        raise ValueError("Structured training/evaluation declarations require a relationship registry")
    return catalog


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load a benchmark run manifest."""
    run_path = Path(path)
    data = _read_yaml(run_path)
    if "id" not in data:
        raise ValueError(f"Run manifest {run_path} is missing id")
    metadata = dict(data.get("metadata") or {})
    explicit_evidence_tier = data.get("evidence_tier")
    if explicit_evidence_tier is not None:
        evidence_tier = normalize_evidence_tier(explicit_evidence_tier)
        if evidence_tier == "unknown" and str(explicit_evidence_tier).strip().lower() != "unknown":
            raise ValueError(f"Run manifest {run_path} has unsupported evidence_tier: {explicit_evidence_tier}")
    else:
        evidence_tier = normalize_evidence_tier(
            metadata.get("evidence_tier", metadata.get("tier")),
            default="benchmark",
        )
    publish = bool(data.get("publish", True))
    if evidence_tier == "fixture" and publish:
        raise ValueError(f"Fixture run manifest {run_path} must disable publication")
    return RunManifest(
        id=str(data["id"]),
        description=str(data.get("description", "")),
        model_ids=[str(item) for item in data.get("models", [])],
        tasks=[RunTask.from_value(item) for item in data.get("tasks", [])],
        metadata=metadata,
        publish=publish,
        evidence_tier=evidence_tier,
    )
