"""Manifest-based benchmark runner."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable

from mm_embed.benchmark.materialization import prepare_data_source_contract
from mm_embed.benchmark.registry import BenchmarkCatalog, RunManifest, RunTask, load_catalog, load_run_manifest
from mm_embed.benchmark.results import append_jsonl, make_result_record, utc_now_iso
from mm_embed.providers import get_provider
from mm_embed.providers.sparse_base import SparseEmbeddingProvider
from mm_embed.tasks import get_task
from mm_embed.tasks.base import EvalResult

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run a v2 manifest against the existing provider/task implementations."""

    def __init__(
        self,
        catalog: BenchmarkCatalog | None = None,
        output: str | Path = "results/benchmark-v2.jsonl",
        limit: int | None = None,
        overwrite: bool = False,
    ):
        self.catalog = catalog or load_catalog()
        self.output = Path(output)
        self.limit = limit
        self.overwrite = overwrite

    @classmethod
    def from_paths(
        cls,
        benchmark_root: str | Path | None,
        output: str | Path,
        limit: int | None = None,
        overwrite: bool = False,
    ) -> "BenchmarkRunner":
        return cls(catalog=load_catalog(benchmark_root), output=output, limit=limit, overwrite=overwrite)

    def run_manifest(self, manifest: RunManifest) -> list[dict]:
        """Run all model/task combinations from a manifest."""
        if self.overwrite and self.output.exists():
            self.output.unlink()

        records: list[dict] = []
        completed = 0

        task_plans = []
        for run_task in manifest.tasks:
            task = self.catalog.require_task(run_task.id)
            task_kwargs = self._merged_task_kwargs(task.default_kwargs, run_task)
            prepared_source = prepare_data_source_contract(task, task_kwargs, self.catalog.root)
            task_plans.append((task, task_kwargs, prepared_source))

        requires_provider = any(
            prepared_source is None or prepared_source.error is None
            for _, _, prepared_source in task_plans
        )

        for model_id in manifest.model_ids:
            model = self.catalog.require_model(model_id)
            provider = None
            provider_error = self._provider_preflight_error(model) if requires_provider else None
            if requires_provider and provider_error is None:
                try:
                    provider = get_provider(model.provider, **model.provider_kwargs)
                except Exception as exc:
                    provider_error = f"provider init failed: {exc}"

            for task, task_kwargs, prepared_source in task_plans:
                effective_run_task = RunTask(id=task.id, kwargs=task_kwargs)
                started_at = utc_now_iso()
                start = time.perf_counter()

                if prepared_source is not None and prepared_source.error:
                    result = EvalResult(
                        task_name=task.task,
                        provider_name=model.provider,
                        model_name=model.provider_kwargs.get("model", model.id),
                        metrics={},
                        error=f"data source contract validation failed: {prepared_source.error}",
                    )
                elif provider_error:
                    result = EvalResult(
                        task_name=task.task,
                        provider_name=model.provider,
                        model_name=model.provider_kwargs.get("model", model.id),
                        metrics={},
                        error=provider_error,
                    )
                else:
                    execution_kwargs = dict(task_kwargs)
                    if prepared_source is not None:
                        execution_kwargs["materialization_binding"] = prepared_source.authorization
                    result = self._run_one(provider, model, task, execution_kwargs)

                finished_at = utc_now_iso()
                record = make_result_record(
                    run=manifest,
                    model=model,
                    task=task,
                    run_task=effective_run_task,
                    result=result,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_s=time.perf_counter() - start,
                    catalog=self.catalog,
                    data_source_contract=(prepared_source.snapshot if prepared_source is not None else None),
                )
                append_jsonl(self.output, record)
                records.append(record)
                completed += 1

                status = "ERROR" if record.get("error") else "OK"
                logger.info("%s %s / %s -> %s", status, model.id, task.id, self.output)

                if self.limit is not None and completed >= self.limit:
                    return records

        return records

    @staticmethod
    def from_manifest_path(
        run_manifest: str | Path,
        benchmark_root: str | Path | None = None,
        output: str | Path = "results/benchmark-v2.jsonl",
        limit: int | None = None,
        overwrite: bool = False,
    ) -> tuple["BenchmarkRunner", RunManifest]:
        runner = BenchmarkRunner.from_paths(benchmark_root, output, limit=limit, overwrite=overwrite)
        manifest = load_run_manifest(run_manifest)
        return runner, manifest

    @staticmethod
    def _provider_preflight_error(model) -> str | None:
        if model.status != "active":
            return f"model status is {model.status}"
        if model.api_key_env and not os.environ.get(model.api_key_env):
            return f"missing required environment variable: {model.api_key_env}"
        return None

    @staticmethod
    def _merged_task_kwargs(defaults: dict, run_task: RunTask) -> dict:
        merged = dict(defaults)
        merged.update(run_task.kwargs)
        return merged

    @staticmethod
    def _run_one(provider, model, task, kwargs: dict) -> EvalResult:
        try:
            if model.representation_kind == "sparse_csr" and task.execution_kind != "sparse_exact":
                raise ValueError("Sparse models require an explicitly sparse task")
            if model.representation_kind != "sparse_csr" and task.execution_kind == "sparse_exact":
                raise ValueError("Sparse tasks require an explicitly sparse model")
            if task.execution_kind == "sparse_exact":
                if not isinstance(provider, SparseEmbeddingProvider):
                    raise TypeError("Sparse task provider does not implement SparseEmbeddingProvider")
                expected = {
                    "model_revision": provider.revision,
                    "vocabulary_id": provider.representation.vocabulary_id,
                    "representation_id": provider.representation.representation_id,
                    "dimensions": provider.representation.dimensions,
                    "query_route": provider.query_route.value,
                    "document_route": provider.document_route.value,
                }
                declared = {
                    "model_revision": model.model_revision,
                    "vocabulary_id": model.vocabulary_id,
                    "representation_id": model.representation_id,
                    "dimensions": model.dimensions,
                    "query_route": model.query_route,
                    "document_route": model.document_route,
                }
                if declared != expected:
                    raise ValueError("Sparse provider identity does not match the model registry contract")
            task_instance = get_task(task.task, **kwargs)
            return task_instance.run(provider)
        except Exception as exc:
            return EvalResult(
                task_name=task.task,
                provider_name=getattr(provider, "name", "unknown"),
                model_name=getattr(provider, "model", "unknown"),
                metrics={},
                error=str(exc),
            )


def selected_ids(all_ids: Iterable[str], requested: tuple[str, ...]) -> list[str]:
    """Return requested ids, or all ids if no selection was provided."""
    if not requested:
        return list(all_ids)
    available = set(all_ids)
    missing = [item for item in requested if item not in available]
    if missing:
        raise KeyError(f"Unknown selection: {', '.join(missing)}")
    return list(requested)
