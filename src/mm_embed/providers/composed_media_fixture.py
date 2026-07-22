"""Deterministic zero-network test double for the composed-media fixture."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from mm_embed.data.composed_media_retrieval import FIXTURE_ROOT, load_composed_media_retrieval_fixture
from mm_embed.providers.composed_media import (
    ComposedMediaEmbeddingRequest,
    ComposedMediaEmbeddingResult,
    ComposedMediaEmbeddingRow,
    result_fingerprint,
    validate_composed_request,
)


class DeterministicComposedMediaTestDouble:
    """Fixture-only provider that never performs inference or network I/O."""

    name = "deterministic-composed-media-local"
    model = "composed-media-fixture-label-test-double"
    model_revision = "composed-media-fixture-test-double-v0"
    composed_dimensions = 16

    def __init__(self, *, corrupt_system_result: str | None = None) -> None:
        self.corrupt_system_result = corrupt_system_result
        self.requests: list[ComposedMediaEmbeddingRequest] = []

    def _fallback_vector(self, item_id: str) -> tuple[float, ...]:
        digest = hashlib.sha256(item_id.encode("utf-8")).digest()
        values = [float(digest[index] + 1) for index in range(self.composed_dimensions)]
        norm = sum(value * value for value in values) ** 0.5
        return tuple(value / norm for value in values)

    def _vector_for_item(self, item_id: str, composition_mode: str) -> tuple[float, ...]:
        fixture = load_composed_media_retrieval_fixture()
        corpus_ids = [row.corpus_id for row in fixture.corpus]
        if item_id in corpus_ids:
            values = [0.0] * self.composed_dimensions
            values[corpus_ids.index(item_id)] = 1.0
            return tuple(values)

        qrels = [qrel for qrel in fixture.qrels if qrel.query_id == item_id]
        if not qrels:
            return self._fallback_vector(item_id)
        values = [0.0] * self.composed_dimensions
        for qrel in qrels:
            values[corpus_ids.index(qrel.corpus_id)] = 1.0 if qrel.relevance == 2 else 0.7
        if composition_mode == "benchmark_system_fusion":
            query_index = [query.query_id for query in fixture.queries].index(item_id)
            if query_index % 3 == 0:
                distractor = next(
                    negative
                    for negative in fixture.hard_negatives
                    if negative.query_id == item_id
                )
                values[corpus_ids.index(distractor.corpus_id)] = 1.25
        return tuple(values)

    def embed_composed_media(
        self,
        request: ComposedMediaEmbeddingRequest,
    ) -> ComposedMediaEmbeddingResult:
        validate_composed_request(request, FIXTURE_ROOT)
        self.requests.append(request)
        route_evidence: dict[str, Any] = {
            "endpoint": "deterministic_local_test_double",
            "fusion_strategy": request.fusion_strategy,
            "network": "forbidden",
            "provider_api_calls": 0,
        }
        rows = tuple(
            ComposedMediaEmbeddingRow(
                item_id=item.item_id,
                item_sha256=item.item_sha256,
                request_sha256=request.request_sha256,
                provider=request.provider,
                model_id=request.model_id,
                model_revision=request.model_revision,
                composition_mode=request.composition_mode,
                track_label=request.track_label,
                dimensions=request.dimensions,
                route_evidence=route_evidence,
                embedding=self._vector_for_item(item.item_id, request.composition_mode),
            )
            for item in request.items
        )
        result = ComposedMediaEmbeddingResult(
            request_sha256=request.request_sha256,
            rows=rows,
            dimensions=request.dimensions,
            provider=request.provider,
            model_id=request.model_id,
            model_revision=request.model_revision,
            composition_mode=request.composition_mode,
            track_label=request.track_label,
            score_validity="contract_fixture_only",
            route_evidence=route_evidence,
            latency_ms=0.0,
            result_sha256="",
        )
        result = replace(result, result_sha256=result_fingerprint(result))
        if request.composition_mode == "benchmark_system_fusion" and self.corrupt_system_result:
            if self.corrupt_system_result == "reordered" and len(result.rows) > 1:
                result = replace(result, rows=(result.rows[1], result.rows[0], *result.rows[2:]))
            elif self.corrupt_system_result == "flattened":
                result = replace(result, rows=(*result.rows, result.rows[0]))
            elif self.corrupt_system_result == "zero_vector":
                row = replace(result.rows[0], embedding=(0.0,) * request.dimensions)
                result = replace(result, rows=(row, *result.rows[1:]))
            result = replace(result, result_sha256=result_fingerprint(result))
        return result
