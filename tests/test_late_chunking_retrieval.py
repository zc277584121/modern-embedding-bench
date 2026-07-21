from __future__ import annotations

import json
import socket
from dataclasses import replace

import numpy as np
import pytest

from mm_embed.benchmark.registry import load_catalog, load_run_manifest
from mm_embed.data.late_chunking_retrieval import (
    LAYOUT_IDS,
    fixture_to_dict,
    fixture_with_qrels,
    load_late_chunking_retrieval_fixture,
    validate_late_chunking_retrieval_fixture,
)
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType
from mm_embed.providers.grouped_chunks import (
    GroupedChunkEmbedding,
    GroupedChunkEmbeddingGroup,
    GroupedChunkEmbeddingRequest,
    GroupedChunkEmbeddingResult,
    grouped_request_fingerprint,
    grouped_result_from_dict,
    grouped_result_to_dict,
    validate_grouped_chunk_result,
)
from mm_embed.tasks.late_chunking_retrieval import _build_grouped_request
from mm_embed.tasks.registry import get_task


class DeterministicGroupedChunkProvider(EmbeddingProvider):
    name = "deterministic-grouped-local"
    model = "fixture-label-test-double"
    supported_modalities = {ModalityType.TEXT}
    dimensions = 17

    def __init__(self, corrupt_contextual_id: bool = False) -> None:
        super().__init__()
        self.fixture = load_late_chunking_retrieval_fixture()
        self.corrupt_contextual_id = corrupt_contextual_id
        self.query_index = {query.query_id: index for index, query in enumerate(self.fixture.queries)}
        self.query_by_id = {query.query_id: query for query in self.fixture.queries}
        self.qrels = {
            (qrel.layout_id, qrel.query_id, qrel.chunk_id)
            for qrel in self.fixture.qrels
        }
        self.negative_family = {
            (negative.layout_id, negative.query_id, negative.chunk_id): negative.negative_family
            for negative in self.fixture.hard_negatives
        }
        self.grouped_requests: list[GroupedChunkEmbeddingRequest] = []

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        assert task_type == "retrieval_query"
        embeddings = np.zeros((len(inputs), self.dimensions), dtype=float)
        for row, item in enumerate(inputs):
            query_id = str(item.metadata["query_id"])
            embeddings[row, self.query_index[query_id]] = 1.0
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=self.dimensions,
            model_name=self.model,
            provider=self.name,
            latency_ms=0.0,
        )

    def embed_grouped_chunks(self, request: GroupedChunkEmbeddingRequest) -> GroupedChunkEmbeddingResult:
        self.grouped_requests.append(request)
        contextual = request.strategy == "deterministic_contextual_stub"
        groups: list[GroupedChunkEmbeddingGroup] = []
        for group in request.groups:
            returned_chunks: list[GroupedChunkEmbedding] = []
            for chunk in group.chunks:
                vector = np.zeros(self.dimensions, dtype=float)
                for query in self.fixture.queries:
                    coordinate = 0.01
                    key = (request.layout_id, query.query_id, chunk.chunk_id)
                    if key in self.qrels:
                        coordinate = 0.20 if contextual or not query.context_required else 0.08
                    elif key in self.negative_family:
                        family_rank = {
                            "cross_document_local_text_collision": 0,
                            "misleading_gold_parent_neighbor": 1,
                            "same_family_scope_collision": 2,
                        }[self.negative_family[key]]
                        if contextual:
                            coordinate = (0.06, 0.05, 0.04)[family_rank]
                        elif query.context_required:
                            coordinate = (0.18, 0.16, 0.14)[family_rank]
                        else:
                            coordinate = (0.06, 0.05, 0.04)[family_rank]
                    vector[self.query_index[query.query_id]] = coordinate
                vector[-1] = np.sqrt(1.0 - float(np.square(vector[:-1]).sum()))
                returned_chunks.append(
                    GroupedChunkEmbedding(
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        chunk_index=chunk.chunk_index,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        text_sha256=chunk.text_sha256,
                        returned_text=chunk.text,
                        mapping_status="exact",
                        embedding=vector,
                        provider_chunk_id=f"stub:{chunk.chunk_id}",
                    )
                )
            groups.append(GroupedChunkEmbeddingGroup(document_id=group.document_id, chunks=tuple(returned_chunks)))

        result = GroupedChunkEmbeddingResult(
            layout_id=request.layout_id,
            strategy=request.strategy,
            groups=tuple(groups),
            dimensions=self.dimensions,
            model_name=self.model,
            provider=self.name,
            latency_ms=0.0,
            token_usage=0,
            cost_usd=0.0,
            metadata={"request_fingerprint": grouped_request_fingerprint(request)},
        )
        if contextual and self.corrupt_contextual_id:
            first_group = result.groups[0]
            first_chunk = replace(first_group.chunks[0], chunk_id="lost-canonical-id")
            corrupted_group = replace(first_group, chunks=(first_chunk, *first_group.chunks[1:]))
            result = replace(result, groups=(corrupted_group, *result.groups[1:]))
        return result


def _request_and_result() -> tuple[GroupedChunkEmbeddingRequest, GroupedChunkEmbeddingResult]:
    fixture = load_late_chunking_retrieval_fixture()
    layout = next(layout for layout in fixture.layouts if layout.layout_id == "fixed_192_v1")
    request = _build_grouped_request(fixture, layout, "deterministic_independent_stub")
    result = DeterministicGroupedChunkProvider().embed_grouped_chunks(request)
    return request, result


def test_fixture_shape_identity_and_generated_labels() -> None:
    fixture = load_late_chunking_retrieval_fixture()

    assert fixture.fixture_only is True
    assert fixture.split == "fixture_only"
    assert fixture.license_status == "not_for_publication"
    assert fixture.leaderboard_publish is False
    assert fixture.network == "forbidden"
    assert len(fixture.documents) == 8
    assert len(fixture.queries) == 16
    assert sum(query.context_required for query in fixture.queries) == 12
    assert sum(not query.context_required for query in fixture.queries) == 4
    assert tuple(layout.layout_id for layout in fixture.layouts) == LAYOUT_IDS
    assert [len(layout.chunks) for layout in fixture.layouts] == [32, 40, 32]

    assert all(len(document.text.split()) == 768 for document in fixture.documents)
    document_by_id = {document.document_id: document for document in fixture.documents}
    for span in fixture.spans:
        parent = document_by_id[span.document_id]
        assert parent.text[span.char_start:span.char_end] == span.text
    for layout in fixture.layouts:
        for chunk in layout.chunks:
            parent = document_by_id[chunk.document_id]
            assert parent.text[chunk.char_start:chunk.char_end] == chunk.text

    qrel_counts = {
        layout_id: sum(qrel.layout_id == layout_id for qrel in fixture.qrels)
        for layout_id in LAYOUT_IDS
    }
    assert qrel_counts == {
        "fixed_192_v1": 16,
        "fixed_192_overlap_48_v1": 24,
        "structure_adaptive_fixture_v1": 16,
    }
    overlap_counts = {
        query.query_id: sum(
            qrel.query_id == query.query_id and qrel.layout_id == "fixed_192_overlap_48_v1"
            for qrel in fixture.qrels
        )
        for query in fixture.queries
    }
    assert sorted(overlap_counts.values()) == [1] * 8 + [2] * 8

    span_by_id = {span.span_id: span for span in fixture.spans}
    fixed_layout = fixture.layouts[0]
    for query in fixture.queries:
        if not query.context_required:
            continue
        evidence_span = span_by_id[query.evidence_span_ids[0]]
        context_span = span_by_id[query.required_context_span_ids[0]]
        evidence_chunk = next(
            chunk
            for chunk in fixed_layout.chunks
            if chunk.document_id == query.gold_document_id
            and chunk.char_start <= evidence_span.char_start
            and chunk.char_end >= evidence_span.char_end
        )
        context_chunk = next(
            chunk
            for chunk in fixed_layout.chunks
            if chunk.document_id == query.gold_document_id
            and chunk.char_start <= context_span.char_start
            and chunk.char_end >= context_span.char_end
        )
        assert evidence_chunk.chunk_id != context_chunk.chunk_id

    for layout_id in LAYOUT_IDS:
        assert sum(negative.layout_id == layout_id for negative in fixture.hard_negatives) == 48


def test_fixture_serialization_and_labels_are_deterministic() -> None:
    first = load_late_chunking_retrieval_fixture()
    second = load_late_chunking_retrieval_fixture()

    assert first.label_sha256 == second.label_sha256
    assert json.dumps(fixture_to_dict(first), sort_keys=True) == json.dumps(fixture_to_dict(second), sort_keys=True)

    missing_overlap_positive = next(
        qrel
        for qrel in first.qrels
        if qrel.layout_id == "fixed_192_overlap_48_v1"
        and sum(
            other.query_id == qrel.query_id and other.layout_id == qrel.layout_id
            for other in first.qrels
        )
        == 2
    )
    corrupted = fixture_with_qrels(
        first,
        tuple(qrel for qrel in first.qrels if qrel != missing_overlap_positive),
    )
    with pytest.raises(ValueError, match="Qrels do not reproduce"):
        validate_late_chunking_retrieval_fixture(corrupted)


def test_grouped_contract_round_trip_retains_text_ids_and_boundaries() -> None:
    request, result = _request_and_result()

    validate_grouped_chunk_result(request, result)
    restored = grouped_result_from_dict(grouped_result_to_dict(result))
    validate_grouped_chunk_result(request, restored)

    assert grouped_result_to_dict(restored) == grouped_result_to_dict(result)
    assert restored.groups[0].chunks[0].returned_text == request.groups[0].chunks[0].text
    assert restored.groups[0].chunks[0].chunk_id == request.groups[0].chunks[0].chunk_id
    assert restored.groups[0].chunks[0].char_start == request.groups[0].chunks[0].char_start
    assert restored.groups[0].chunks[0].text_sha256 == request.groups[0].chunks[0].text_sha256

    reordered_request = replace(request, groups=(request.groups[1], request.groups[0], *request.groups[2:]))
    assert grouped_request_fingerprint(reordered_request) != grouped_request_fingerprint(request)


@pytest.mark.parametrize(
    "corruption",
    ["flattened", "reordered_groups", "reordered_chunks", "cross_document", "lost_text", "lost_id"],
)
def test_grouped_contract_detects_identity_and_group_corruption(corruption: str) -> None:
    request, result = _request_and_result()
    groups = list(result.groups)
    first_group = groups[0]

    if corruption == "flattened":
        all_chunks = tuple(chunk for group in groups for chunk in group.chunks)
        corrupted = replace(result, groups=(replace(first_group, chunks=all_chunks),))
    elif corruption == "reordered_groups":
        corrupted = replace(result, groups=(groups[1], groups[0], *groups[2:]))
    elif corruption == "reordered_chunks":
        reordered_group = replace(first_group, chunks=tuple(reversed(first_group.chunks)))
        corrupted = replace(result, groups=(reordered_group, *groups[1:]))
    else:
        first_chunk = first_group.chunks[0]
        if corruption == "cross_document":
            first_chunk = replace(first_chunk, document_id=groups[1].document_id)
        elif corruption == "lost_text":
            first_chunk = replace(first_chunk, returned_text="")
        elif corruption == "lost_id":
            first_chunk = replace(first_chunk, chunk_id="")
        corrupted = replace(
            result,
            groups=(replace(first_group, chunks=(first_chunk, *first_group.chunks[1:])), *groups[1:]),
        )

    with pytest.raises(ValueError):
        validate_grouped_chunk_result(request, corrupted)


def test_task_runs_full_hard_and_paired_metrics_deterministically() -> None:
    task = get_task("late_chunking_retrieval")
    provider = DeterministicGroupedChunkProvider()

    result = task.run(provider)

    assert result.passed, result.error
    assert len(provider.grouped_requests) == 6
    assert result.metrics["chunk_ndcg@10"] == pytest.approx(1.0)
    assert result.metrics["fixed_contextual_chunk_mrr"] == pytest.approx(1.0)
    assert result.metrics["fixed_contextual_hard_mrr"] == pytest.approx(1.0)
    assert result.metrics["fixed_contextual_neighbor_confusion_rate"] == pytest.approx(0.0)
    assert result.metrics["fixed_independent_chunk_mrr"] == pytest.approx(0.4375)
    assert result.metrics["fixed_independent_hard_mrr"] == pytest.approx(0.4375)
    assert result.metrics["fixed_independent_neighbor_confusion_rate"] == pytest.approx(0.75)
    assert result.metrics["context_gain_chunk_mrr"] == pytest.approx(0.5625)
    assert result.metrics["context_gain_chunk_ndcg@10"] == pytest.approx(0.42699258144495517)
    assert result.metrics["context_gain_hard_mrr"] == pytest.approx(0.5625)
    assert result.metrics["irrelevant_context_delta"] == pytest.approx(0.0)
    assert result.metrics["overlap_gain_evidence_span_recall@5"] == pytest.approx(0.0)
    assert result.metrics["adaptive_gain_parent_recall@5"] == pytest.approx(0.0)

    assert result.details["fixture_only"] is True
    assert result.details["not_for_publication"] is True
    assert result.details["leaderboard_publish"] is False
    assert result.details["network"] == "forbidden"
    assert result.details["provider_api_calls"] == 0
    assert result.details["model_downloads"] == 0
    assert result.details["layout_counts"] == {
        "fixed_192_v1": {"chunks": 32, "positive_qrels": 16, "hard_negative_links": 48},
        "fixed_192_overlap_48_v1": {"chunks": 40, "positive_qrels": 24, "hard_negative_links": 48},
        "structure_adaptive_fixture_v1": {"chunks": 32, "positive_qrels": 16, "hard_negative_links": 48},
    }
    serialized = result.details["serialized_primary_results"]
    assert serialized["contextual"]["groups"][0]["chunks"][0]["returned_text"]
    assert serialized["contextual"]["groups"][0]["chunks"][0]["chunk_id"].startswith("fixed_192_v1:")
    assert serialized["contextual"]["groups"][0]["chunks"][0]["char_end"] > 0
    assert len(serialized["contextual"]["groups"][0]["chunks"][0]["text_sha256"]) == 64


def test_task_rejects_unpairable_context_delta() -> None:
    task = get_task("late_chunking_retrieval")
    result = task.run(DeterministicGroupedChunkProvider(corrupt_contextual_id=True))

    assert not result.passed
    assert "Unpairable context delta" in str(result.error)


def test_task_runtime_is_zero_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("network access is forbidden for the fixture smoke")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    result = get_task("late_chunking_retrieval").run(DeterministicGroupedChunkProvider())

    assert result.passed, result.error
    assert result.details["network"] == "forbidden"
    assert result.details["provider_api_calls"] == 0


def test_catalog_registry_and_manifest_are_explicitly_no_publish() -> None:
    catalog = load_catalog()
    spec = catalog.tasks["late_chunking_retrieval"]

    assert spec.primary_metric == "chunk_ndcg@10"
    assert spec.dataset_version == "late-chunking-retrieval-fixture-v0"
    assert spec.publish is False
    assert {"fixture-only", "no-publish", "contextual", "hard-negative"}.issubset(spec.tags)
    assert get_task("late_chunking_retrieval").name == "late_chunking_retrieval"

    manifest = load_run_manifest("benchmark/runs/late-chunking-retrieval-local-smoke.yaml")
    assert manifest.publish is False
    assert manifest.model_ids == []
    assert [task.id for task in manifest.tasks] == ["late_chunking_retrieval"]
    assert manifest.metadata["network"] == "forbidden"
    assert manifest.metadata["leaderboard_publish"] is False
