from __future__ import annotations

import inspect
import json
import shutil
import socket
from dataclasses import replace

import numpy as np
import pytest

from mm_embed.benchmark.leaderboard import build_leaderboard
from mm_embed.benchmark.registry import load_catalog, load_run_manifest
from mm_embed.data.composed_media_retrieval import (
    DATASET_VERSION,
    FIXTURE_ROOT,
    HARD_NEGATIVE_FAMILIES,
    ComposedMediaQrel,
    fixture_fingerprint,
    fixture_to_dict,
    fixture_tracked_bytes,
    fixture_tree_sha256,
    generate_composed_media_retrieval_fixture,
    load_composed_media_retrieval_fixture,
    validate_composed_media_retrieval_fixture,
)
from mm_embed.hf_publish.export import export_dataset_repo, export_space_repo
from mm_embed.providers.base import EmbeddingProvider
from mm_embed.providers.composed_media import (
    ComposedMediaEmbeddingRequest,
    ComposedMediaEmbeddingResult,
    canonical_json_bytes,
    request_fingerprint,
    result_fingerprint,
    result_to_dict,
    validate_composed_request,
    validate_composed_result,
)
from mm_embed.providers.composed_media_fixture import DeterministicComposedMediaTestDouble
from mm_embed.tasks.composed_media_retrieval import (
    RetrievalEvaluation,
    build_composed_request,
    evaluate_composed_retrieval,
    paired_fusion_deltas,
    stable_corpus_ranking,
)
from mm_embed.tasks.registry import get_task


def _request(
    *,
    mode: str = "provider_native_fusion",
    role: str = "query",
) -> ComposedMediaEmbeddingRequest:
    fixture = load_composed_media_retrieval_fixture()
    items = (
        [query.item for query in fixture.queries]
        if role == "query"
        else [row.item for row in fixture.corpus]
    )
    track = "provider_valid_embedding" if mode == "provider_native_fusion" else "benchmark_system_fusion"
    return build_composed_request(
        items=items,
        provider=DeterministicComposedMediaTestDouble.name,
        model_id=DeterministicComposedMediaTestDouble.model,
        model_revision=DeterministicComposedMediaTestDouble.model_revision,
        dimensions=DeterministicComposedMediaTestDouble.composed_dimensions,
        task_route=f"composed_media_retrieval_{role}",
        preprocessing="composed-media-fixture-generator-v0",
        composition_mode=mode,
        track_label=track,
        fusion_strategy=f"deterministic_{mode}_v0",
    )


def _result(
    *,
    mode: str = "provider_native_fusion",
    role: str = "query",
) -> tuple[ComposedMediaEmbeddingRequest, ComposedMediaEmbeddingResult]:
    request = _request(mode=mode, role=role)
    result = DeterministicComposedMediaTestDouble().embed_composed_media(request)
    return request, result


def _refingerprint_fixture(fixture):
    fixture = replace(fixture, fixture_sha256="")
    return replace(fixture, fixture_sha256=fixture_fingerprint(fixture))


def _evaluation(mode: str) -> RetrievalEvaluation:
    fixture = load_composed_media_retrieval_fixture()
    provider = DeterministicComposedMediaTestDouble()
    query_request = _request(mode=mode, role="query")
    corpus_request = _request(mode=mode, role="corpus")
    query_result = provider.embed_composed_media(query_request)
    corpus_result = provider.embed_composed_media(corpus_request)
    return evaluate_composed_retrieval(
        fixture=fixture,
        query_result=query_result,
        corpus_result=corpus_result,
        preprocessing=query_request.preprocessing,
    )


def test_flat_embedding_provider_signature_is_unchanged() -> None:
    parameters = list(inspect.signature(EmbeddingProvider.embed).parameters)
    assert parameters == ["self", "inputs", "dimensions", "task_type"]


def test_fixture_has_exact_authored_shape_and_bounded_assets() -> None:
    fixture = load_composed_media_retrieval_fixture()

    assert fixture.dataset_version == DATASET_VERSION
    assert fixture.fixture_only is True
    assert fixture.publish is False
    assert fixture.leaderboard_publish is False
    assert fixture.license_status == "not_for_publication"
    assert fixture.network == "forbidden"
    assert fixture.provider_api_calls == 0
    assert len(fixture.queries) == 12
    assert [query.shape for query in fixture.queries].count("text_image_to_video") == 4
    assert [query.shape for query in fixture.queries].count("text_video_to_video") == 4
    assert [query.shape for query in fixture.queries].count("source_audio_text_to_audio") == 4
    assert len(fixture.corpus) == 12
    assert len(fixture.qrels) == 16
    assert len(fixture.hard_negatives) == 36
    assert set(negative.negative_family for negative in fixture.hard_negatives) == set(HARD_NEGATIVE_FAMILIES)
    assert all(
        sum(negative.query_id == query.query_id for negative in fixture.hard_negatives) == 3
        for query in fixture.queries
    )
    assert len(list(FIXTURE_ROOT.glob("videos/video_*/manifest.json"))) == 6
    assert len(list(FIXTURE_ROOT.glob("videos/video_*/*.png"))) == 96
    assert len(list(FIXTURE_ROOT.glob("audio/*.wav"))) == 6
    assert len(list(FIXTURE_ROOT.glob("stills/*.png"))) == 6
    assert fixture_tracked_bytes() == 1_411_262
    assert fixture_tracked_bytes() < 10 * 1024 * 1024
    assert fixture.fixture_sha256 == "b1aea6428314524bb0278249499a933d8d154b468bd26c4c71f2ab45322092cb"
    assert fixture_tree_sha256() == "307e13e959a99ee134340cd5acf2ea1efcd8ff297c4251ec48fa8cf0af013c43"


def test_fixture_generation_repeats_byte_for_byte(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = generate_composed_media_retrieval_fixture(first_root)
    second = generate_composed_media_retrieval_fixture(second_root)
    canonical = load_composed_media_retrieval_fixture()
    canonical_tree = fixture_tree_sha256()
    canonical_size = fixture_tracked_bytes()

    assert first.fixture_sha256 == second.fixture_sha256 == canonical.fixture_sha256
    assert fixture_tree_sha256(first_root) == fixture_tree_sha256(second_root) == canonical_tree
    assert fixture_tracked_bytes(first_root) == fixture_tracked_bytes(second_root) == canonical_size
    assert canonical_json_bytes(fixture_to_dict(first)) == canonical_json_bytes(fixture_to_dict(canonical))
    assert canonical_json_bytes(fixture_to_dict(second)) == canonical_json_bytes(fixture_to_dict(canonical))
    first_files = {
        path.relative_to(first_root).as_posix(): path.read_bytes()
        for path in first_root.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second_root).as_posix(): path.read_bytes()
        for path in second_root.rglob("*")
        if path.is_file()
    }
    canonical_files = {
        path.relative_to(FIXTURE_ROOT).as_posix(): path.read_bytes()
        for path in FIXTURE_ROOT.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files == canonical_files
    shutil.rmtree(first_root)
    shutil.rmtree(second_root)
    assert not first_root.exists()
    assert not second_root.exists()


def test_composed_cache_identity_is_not_the_flat_cache_key() -> None:
    request = _request()
    reordered_item = replace(request.items[0], parts=tuple(reversed(request.items[0].parts)))
    reordered = replace(request, items=(reordered_item, *request.items[1:]), request_sha256="")
    reordered = replace(reordered, request_sha256=request_fingerprint(reordered))
    rerouted = replace(request, fusion_strategy="another_explicit_route", request_sha256="")
    rerouted = replace(rerouted, request_sha256=request_fingerprint(rerouted))

    assert request.request_sha256 != reordered.request_sha256
    assert request.request_sha256 != rerouted.request_sha256


@pytest.mark.parametrize(
    "corruption",
    [
        "reordered_parts",
        "non_contiguous_index",
        "duplicate_part_id",
        "wrong_item_attachment",
        "changed_mime",
        "changed_instruction",
        "mislabeled_route",
        "duplicate_item_id",
    ],
)
def test_request_validator_rejects_one_field_identity_corruption(corruption: str) -> None:
    request = _request()
    items = list(request.items)
    first = items[0]
    parts = list(first.parts)

    if corruption == "reordered_parts":
        first = replace(first, parts=tuple(reversed(parts)))
    elif corruption == "non_contiguous_index":
        first = replace(first, parts=(replace(parts[0], part_index=1), *parts[1:]))
    elif corruption == "duplicate_part_id":
        first = replace(first, parts=(parts[0], replace(parts[1], part_id=parts[0].part_id)))
    elif corruption == "wrong_item_attachment":
        first = replace(first, parts=(replace(parts[0], part_id="other_item:p00"), *parts[1:]))
    elif corruption == "changed_mime":
        first = replace(first, parts=(replace(parts[0], mime_type="text/html"), *parts[1:]))
    elif corruption == "changed_instruction":
        first = replace(first, instruction="Changed instruction.")
    elif corruption == "mislabeled_route":
        request = replace(request, track_label="benchmark_system_fusion")
    else:
        items[1] = replace(items[1], item_id=items[0].item_id)

    if corruption not in {"mislabeled_route", "duplicate_item_id"}:
        items[0] = first
    corrupted = replace(request, items=tuple(items))
    with pytest.raises(ValueError):
        validate_composed_request(corrupted, FIXTURE_ROOT)


@pytest.mark.parametrize("corruption", ["media_byte", "frame_order", "duration"])
def test_request_validator_rejects_media_corruption(tmp_path, corruption: str) -> None:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE_ROOT, copied)
    fixture = load_composed_media_retrieval_fixture(copied)
    request = build_composed_request(
        items=[query.item for query in fixture.queries],
        provider=DeterministicComposedMediaTestDouble.name,
        model_id=DeterministicComposedMediaTestDouble.model,
        model_revision=DeterministicComposedMediaTestDouble.model_revision,
        dimensions=16,
        task_route="composed_media_retrieval_query",
        preprocessing=fixture.generator_version,
        composition_mode="provider_native_fusion",
        track_label="provider_valid_embedding",
        fusion_strategy="deterministic_native_fixture_v0",
    )

    if corruption == "media_byte":
        path = copied / "stills/still_00.png"
        data = bytearray(path.read_bytes())
        data[-12] ^= 1
        path.write_bytes(data)
    else:
        path = copied / "videos/video_00/manifest.json"
        manifest = json.loads(path.read_bytes())
        if corruption == "frame_order":
            manifest["frames"][0], manifest["frames"][1] = manifest["frames"][1], manifest["frames"][0]
        else:
            manifest["duration_ms"] = 1999
        path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError):
        validate_composed_request(request, copied)


@pytest.mark.parametrize(
    "corruption",
    [
        "flattened",
        "reordered",
        "item_hash",
        "request_hash",
        "dimensions",
        "nan",
        "infinity",
        "zero",
        "track_label",
        "reranker_label",
        "route_secret",
    ],
)
def test_result_validator_rejects_one_field_corruption(corruption: str) -> None:
    request, result = _result()
    rows = list(result.rows)
    first = rows[0]

    if corruption == "flattened":
        result = replace(result, rows=(*result.rows, first))
    elif corruption == "reordered":
        result = replace(result, rows=(rows[1], rows[0], *rows[2:]))
    elif corruption == "item_hash":
        rows[0] = replace(first, item_sha256="0" * 64)
        result = replace(result, rows=tuple(rows))
    elif corruption == "request_hash":
        rows[0] = replace(first, request_sha256="0" * 64)
        result = replace(result, rows=tuple(rows))
    elif corruption == "dimensions":
        result = replace(result, dimensions=result.dimensions + 1)
    elif corruption == "nan":
        rows[0] = replace(first, embedding=(float("nan"), *first.embedding[1:]))
        result = replace(result, rows=tuple(rows))
    elif corruption == "infinity":
        rows[0] = replace(first, embedding=(float("inf"), *first.embedding[1:]))
        result = replace(result, rows=tuple(rows))
    elif corruption == "zero":
        rows[0] = replace(first, embedding=(0.0,) * result.dimensions)
        result = replace(result, rows=tuple(rows))
    elif corruption == "track_label":
        result = replace(result, track_label="benchmark_system_fusion")
    elif corruption == "reranker_label":
        result = replace(result, score_validity="reranker_system_only")
    else:
        result = replace(result, route_evidence={"api_key": "redacted-but-forbidden-field"})

    with pytest.raises(ValueError):
        validate_composed_result(request, result, FIXTURE_ROOT)


def test_system_fusion_cannot_be_labeled_provider_valid() -> None:
    request, result = _result(mode="benchmark_system_fusion")
    corrupted = replace(result, score_validity="provider_valid_embedding")
    with pytest.raises(ValueError, match="incompatible"):
        validate_composed_result(request, corrupted, FIXTURE_ROOT)


def test_request_and_result_serialization_repeat_exactly() -> None:
    request = _request()
    first = DeterministicComposedMediaTestDouble().embed_composed_media(request)
    second = DeterministicComposedMediaTestDouble().embed_composed_media(request)

    assert canonical_json_bytes(result_to_dict(first)) == canonical_json_bytes(result_to_dict(second))
    assert first.result_sha256 == second.result_sha256
    assert first.result_sha256 == result_fingerprint(first)


@pytest.mark.parametrize("corruption", ["duplicate", "unknown_query", "zero_grade", "missing_multi_positive"])
def test_fixture_rejects_qrel_corruption(corruption: str) -> None:
    fixture = load_composed_media_retrieval_fixture()
    qrels = list(fixture.qrels)
    if corruption == "duplicate":
        qrels.append(qrels[0])
    elif corruption == "unknown_query":
        qrels[0] = replace(qrels[0], query_id="unknown_query")
    elif corruption == "zero_grade":
        qrels[0] = replace(qrels[0], relevance=0)
    else:
        qrels.remove(next(qrel for qrel in qrels if qrel.relevance == 1))
    corrupted = _refingerprint_fixture(replace(fixture, qrels=tuple(qrels)))
    with pytest.raises(ValueError):
        validate_composed_media_retrieval_fixture(corrupted)


def test_utf8_tie_breaking_is_exact_and_repeatable() -> None:
    corpus_ids = ["éclair", "zeta", "alpha"]
    scores = np.asarray([0.5, 0.5, 0.5])
    first = stable_corpus_ranking(corpus_ids, scores)
    second = stable_corpus_ranking(corpus_ids, scores)
    assert [corpus_ids[index] for index in first] == ["alpha", "zeta", "éclair"]
    assert first == second


def test_task_runs_full_corpus_hard_pool_slices_and_paired_deltas() -> None:
    provider = DeterministicComposedMediaTestDouble()
    result = get_task("composed_media_retrieval").run(provider)

    assert result.passed, result.error
    assert len(provider.requests) == 4
    assert result.metrics["composed_ndcg@10"] == pytest.approx(1.0)
    assert result.metrics["composed_map@5"] == pytest.approx(1.0)
    assert result.metrics["composed_recall@1"] == pytest.approx(5 / 6)
    assert result.metrics["hard_negative_outrank_rate"] == pytest.approx(0.0)
    assert result.metrics["system_hard_negative_outrank_rate"] == pytest.approx(1 / 3)
    assert result.metrics["native_fusion_gain_ndcg@10"] > 0
    assert result.metrics["native_fusion_gain_map@5"] > 0
    assert result.metrics["native_fusion_gain_recall@1"] > 0
    assert result.details["full_corpus_metrics_authoritative"] is True
    assert result.details["labels"] == {
        "provider_valid": "provider_valid_embedding",
        "system_fusion": "benchmark_system_fusion",
        "reranker": "reranker_system_only",
        "fixture": "contract_fixture_only",
    }
    required_slices = {
        "query_shape",
        "part_count",
        "positive_cardinality",
        "target_modality",
        "reference_duration_bucket",
        "hard_negative_family",
        "fusion_track",
        "order_sensitive_family",
        "media_reuse",
    }
    assert required_slices == set(result.details["slices"]["provider_native_fusion"])
    assert result.details["n_positive_qrels"] == 16
    assert result.details["n_hard_negatives"] == 36
    assert result.details["network"] == "forbidden"
    assert result.details["provider_api_calls"] == 0
    assert result.details["hugging_face_operations"] == 0


def test_task_rejects_flattened_or_reordered_provider_output() -> None:
    for corruption in ("flattened", "reordered", "zero_vector"):
        result = get_task("composed_media_retrieval").run(
            DeterministicComposedMediaTestDouble(corrupt_system_result=corruption)
        )
        assert not result.passed


def test_paired_deltas_reject_unpairable_ids_preprocessing_and_labels() -> None:
    native = _evaluation("provider_native_fusion")
    system = _evaluation("benchmark_system_fusion")
    assert paired_fusion_deltas(native, system)["native_fusion_gain_ndcg@10"] > 0

    with pytest.raises(ValueError, match="ordered query/corpus ids"):
        paired_fusion_deltas(native, replace(system, query_ids=tuple(reversed(system.query_ids))))
    with pytest.raises(ValueError, match="preprocessing"):
        paired_fusion_deltas(native, replace(system, preprocessing="different-preprocessing"))
    with pytest.raises(ValueError, match="tracks"):
        paired_fusion_deltas(native, replace(system, track_label="provider_valid_embedding"))


def test_task_runtime_is_zero_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("network access is forbidden for the composed-media fixture")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    result = get_task("composed_media_retrieval").run(DeterministicComposedMediaTestDouble())
    assert result.passed, result.error
    assert result.details["network"] == "forbidden"


def test_registry_manifest_and_hf_exports_exclude_fixture(tmp_path) -> None:
    catalog = load_catalog()
    spec = catalog.tasks["composed_media_retrieval"]
    manifest = load_run_manifest("benchmark/runs/composed-media-retrieval-local-smoke.yaml")
    assert spec.primary_metric == "composed_ndcg@10"
    assert spec.dataset_version == DATASET_VERSION
    assert spec.publish is False
    assert spec.leaderboard_publish is False
    assert {"fixture-only", "no-publish", "composed-media"}.issubset(spec.tags)
    assert manifest.publish is False
    assert manifest.model_ids == []
    assert manifest.metadata["leaderboard_publish"] is False
    assert manifest.metadata["hugging_face_operations"] == 0

    record = {
        "run": {"id": manifest.id, "publish": False, "evidence_tier": "fixture"},
        "timestamps": {"duration_s": 0.0},
        "model": {"id": "fixture-model", "display_name": "Fixture", "provider": "local"},
        "provider_result": {"provider": "deterministic-composed-media-local", "model_name": "fixture"},
        "task": {
            "id": "composed_media_retrieval",
            "display_name": spec.display_name,
            "primary_metric": spec.primary_metric,
            "publish": False,
        },
        "metrics": {"composed_ndcg@10": 1.0},
        "error": None,
    }
    assert build_leaderboard([record], catalog) == []

    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(record) + "\n", encoding="utf-8")
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s\n"
        "composed_media_retrieval,Fixture,fixture-model,Fixture,local,composed_ndcg@10,1.0,fixture,0.0\n",
        encoding="utf-8",
    )
    dataset = export_dataset_repo(
        output_dir=tmp_path / "dataset",
        results_path=results,
        leaderboard_path=leaderboard,
    )
    space = export_space_repo(output_dir=tmp_path / "space", bundled_leaderboard=leaderboard)
    dataset_text = "\n".join(path.read_text(encoding="utf-8") for path in dataset.rglob("*") if path.is_file())
    assert "composed_media_retrieval" not in dataset_text
    assert "composed_media_retrieval" not in (space / "leaderboard.csv").read_text(encoding="utf-8")
    assert not (dataset / "runs" / "composed-media-retrieval-local-smoke.yaml").exists()
