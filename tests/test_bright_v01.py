from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from mm_embed.benchmark import bright_v01
from mm_embed.hf_publish.export import _copy_benchmark_data


def test_download_budget_is_bounded() -> None:
    total = sum(
        bright_v01.EXPECTED[track][role]["bytes"]
        for track in bright_v01.TRACKS
        for role in bright_v01.ROLES
    )
    assert total == 34_050_261
    assert total < bright_v01.MAX_TOTAL_DOWNLOAD_BYTES
    assert max(
        bright_v01.EXPECTED[track][role]["bytes"]
        for track in bright_v01.TRACKS
        for role in bright_v01.ROLES
    ) < bright_v01.MAX_FILE_BYTES


def test_pinned_urls_and_remote_code_policy() -> None:
    assert bright_v01.HF_REVISION in bright_v01.source_url("biology", "documents")
    with pytest.raises(bright_v01.BrightMaterializationError):
        bright_v01.source_url("leetcode", "documents")


def test_load_track_rejects_label_contract_violations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    documents = [{"id": "d1", "content": "alpha"}]
    base = {
        "id": "q1",
        "query": "question",
        "gold_ids": ["d1"],
        "excluded_ids": [],
        "gold_ids_long": ["long-1"],
    }

    class FakeTable:
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def to_pylist(self) -> list[dict]:
            return self.rows

    monkeypatch.setattr(bright_v01, "preflight_track", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(bright_v01, "_parquet_file", lambda *_args: tmp_path / "source.parquet")
    import pyarrow.parquet as pq

    for mutation, message in (
        (lambda row: row.__setitem__("gold_ids", []), "empty gold"),
        (lambda row: row.__setitem__("gold_ids", ["missing"]), "dangling passage"),
        (lambda row: row.__setitem__("excluded_ids", ["missing"]), "dangling passage"),
        (lambda row: row.__setitem__("gold_ids", ["d1", "d1"]), "duplicate gold_ids"),
    ):
        example = dict(base)
        mutation(example)
        tables = iter((FakeTable(documents), FakeTable([example])))
        monkeypatch.setattr(pq, "read_table", lambda *_args, **_kwargs: next(tables))
        with pytest.raises(bright_v01.BrightMaterializationError, match=message):
            bright_v01.load_track(tmp_path, "biology")


def test_materialization_emits_only_positive_qrels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = bright_v01.TrackRows(
        "biology",
        ({"id": "d1", "content": "alpha"}, {"id": "d2", "content": "beta"}),
        ({"id": "q1", "query": "question", "gold_ids": ["d1"], "excluded_ids": ["d2"], "gold_ids_long": ["long-1"]},),
    )
    monkeypatch.setattr(bright_v01, "load_track", lambda *_args, **_kwargs: rows)
    real_validate = bright_v01.validate_materialization
    monkeypatch.setattr(bright_v01, "validate_materialization", lambda *_args: {})
    output = tmp_path / "out"
    manifest = bright_v01.materialize_track(tmp_path, output)
    monkeypatch.setattr(bright_v01, "validate_materialization", real_validate)
    monkeypatch.setitem(
        bright_v01.CANONICAL_MATERIALIZATION,
        "biology",
        manifest["materialization"]["files"],
    )
    assert (output / "qrels/test.tsv").read_text() == "query-id\tcorpus-id\tscore\nq1\td1\t1\n"
    query = json.loads((output / "queries.jsonl").read_text())
    assert query["excluded_ids"] == ["d2"]
    assert query["gold_ids_long"] == ["long-1"]
    assert manifest["publication"]["public_export_allowed"] is False
    schema = json.loads(Path("schemas/bright-materialization-v01.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    assert bright_v01.validate_materialization(output) == manifest


def _fixture_materialization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    rows = bright_v01.TrackRows(
        "biology",
        ({"id": "d1", "content": "alpha"}, {"id": "d2", "content": "beta"}),
        ({"id": "q1", "query": "question", "gold_ids": ["d1"], "excluded_ids": ["d2"], "gold_ids_long": ["long-1"]},),
    )
    monkeypatch.setattr(bright_v01, "load_track", lambda *_args, **_kwargs: rows)
    real_validate = bright_v01.validate_materialization
    monkeypatch.setattr(bright_v01, "validate_materialization", lambda *_args: {})
    output = tmp_path / "data" / bright_v01.BENCHMARK_VERSION / "biology"
    manifest = bright_v01.materialize_track(tmp_path, output)
    monkeypatch.setattr(bright_v01, "validate_materialization", real_validate)
    monkeypatch.setitem(
        bright_v01.CANONICAL_MATERIALIZATION,
        "biology",
        manifest["materialization"]["files"],
    )
    return output


def test_complete_materialization_is_limited_to_biology(tmp_path: Path) -> None:
    with pytest.raises(bright_v01.BrightMaterializationError, match="not approved"):
        bright_v01.materialize_track(tmp_path, tmp_path / "economics", "economics")


@pytest.mark.parametrize("field", ["path", "rows", "bytes", "sha256"])
def test_validate_materialization_rejects_changed_canonical_entry(field: str) -> None:
    output = Path("data") / bright_v01.BENCHMARK_VERSION / "biology"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["materialization"]["files"]["corpus.jsonl"]
    if field == "path":
        entry[field] = "queries.jsonl"
    elif field == "sha256":
        entry[field] = "0" * 64
    else:
        entry[field] += 1
    original = manifest_path.read_bytes()
    try:
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(bright_v01.BrightMaterializationError):
            bright_v01.validate_materialization(output)
    finally:
        manifest_path.write_bytes(original)


def test_validate_materialization_rejects_fully_resigned_replacement(tmp_path: Path) -> None:
    source = Path("data") / bright_v01.BENCHMARK_VERSION / "biology"
    output = tmp_path / "biology"
    import shutil

    shutil.copytree(source, output)
    corpus = {"id": "fake-document", "content": "internally consistent biology replacement"}
    query = {"id": "fake-query", "text": "fake question", "excluded_ids": [], "gold_ids_long": []}
    (output / "corpus.jsonl").write_bytes(bright_v01._canonical_json(corpus))
    (output / "queries.jsonl").write_bytes(bright_v01._canonical_json(query))
    (output / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nfake-query\tfake-document\t1\n",
        encoding="utf-8",
    )
    rows = bright_v01.TrackRows(
        "biology",
        (corpus,),
        ({**query, "query": query["text"], "gold_ids": [corpus["id"]]},),
    )
    audit_query = dict(rows.examples[0])
    audit_query.pop("text")
    (output / "audit.json").write_bytes(
        bright_v01._canonical_json(
            bright_v01.audit_track(
                bright_v01.TrackRows("biology", rows.documents, (audit_query,))
            )
        )
    )
    for relative in bright_v01.MATERIALIZED_FILE_KEYS:
        _resign_materialized_file(output, relative)
    with pytest.raises(bright_v01.BrightMaterializationError, match="pinned BRIGHT contract"):
        bright_v01.validate_materialization(output)


def _resign_materialized_file(output: Path, relative: str) -> None:
    path = output / relative
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["materialization"]["files"][relative]
    entry["bytes"] = path.stat().st_size
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "rows" in entry:
        if relative == "qrels/test.tsv":
            entry["rows"] = len(path.read_text().splitlines()) - 1
        else:
            entry["rows"] = len(path.read_text().splitlines())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["source"]["files"][0].pop("sha256"),
        lambda value: value["source"]["files"][0].__setitem__("sha256", "0" * 64),
        lambda value: value["source"]["files"][0].__setitem__("url", "https://example.invalid/source.parquet"),
        lambda value: value.__setitem__("materialization", {}),
        lambda value: value.__setitem__("runner_adapter", {}),
        lambda value: value["publication"].__setitem__("extra", True),
    ],
)
def test_schema_rejects_contract_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation,
) -> None:
    output = _fixture_materialization(monkeypatch, tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    mutation(manifest)
    schema = json.loads(Path("schemas/bright-materialization-v01.schema.json").read_text())
    assert list(Draft202012Validator(schema).iter_errors(manifest))


@pytest.mark.parametrize("target", ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv", "audit.json"])
def test_validate_materialization_rejects_tampered_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
) -> None:
    output = _fixture_materialization(monkeypatch, tmp_path)
    with (output / target).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(bright_v01.BrightMaterializationError):
        bright_v01.validate_materialization(output)


@pytest.mark.parametrize("mutation", ["manifest_hash", "source_identity", "unsafe_path", "extra_file"])
def test_validate_materialization_rejects_manifest_and_tree_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    output = _fixture_materialization(monkeypatch, tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "manifest_hash":
        manifest["materialization"]["files"]["corpus.jsonl"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "source_identity":
        manifest["source"]["files"][0]["bytes"] += 1
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "unsafe_path":
        manifest["materialization"]["files"]["corpus.jsonl"]["path"] = "../corpus.jsonl"
        manifest_path.write_text(json.dumps(manifest))
    else:
        (output / "extra.txt").write_text("unexpected")
    with pytest.raises(bright_v01.BrightMaterializationError):
        bright_v01.validate_materialization(output)


@pytest.mark.parametrize("relative_root", ["biology", "source", ".", "..", "benchmark_data"])
def test_export_rejects_restricted_materialization_from_all_related_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_root: str,
) -> None:
    materialized = _fixture_materialization(monkeypatch, tmp_path)
    dataset_root = materialized.parent
    source = dataset_root / "source" / "documents"
    source.mkdir(parents=True)
    (source / "biology-00000-of-00001.parquet").write_bytes(b"source")
    roots = {
        "biology": materialized,
        "source": dataset_root / "source",
        ".": dataset_root,
        "..": dataset_root.parent,
    }
    if relative_root == "benchmark_data":
        benchmark_data = tmp_path / "benchmark_data"
        benchmark_data.mkdir()
        dataset_root.rename(benchmark_data / dataset_root.name)
        roots[relative_root] = benchmark_data
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(roots[relative_root], tmp_path / "export", include_images=False)


def test_export_rejects_invalid_restricted_manifest_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = _fixture_materialization(monkeypatch, tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(bright_v01.BrightMaterializationError):
        bright_v01.validate_materialization(output)
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(output.parent.parent, tmp_path / "export", include_images=False)


@pytest.mark.parametrize("manifest_mutation", ["empty", "invalid_json", "missing_version", "missing", "renamed", "extra"])
def test_export_manifest_discovery_cannot_be_bypassed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest_mutation: str,
) -> None:
    output = _fixture_materialization(monkeypatch, tmp_path)
    manifest_path = output / "manifest.json"
    if manifest_mutation == "empty":
        manifest_path.write_text("{}", encoding="utf-8")
    elif manifest_mutation == "invalid_json":
        manifest_path.write_text("{", encoding="utf-8")
    elif manifest_mutation == "missing_version":
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("benchmark_version")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif manifest_mutation == "missing":
        manifest_path.unlink()
    elif manifest_mutation == "renamed":
        manifest_path.rename(output / "metadata.json")
    else:
        (output / "unrecognized-manifest.json").write_text("{}", encoding="utf-8")
        manifest_path.unlink()
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(output.parent.parent, tmp_path / "export", include_images=False)


def test_export_allows_public_data_without_restricted_markers(tmp_path: Path) -> None:
    public = tmp_path / "public_data"
    public.mkdir()
    (public / "corpus.jsonl").write_text('{"id":"safe","content":"public"}\n', encoding="utf-8")
    _copy_benchmark_data(public, tmp_path / "export", include_images=False)
    assert (tmp_path / "export" / "corpus.jsonl").is_file()


@pytest.mark.parametrize(
    "mutation",
    [
        "invalid_jsonl",
        "missing_field",
        "extra_field",
        "duplicate_corpus_id",
        "duplicate_query_id",
        "dangling_excluded",
        "dangling_qrel",
        "grade_zero",
        "malformed_tsv",
        "duplicate_pair",
        "audit_forgery",
    ],
)
def test_validate_materialization_rejects_resigned_invalid_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    output = _fixture_materialization(monkeypatch, tmp_path)
    target = "corpus.jsonl"
    if mutation == "invalid_jsonl":
        (output / target).write_text("{\n", encoding="utf-8")
    elif mutation in {"missing_field", "extra_field"}:
        row = json.loads((output / target).read_text().splitlines()[0])
        row.pop("content", None)
        if mutation == "extra_field":
            row["content"] = "alpha"
            row["extra"] = True
        (output / target).write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif mutation == "duplicate_corpus_id":
        rows = [json.loads(line) for line in (output / target).read_text().splitlines()]
        rows[1]["id"] = rows[0]["id"]
        (output / target).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    elif mutation in {"duplicate_query_id", "dangling_excluded"}:
        target = "queries.jsonl"
        row = json.loads((output / target).read_text())
        if mutation == "duplicate_query_id":
            (output / target).write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        else:
            row["excluded_ids"] = ["missing"]
            (output / target).write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif mutation in {"dangling_qrel", "grade_zero", "malformed_tsv", "duplicate_pair"}:
        target = "qrels/test.tsv"
        header = "query-id\tcorpus-id\tscore\n"
        row = "q1\td1\t1\n"
        replacements = {
            "dangling_qrel": "q1\tmissing\t1\n",
            "grade_zero": "q1\td1\t0\n",
            "malformed_tsv": "q1\td1\n",
            "duplicate_pair": row + row,
        }
        (output / target).write_text(header + replacements[mutation], encoding="utf-8")
    else:
        target = "audit.json"
        audit = json.loads((output / target).read_text())
        audit["documents"] += 1
        (output / target).write_text(json.dumps(audit) + "\n", encoding="utf-8")
    _resign_materialized_file(output, target)
    with pytest.raises(bright_v01.BrightMaterializationError):
        bright_v01.validate_materialization(output)


def test_excluded_sentinel_is_not_treated_as_a_passage_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    documents = [{"id": "d1", "content": "alpha"}]
    example = {
        "id": "q1",
        "query": "question",
        "gold_ids": ["d1"],
        "excluded_ids": ["N/A"],
        "gold_ids_long": ["long-1"],
    }

    class FakeTable:
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def to_pylist(self) -> list[dict]:
            return self.rows

    monkeypatch.setattr(bright_v01, "preflight_track", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(bright_v01, "_parquet_file", lambda *_args: tmp_path / "source.parquet")
    import pyarrow.parquet as pq

    tables = iter((FakeTable(documents), FakeTable([example])))
    monkeypatch.setattr(pq, "read_table", lambda *_args, **_kwargs: next(tables))
    assert bright_v01.load_track(tmp_path, "biology").examples[0]["excluded_ids"] == []
