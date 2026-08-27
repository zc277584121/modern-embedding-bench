import copy
import hashlib
import json
import os
import shutil
import socket
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from mm_embed.benchmark import bright_multidomain_v02 as pilot
from mm_embed.benchmark import bright_v01
from mm_embed.benchmark.retrieval_v01 import (
    TrackData,
    bm25_rank,
    bootstrap_confidence_intervals,
    query_metrics,
    validate_track_data,
)
from mm_embed.hf_publish import export as hf_export
from mm_embed.hf_publish.export import (
    _BenchmarkCopyLimits,
    _copy_benchmark_data,
    _reject_restricted_materializations,
)


DATA_ROOT = Path("data/bright-nontechnical-pilot-v0.2")
SOURCE_ROOT = Path("data/bright-nontechnical-pilot-v0.1/source")
RESULTS_ROOT = Path("results/bright-nontechnical-pilot-v0.2")
AUDIT_ROOT = RESULTS_ROOT / "audit"
SAFE_ROOT = Path("benchmark/artifacts/bright-nontechnical-pilot-v0.2")
MATERIALIZATION_SCHEMA = json.loads(Path("schemas/bright-multidomain-materialization-v02.schema.json").read_text())
RESULT_SCHEMA = json.loads(Path("schemas/bright-multidomain-result-v02.schema.json").read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_materialization(tmp_path: Path) -> Path:
    output = tmp_path / pilot.BENCHMARK_VERSION
    shutil.copytree(DATA_ROOT, output)
    return output


def _copy_result_artifacts(tmp_path: Path, method: str = "long_dense", track: str = "economics") -> Path:
    output = tmp_path / "results"
    output.mkdir()
    for suffix in ("json", "rankings.jsonl", "per-query.jsonl"):
        name = f"{method}-{track}.{suffix}"
        shutil.copy2(RESULTS_ROOT / name, output / name)
    return output


def _rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(pilot._json_bytes(row) for row in rows))


def _resign_artifact(result_root: Path, result: dict, artifact_key: str) -> None:
    artifact = result[artifact_key]
    path = result_root / artifact["path"]
    artifact["bytes"] = path.stat().st_size
    artifact["sha256"] = _sha256(path)
    result_path = result_root / f"{result['method']}-{result['track']}.json"
    result_path.write_bytes(pilot._json_bytes(result))


@pytest.fixture(scope="module")
def validated_economics_binding() -> tuple[dict, pilot.TrackData]:
    manifest = pilot.validate_materialization(DATA_ROOT)
    return manifest, pilot._load_track_from_files(DATA_ROOT, "economics").data


def test_biology_v01_contract_remains_closed() -> None:
    assert bright_v01.MATERIALIZABLE_TRACKS == ("biology",)
    assert pilot.TRACKS == ("economics", "psychology")
    assert pilot.BENCHMARK_VERSION != bright_v01.BENCHMARK_VERSION
    assert json.loads(Path("schemas/bright-materialization-v01.schema.json").read_text())["properties"]["track"] == {
        "const": "biology"
    }


def test_fixed_materialization_is_valid_and_source_derived() -> None:
    Draft202012Validator.check_schema(MATERIALIZATION_SCHEMA)
    manifest = pilot.validate_materialization(DATA_ROOT, source_root=SOURCE_ROOT)
    Draft202012Validator(MATERIALIZATION_SCHEMA).validate(manifest)
    assert manifest["aggregate"] == {
        "tracks": 2,
        "documents": 15_000,
        "queries": 204,
        "positive_qrels": 1_492,
        "cross_track_shared_exact_content_values": 25,
    }
    assert manifest["publication"]["public_export_allowed"] is False
    assert _sha256(DATA_ROOT / "manifest.json") == pilot.CANONICAL_MANIFEST_SHA256


@pytest.mark.parametrize("track", pilot.TRACKS)
def test_candidate_selection_keeps_every_gold_and_exact_content_is_unique(track: str) -> None:
    data = pilot.load_materialized(DATA_ROOT, track)
    corpus_ids = {row["id"] for row in data.corpus}
    contents = [row["content"] for row in data.corpus]
    gold_ids = {doc_id for rels in data.qrels.values() for doc_id in rels}
    assert len(data.corpus) == 7_500
    assert gold_ids <= corpus_ids
    assert len(contents) == len(set(contents))
    assert all(grade == 1 for rels in data.qrels.values() for grade in rels.values())
    assert all(not row["excluded_ids"] for row in data.queries)


def test_review_plan_is_fixed_before_model_runs_and_stratified() -> None:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    rows = pilot._read_jsonl(DATA_ROOT / manifest["review_plan"]["path"])
    assert len(rows) == 120
    assert Counter(row["track"] for row in rows) == Counter({"economics": 60, "psychology": 60})
    assert len({(row["track"], row["query_id"]) for row in rows}) == 120
    assert all(row["selection"]["fixed_before_model_runs"] for row in rows)
    assert all(row["selection"]["model_score_used"] is False for row in rows)
    assert all(set(row["strata"]) == {"query_length", "qrel_density", "gold_ids_long"} for row in rows)
    assert manifest["review_plan"]["sha256"] == pilot.CANONICAL_REVIEW_PLAN["sha256"]


def test_long_dense_cap_is_fixed_from_tokenizer_evidence_without_scores() -> None:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    contract = manifest["long_dense"]
    assert contract["model"]["repo_id"] == "BAAI/bge-m3"
    assert contract["model"]["revision"] == pilot.LONG_DENSE_MODEL_REVISION
    assert contract["model"]["aggregate_sha256"] == pilot.LONG_DENSE_SNAPSHOT["aggregate_sha256"]
    assert contract["cap_selection"]["selected_cap"] == 1_024
    assert contract["cap_selection"]["score_used"] is False
    assert contract["cap_selection"]["fixed_before_model_scoring"] is True
    assert hashlib.sha256(pilot._json_bytes(contract)).hexdigest() == pilot.LONG_DENSE_CONTRACT_SHA256
    assert contract["cap_selection"]["tracks"]["economics"]["gold_documents"]["by_cap"]["1024"] == {
        "truncated": 37,
        "rate": 37 / 800,
    }
    assert contract["cap_selection"]["tracks"]["psychology"]["gold_documents"]["by_cap"]["1024"] == {
        "truncated": 63,
        "rate": 63 / 688,
    }
    assert all(
        contract["cap_selection"]["tracks"][track]["gold_documents"]["by_cap"]["1024"]["rate"] <= 0.10
        for track in pilot.TRACKS
    )


def test_sensitive_evidence_contains_only_safe_non_reversible_fields() -> None:
    expected_rows = {"economics": 164, "psychology": 87}
    for track in pilot.TRACKS:
        audit = json.loads((DATA_ROOT / track / "audit.json").read_text())
        evidence = pilot._read_jsonl(DATA_ROOT / track / "sensitive-evidence.jsonl")
        assert len(evidence) == expected_rows[track]
        assert audit["sensitive_information"]["raw_matches_or_context_stored"] is False
        for row in evidence:
            assert set(row) == {
                "track",
                "role",
                "record_id_sha256",
                "pattern",
                "classification",
                "ordinal",
                "match_sha256",
                "context_sha256",
                "safe_example",
                "raw_text_included",
                "requires_manual_review",
            }
            assert row["raw_text_included"] is False
            assert row["safe_example"] in {"***@<classified-domain>", "x.x.x.x"}
            assert "@" not in row["match_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["source"].__setitem__("revision", "main"),
        lambda value: value["aggregate"].__setitem__("documents", 14_999),
        lambda value: value["selection"].__setitem__("model_score_used", True),
        lambda value: value["publication"].__setitem__("public_export_allowed", True),
        lambda value: value["tracks"]["economics"].__setitem__("selection_ids_sha256", "0" * 64),
        lambda value: value["review_plan"].__setitem__("rows", 119),
        lambda value: value["long_dense"]["cap_selection"].__setitem__("selected_cap", 512),
        lambda value: value["long_dense"]["cap_selection"].__setitem__("score_used", True),
        lambda value: value["long_dense"]["model"].__setitem__("revision", "main"),
        lambda value: value.__setitem__("extra", True),
    ],
)
def test_materialization_schema_rejects_contract_mutations(mutation) -> None:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    mutation(manifest)
    assert not Draft202012Validator(MATERIALIZATION_SCHEMA).is_valid(manifest)


@pytest.mark.parametrize("mutation", ["corpus", "selection", "review-plan", "extra-file"])
def test_validator_rejects_resigned_or_extra_materialized_content(tmp_path: Path, mutation: str) -> None:
    output = _copy_materialization(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "extra-file":
        (output / "extra.json").write_text("{}\n", encoding="utf-8")
    else:
        if mutation == "review-plan":
            relative = Path("review-plan-120.jsonl")
            entry = manifest["review_plan"]
        else:
            relative = Path("economics") / f"{mutation}.jsonl"
            entry = manifest["tracks"]["economics"]["files"][f"{mutation}.jsonl"]
        path = output / relative
        rows = path.read_text(encoding="utf-8").splitlines()
        row = json.loads(rows[0])
        row[next(iter(row))] = "changed"
        rows[0] = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = _sha256(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(pilot.BrightPilotError):
        pilot.validate_materialization(output)


def test_export_gate_rejects_multidomain_root_and_track() -> None:
    with pytest.raises(ValueError, match="public export denied"):
        _reject_restricted_materializations(DATA_ROOT)
    with pytest.raises(ValueError, match="public export denied"):
        _reject_restricted_materializations(DATA_ROOT / "economics")


def test_export_gate_allows_commit_safe_aggregate_artifacts() -> None:
    _reject_restricted_materializations(SAFE_ROOT)


@pytest.mark.parametrize("restricted_root", [RESULTS_ROOT, AUDIT_ROOT])
def test_real_export_copy_rejects_research_only_result_and_audit_roots(
    tmp_path: Path,
    restricted_root: Path,
) -> None:
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(restricted_root, output, include_images=False)
    assert not output.exists()


def test_real_export_copy_allows_only_safe_aggregate_root(tmp_path: Path) -> None:
    output = tmp_path / "export"
    _copy_benchmark_data(SAFE_ROOT, output, include_images=False)
    assert {path.name for path in output.iterdir()} == {"manifest.json", "summary.json", "README.md"}
    assert (output / "manifest.json").read_bytes() == (SAFE_ROOT / "manifest.json").read_bytes()
    assert (output / "summary.json").read_bytes() == (SAFE_ROOT / "summary.json").read_bytes()


def test_real_export_copy_rejects_extra_text_in_safe_aggregate_tree(tmp_path: Path) -> None:
    attack = tmp_path / "safe-aggregate"
    shutil.copytree(SAFE_ROOT, attack)
    (attack / "notes.txt").write_text("restricted body text\n", encoding="utf-8")
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(attack, output, include_images=False)
    assert not output.exists()


@pytest.mark.parametrize("extra_kind", ["symlink", "directory", "special", "nested"])
def test_safe_aggregate_tree_allowlist_rejects_non_files_and_nested_content(
    tmp_path: Path,
    extra_kind: str,
) -> None:
    attack = tmp_path / "safe-aggregate"
    shutil.copytree(SAFE_ROOT, attack)
    if extra_kind == "symlink":
        (attack / "extra-link").symlink_to(attack / "summary.json")
    elif extra_kind == "directory":
        (attack / "extra-dir").mkdir()
    elif extra_kind == "special":
        os.mkfifo(attack / "extra-pipe")
    else:
        nested = attack / "extra-dir" / "level-two"
        nested.mkdir(parents=True)
        (nested / "notes.txt").write_text("restricted body text\n", encoding="utf-8")
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(attack, output, include_images=False)
    assert not output.exists()


def test_safe_aggregate_tree_rejects_allowed_name_replaced_by_symlink(tmp_path: Path) -> None:
    attack = tmp_path / "safe-aggregate"
    shutil.copytree(SAFE_ROOT, attack)
    external_manifest = tmp_path / "manifest.json"
    shutil.copy2(attack / "manifest.json", external_manifest)
    (attack / "manifest.json").unlink()
    (attack / "manifest.json").symlink_to(external_manifest)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(attack, output, include_images=False)
    assert not output.exists()


@pytest.mark.parametrize("prefix", [b"", b"{}\n", b"\n\n", b"\n\n{}\n\n"])
def test_real_export_copy_scans_all_rows_of_prefixed_renamed_rankings(
    tmp_path: Path,
    prefix: bytes,
) -> None:
    attack = tmp_path / "neutral" / "level-one" / "level-two"
    attack.mkdir(parents=True)
    payload = attack / "payload.bin"
    payload.write_bytes(prefix + (RESULTS_ROOT / "long_dense-economics.rankings.jsonl").read_bytes())
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(tmp_path / "neutral", output, include_images=False)
    assert not output.exists()


@pytest.mark.parametrize(
    "prefix",
    [
        b"not-json\n",
        b"a" * (1024 * 1024 + 1) + b"\n",
    ],
    ids=["plain-text", "overlong-plain-text"],
)
def test_real_export_copy_rejects_plain_text_prefix_before_rankings(
    tmp_path: Path,
    prefix: bytes,
) -> None:
    attack = tmp_path / "neutral"
    attack.mkdir()
    payload = attack / "payload.bin"
    payload.write_bytes(prefix + (RESULTS_ROOT / "long_dense-economics.rankings.jsonl").read_bytes())
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(attack, output, include_images=False)
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "corpus.jsonl",
        "queries.jsonl",
        "qrels.jsonl",
        "selection.jsonl",
        "sensitive-evidence.jsonl",
    ],
)
def test_real_export_copy_rejects_renamed_v02_materialization_rows(
    tmp_path: Path,
    name: str,
) -> None:
    attack = tmp_path / "neutral"
    attack.mkdir()
    shutil.copy2(DATA_ROOT / "economics" / name, attack / "payload.bin")
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(attack, output, include_images=False)
    assert not output.exists()


@pytest.mark.parametrize(
    "variant",
    ["utf8-bom-per-line", "utf16", "invalid-utf8", "array", "wrapper", "nul-per-line"],
)
def test_real_export_copy_rejects_encoded_or_nested_rankings(
    tmp_path: Path,
    variant: str,
) -> None:
    source = (RESULTS_ROOT / "long_dense-economics.rankings.jsonl").read_bytes()
    lines = source.splitlines()
    if variant == "utf8-bom-per-line":
        payload = b"".join(b"\xef\xbb\xbf" + line + b"\n" for line in lines)
    elif variant == "utf16":
        payload = source.decode("utf-8").encode("utf-16")
    elif variant == "invalid-utf8":
        payload = b"\xff" + source
    elif variant == "array":
        payload = b"".join(b"[" + line + b"]\n" for line in lines)
    elif variant == "wrapper":
        payload = b"".join(b'{"row":' + line + b"}\n" for line in lines)
    else:
        payload = b"".join(b"\x00" + line + b"\n" for line in lines)

    attack = tmp_path / "neutral"
    attack.mkdir()
    (attack / "payload.bin").write_bytes(payload)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(attack, output, include_images=False)
    assert not output.exists()


@pytest.mark.parametrize("entry_kind", ["symlink", "fifo", "unix-socket"])
def test_real_export_copy_preflight_rejects_non_regular_entries(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    attack = tmp_path / "neutral"
    attack.mkdir()
    safe_file = attack / "safe.jsonl"
    safe_file.write_text('{"id":"safe","content":"public"}\n', encoding="utf-8")
    special = attack / "special"
    unix_socket = None
    if entry_kind == "symlink":
        special.symlink_to(safe_file)
    elif entry_kind == "fifo":
        os.mkfifo(special)
    else:
        unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix_socket.bind(str(special))

    output = tmp_path / "export"
    try:
        with pytest.raises(ValueError, match="public export denied"):
            _copy_benchmark_data(attack, output, include_images=False)
    finally:
        if unix_socket is not None:
            unix_socket.close()
    assert not output.exists()


@pytest.mark.parametrize(
    "payload",
    [
        b'[{"id":"safe","content":"public"}]\n',
        b'{"row":{"id":"safe","content":"public"}}\n',
    ],
    ids=["array", "wrapper"],
)
def test_real_export_copy_allows_safe_nested_public_json(tmp_path: Path, payload: bytes) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "payload.data").write_bytes(payload)
    output = tmp_path / "export"
    _copy_benchmark_data(public, output, include_images=False)
    assert (output / "payload.data").read_bytes() == payload


@pytest.mark.parametrize("mutation", ["modify", "replace"])
def test_real_export_copy_rejects_source_changes_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source_root = tmp_path / "public"
    source_root.mkdir()
    source = source_root / "payload.jsonl"
    original = b'{"id":"safe","content":"public"}\n'
    replacement = b'{"id":"evil","content":"public"}\n'
    assert len(original) == len(replacement)
    source.write_bytes(original)
    original_stage = hf_export._stage_preflighted_benchmark_files
    hook_called = False

    def mutate_after_preflight(plan, staging) -> None:
        nonlocal hook_called
        hook_called = True
        if mutation == "modify":
            source.write_bytes(replacement)
        else:
            alternate = tmp_path / "replacement.jsonl"
            alternate.write_bytes(replacement)
            alternate.replace(source)
        original_stage(plan, staging)

    monkeypatch.setattr(hf_export, "_stage_preflighted_benchmark_files", mutate_after_preflight)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(source_root, output, include_images=False)
    assert hook_called
    assert not output.exists()
    assert not list(tmp_path.glob(".export.staging-*"))


@pytest.mark.parametrize("resource", ["depth", "entries", "aggregate-bytes", "scan-bytes"])
@pytest.mark.parametrize("over_limit", [False, True], ids=["at-limit", "over-limit"])
def test_real_export_copy_enforces_tree_resource_limits(
    tmp_path: Path,
    resource: str,
    over_limit: bool,
) -> None:
    source_root = tmp_path / "public"
    source_root.mkdir()
    limits = _BenchmarkCopyLimits(
        max_depth=2,
        max_entries=2,
        max_aggregate_bytes=4,
        max_scan_bytes=4,
    )
    if resource == "depth":
        limits = _BenchmarkCopyLimits(max_depth=2, max_entries=10, max_aggregate_bytes=100, max_scan_bytes=100)
        parent = source_root / "level-one"
        if over_limit:
            parent /= "level-two"
        parent.mkdir(parents=True)
        (parent / "payload.txt").write_bytes(b"safe")
    elif resource == "entries":
        limits = _BenchmarkCopyLimits(max_depth=2, max_entries=2, max_aggregate_bytes=100, max_scan_bytes=100)
        for index in range(3 if over_limit else 2):
            (source_root / f"payload-{index}.txt").write_bytes(b"x")
    else:
        limits = _BenchmarkCopyLimits(
            max_depth=2,
            max_entries=2,
            max_aggregate_bytes=4 if resource == "aggregate-bytes" else 100,
            max_scan_bytes=4 if resource == "scan-bytes" else 100,
        )
        (source_root / "payload.txt").write_bytes(b"x" * (5 if over_limit else 4))

    output = tmp_path / "export"
    if over_limit:
        with pytest.raises(ValueError, match="public export denied"):
            _copy_benchmark_data(source_root, output, include_images=False, _limits=limits)
        assert not output.exists()
    else:
        _copy_benchmark_data(source_root, output, include_images=False, _limits=limits)
        assert output.is_dir()
        assert (output / "README.md").is_file()


@pytest.mark.parametrize(
    "payload",
    [
        b"{}\nnot-json\n",
        b"{}\n" + b'{"value":"' + b"a" * (1024 * 1024) + b'"}\n',
    ],
    ids=["mixed-text", "overlong-row"],
)
def test_real_export_copy_rejects_invalid_or_overlong_rows_after_json_prefix(
    tmp_path: Path,
    payload: bytes,
) -> None:
    attack = tmp_path / "neutral"
    attack.mkdir()
    (attack / "payload.bin").write_bytes(payload)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(attack, output, include_images=False)
    assert not output.exists()


@pytest.mark.parametrize(
    ("source", "renamed"),
    [
        (RESULTS_ROOT / "long_dense-economics.json", "payload.bin"),
        (RESULTS_ROOT / "long_dense-economics.rankings.jsonl", "ranking.data"),
        (RESULTS_ROOT / "long_dense-economics.per-query.jsonl", "metrics.txt"),
        (AUDIT_ROOT / "audit-manifest.json", "metadata.bin"),
        (AUDIT_ROOT / "audit-pack-120.jsonl", "review.data"),
        (AUDIT_ROOT / "failure-cases.jsonl", "cases.txt"),
    ],
)
def test_export_rejects_renamed_restricted_artifacts_by_content(
    tmp_path: Path,
    source: Path,
    renamed: str,
) -> None:
    attack = tmp_path / "neutral"
    attack.mkdir()
    shutil.copy2(source, attack / renamed)
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(attack, tmp_path / "export", include_images=False)


def test_shared_bm25_filters_excluded_candidates() -> None:
    data = TrackData(
        "fixture",
        (
            {"id": "d1", "content": "alpha"},
            {"id": "d2", "content": "alpha alpha alpha"},
            {"id": "d3", "content": "beta"},
        ),
        ({"id": "q1", "text": "alpha", "excluded_ids": ["d2"]},),
        {"q1": {"d1": 1}},
    )
    rankings, _ = bm25_rank(data, top_k=2)
    assert [document_id for document_id, _ in rankings["q1"]] == ["d1", "d3"]
    assert query_metrics(data, rankings)["q1"]["recall@10"] == 1.0


def test_shared_validation_rejects_gold_excluded_overlap() -> None:
    data = TrackData(
        "fixture",
        ({"id": "d1", "content": "alpha"},),
        ({"id": "q1", "text": "alpha", "excluded_ids": ["d1"]},),
        {"q1": {"d1": 1}},
    )
    with pytest.raises(ValueError, match="gold/excluded overlap"):
        validate_track_data(data)


def test_bootstrap_is_deterministic() -> None:
    values = {
        "q1": {metric: 0.0 for metric in pilot.METRIC_NAMES},
        "q2": {metric: 1.0 for metric in pilot.METRIC_NAMES},
    }
    assert bootstrap_confidence_intervals(values, samples=100, seed=7) == bootstrap_confidence_intervals(
        values, samples=100, seed=7
    )


def test_results_and_safe_summary_are_schema_valid_when_present() -> None:
    if not (RESULTS_ROOT / "long_dense-economics.json").is_file():
        pytest.skip("real baselines have not run yet")
    Draft202012Validator.check_schema(RESULT_SCHEMA)
    for track in pilot.TRACKS:
        for method in pilot.BASELINE_METHODS:
            result = json.loads((RESULTS_ROOT / f"{method}-{track}.json").read_text())
            Draft202012Validator(RESULT_SCHEMA).validate(result)
            pilot._load_result(RESULTS_ROOT, DATA_ROOT, method, track)
    summary = json.loads((SAFE_ROOT / "summary.json").read_text())
    assert summary["restricted_text_or_identifiers_included"] is False
    assert summary["scope"]["comparison_to_full_corpus_biology_or_official_bright_allowed"] is False
    assert set(summary["unweighted_macro"]) == {"bm25", "long_dense"}
    assert summary["tracks"]["economics"]["minilm_diagnostic"]["baseline_role"] == (
        "diagnostic_truncation_degraded"
    )
    serialized = json.dumps(summary)
    assert '"query_id"' not in serialized
    assert '"document_id"' not in serialized


def test_result_schema_rejects_mutations_when_present() -> None:
    path = RESULTS_ROOT / "long_dense-economics.json"
    if not path.is_file():
        pytest.skip("real baselines have not run yet")
    validator = Draft202012Validator(RESULT_SCHEMA)
    result = json.loads(path.read_text())
    mutations = []
    for operation in (
        lambda value: value["execution"].__setitem__("trust_remote_code", True),
        lambda value: value["execution"].__setitem__("revision", "main"),
        lambda value: value["execution"].__setitem__("configured_max_sequence_length", 512),
        lambda value: value["execution"]["cap_selection"].__setitem__("score_used", True),
        lambda value: value.__setitem__("baseline_role", "diagnostic_truncation_degraded"),
        lambda value: value["rankings"].__setitem__("depth", 10),
        lambda value: value["publication"].__setitem__("public_export_allowed", True),
        lambda value: value.__setitem__("documents_searched", 7_499),
        lambda value: value.__setitem__("extra", True),
    ):
        mutated = copy.deepcopy(result)
        operation(mutated)
        mutations.append(mutated)
    assert all(not validator.is_valid(mutated) for mutated in mutations)


def test_safe_summary_artifact_hashes_match_and_contain_no_restricted_ids() -> None:
    manifest = json.loads((SAFE_ROOT / "manifest.json").read_text())
    summary_path = SAFE_ROOT / manifest["summary"]["path"]
    assert summary_path.stat().st_size == manifest["summary"]["bytes"]
    assert _sha256(summary_path) == manifest["summary"]["sha256"]
    summary = json.loads(summary_path.read_text())
    serialized = json.dumps(summary)
    assert summary["restricted_text_or_identifiers_included"] is False
    assert '"query_id"' not in serialized
    assert '"document_id"' not in serialized


def test_resigned_missing_ranking_row_is_rejected_by_black_box_loader(tmp_path: Path) -> None:
    results = _copy_result_artifacts(tmp_path)
    result_path = results / "long_dense-economics.json"
    result = json.loads(result_path.read_text())
    ranking_path = results / result["rankings"]["path"]
    rows = list(pilot._read_jsonl(ranking_path))
    _rewrite_jsonl(ranking_path, rows[:-1])
    _resign_artifact(results, result, "rankings")
    with pytest.raises(pilot.BrightPilotError, match="row count"):
        pilot._load_result(results, DATA_ROOT, "long_dense", "economics")


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_row",
        "unknown_query",
        "unknown_document",
        "valid_unjudged_document",
        "duplicate_rank",
        "duplicate_document",
        "row_disorder",
        "nan_score",
        "per_query_metric",
        "per_query_duplicate",
    ],
)
def test_resigned_result_artifact_mutations_fail_closed(
    tmp_path: Path,
    validated_economics_binding: tuple[dict, TrackData],
    mutation: str,
) -> None:
    results = _copy_result_artifacts(tmp_path)
    result_path = results / "long_dense-economics.json"
    result = json.loads(result_path.read_text())
    artifact_key = "per_query_metrics" if mutation.startswith("per_query") else "rankings"
    artifact_path = results / result[artifact_key]["path"]
    rows = list(pilot._read_jsonl(artifact_path))
    manifest, data = validated_economics_binding
    if mutation == "extra_row":
        extra = copy.deepcopy(rows[-1])
        extra["rank"] = 101
        rows.append(extra)
    elif mutation == "unknown_query":
        rows[0]["query_id"] = "not-a-materialized-query"
    elif mutation == "unknown_document":
        rows[0]["document_id"] = "not-a-materialized-document"
    elif mutation == "valid_unjudged_document":
        query_id = rows[0]["query_id"]
        query_rows = [row for row in rows if row["query_id"] == query_id]
        ranked_ids = {row["document_id"] for row in query_rows}
        replacement = next(
            row["id"]
            for row in data.corpus
            if row["id"] not in ranked_ids and row["id"] not in data.qrels[query_id]
        )
        index = next(
            index
            for index in range(1, len(query_rows) - 1)
            if query_rows[index]["document_id"] not in data.qrels[query_id]
            and query_rows[index - 1]["score"] > query_rows[index]["score"] > query_rows[index + 1]["score"]
        )
        rows[index]["document_id"] = replacement
    elif mutation == "duplicate_rank":
        rows[1]["rank"] = rows[0]["rank"]
    elif mutation == "duplicate_document":
        rows[1]["document_id"] = rows[0]["document_id"]
    elif mutation == "row_disorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "nan_score":
        rows[0]["score"] = float("nan")
    elif mutation == "per_query_metric":
        rows[0]["metrics"]["ndcg@10"] = 0.123456789
    else:
        rows[1]["query_id"] = rows[0]["query_id"]
    _rewrite_jsonl(artifact_path, rows)
    _resign_artifact(results, result, artifact_key)
    with pytest.raises(pilot.BrightPilotError):
        pilot._load_result_bound(
            results,
            "long_dense",
            "economics",
            data=data,
            materialization_manifest=manifest,
            materialization_manifest_path=DATA_ROOT / "manifest.json",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "metrics",
        "confidence_interval",
        "failure_counts",
        "materialization_manifest_sha256",
        "corpus_sha256",
        "queries_sha256",
        "qrels_sha256",
        "selection_sha256",
    ],
)
def test_resigned_result_contract_mutations_fail_closed(
    tmp_path: Path,
    validated_economics_binding: tuple[dict, TrackData],
    mutation: str,
) -> None:
    results = _copy_result_artifacts(tmp_path)
    result_path = results / "long_dense-economics.json"
    result = json.loads(result_path.read_text())
    if mutation == "metrics":
        result["metrics"]["ndcg@10"] = 0.123456789
    elif mutation == "confidence_interval":
        result["confidence_intervals"]["ndcg@10"]["mean"] = 0.123456789
    elif mutation == "failure_counts":
        result["failure_counts"]["zero_recall_at_10"] += 1
    else:
        result["data_identity"][mutation] = "0" * 64
    result_path.write_bytes(pilot._json_bytes(result))
    manifest, data = validated_economics_binding
    with pytest.raises(pilot.BrightPilotError):
        pilot._load_result_bound(
            results,
            "long_dense",
            "economics",
            data=data,
            materialization_manifest=manifest,
            materialization_manifest_path=DATA_ROOT / "manifest.json",
        )


def test_result_loader_rejects_extra_prefixed_artifact(
    tmp_path: Path,
    validated_economics_binding: tuple[dict, TrackData],
) -> None:
    results = _copy_result_artifacts(tmp_path)
    (results / "long_dense-economics.untracked.jsonl").write_text("{}\n", encoding="utf-8")
    manifest, data = validated_economics_binding
    with pytest.raises(pilot.BrightPilotError, match="file set"):
        pilot._load_result_bound(
            results,
            "long_dense",
            "economics",
            data=data,
            materialization_manifest=manifest,
            materialization_manifest_path=DATA_ROOT / "manifest.json",
        )


def test_ranking_validator_rejects_excluded_candidate_fixture() -> None:
    data = TrackData(
        "fixture",
        (
            {"id": "d1", "content": "one"},
            {"id": "d2", "content": "two"},
            {"id": "d3", "content": "three"},
        ),
        ({"id": "q1", "text": "query", "excluded_ids": ["d2"]},),
        {"q1": {"d1": 1}},
    )
    rows = (
        {"query_id": "q1", "rank": 1, "document_id": "d1", "score": 1.0},
        {"query_id": "q1", "rank": 2, "document_id": "d2", "score": 0.5},
    )
    with pytest.raises(pilot.BrightPilotError, match="candidate pool"):
        pilot._validate_ranking_rows(data, rows, depth=2)
