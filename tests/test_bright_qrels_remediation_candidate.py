import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mm_embed.benchmark import bright_qrels_remediation as control
from mm_embed.benchmark import bright_qrels_remediation_candidate as candidate
from mm_embed.hf_publish.export import _reject_restricted_materializations

RESTRICTED = Path("results/bright-qrels-remediation-v0.1/restricted")
TRACKED = Path("benchmark/artifacts/bright-qrels-remediation-v0.1")


def _object(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_object(path: Path, value: dict) -> None:
    path.write_bytes(candidate._json_bytes(value))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_bytes(candidate._jsonl_bytes(rows))


def _isolated_materialization(tmp_path: Path) -> Path:
    def link_file(source: str, target: str) -> str:
        Path(target).symlink_to(Path(source).resolve())
        return target

    repository = tmp_path / "repository"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(Path.cwd()), str(repository)],
        check=True,
    )
    for key, relative_path in control.DEFAULT_PATHS.items():
        if key in {"control", "restricted_results", "accepted_verification"}:
            continue
        source = Path(relative_path).resolve()
        target = repository / relative_path
        if target.exists():
            if source.is_dir():
                for source_file in source.rglob("*"):
                    if not source_file.is_file():
                        continue
                    target_file = target / source_file.relative_to(source)
                    if not target_file.exists():
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        target_file.symlink_to(source_file)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, copy_function=link_file)
        else:
            target.symlink_to(source)
    accepted_results = repository / control.DEFAULT_PATHS["restricted_results"]
    if not accepted_results.exists():
        accepted_results.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            Path(control.DEFAULT_PATHS["restricted_results"]).resolve(),
            accepted_results,
            copy_function=link_file,
        )
    for relative_path in (
        control.DEFAULT_PATHS["control"],
        control.DEFAULT_PATHS["control"].with_suffix(".json.sha256"),
        TRACKED / "candidate-aggregate.json",
        TRACKED / "candidate-aggregate.json.sha256",
        candidate.REPORT_PATH,
        candidate.REPORT_PATH.with_suffix(".md.sha256"),
    ):
        target = repository / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative_path, target)
    shutil.copytree(RESTRICTED, repository / RESTRICTED)
    return repository


def _resign_materialization(repository: Path) -> None:
    restricted = repository / RESTRICTED
    manifest_path = restricted / "manifest.json"
    manifest = _object(manifest_path)
    for name in manifest["artifacts"]:
        path = restricted / name
        payload = path.read_bytes()
        manifest["artifacts"][name] = {
            "rows": len(payload.splitlines()) if name.endswith(".jsonl") else 1,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        path.with_suffix(path.suffix + ".sha256").write_text(
            manifest["artifacts"][name]["sha256"] + "\n",
            encoding="ascii",
        )
    _write_object(manifest_path, manifest)
    manifest_sha256 = control._sha256_file(manifest_path)
    manifest_path.with_suffix(".json.sha256").write_text(manifest_sha256 + "\n", encoding="ascii")

    aggregate_path = repository / TRACKED / "candidate-aggregate.json"
    aggregate = _object(aggregate_path)
    aggregate["restricted_artifacts"]["files"] = manifest["artifacts"]
    aggregate["restricted_artifacts"]["proposal_sha256"] = manifest["artifacts"]["proposal.json"]["sha256"]
    aggregate["restricted_artifacts"]["manifest_sha256"] = manifest_sha256
    _write_object(aggregate_path, aggregate)
    aggregate_path.with_suffix(".json.sha256").write_text(
        control._sha256_file(aggregate_path) + "\n",
        encoding="ascii",
    )
    report_path = repository / candidate.REPORT_PATH
    report_path.write_text(candidate._report(aggregate), encoding="utf-8")
    report_path.with_suffix(".md.sha256").write_text(
        control._sha256_file(report_path) + "\n",
        encoding="ascii",
    )


def test_full_candidate_validates_end_to_end() -> None:
    result = candidate.validate_candidate()
    assert result == {
        "candidate_id": "bright-qrels-remediation-candidate-v0.1",
        "status": "inactive",
        "coverage": {"queries": 120, "gold_items": 1_235, "candidate_items": 2_804},
        "dispositions": 4_159,
        "diff_rows": 1_235,
        "verdict": {
            "tracks": {
                "economics": "task_semantics_mismatch",
                "psychology": "task_semantics_mismatch",
            },
            "overall": "task_semantics_mismatch",
        },
        "publication": "closed",
    }


def test_restricted_artifact_identities_are_bound() -> None:
    manifest = _object(RESTRICTED / "manifest.json")
    expected = {
        "proposal.json": "d4afe0a4848df0a745d9f06fa01e8679fbb8443dee35158db90bc013b6daf8a1",
        "original-vs-proposed.jsonl": "d883ae92b6548e46422c71d8036fcd6033f14d26b8510522fcc7b180f3cf410a",
        "evidence-ledger.jsonl": "e355cdeb3e46ba2cbdd4be060f2a69ede1b9a8932f11635324e153b38858dbf6",
        "query-risk.jsonl": "59cdb911090825c013cc531bb125042695061f97cb46e5cca9518847b990f6d2",
    }
    assert {name: value["sha256"] for name, value in manifest["artifacts"].items()} == expected
    assert control._sha256_file(RESTRICTED / "manifest.json") == (
        "587947d7e2d4cbac5184742eae3e1e37344a9661bada3e7a697320de56b059d9"
    )
    assert control._sha256_file(TRACKED / "candidate-aggregate.json") == (
        "af716f4b2fbc588b41bd278b105d6767bcabfa43babeb062a9dc4c8ec3bc3322"
    )


def test_tracked_candidate_set_is_exact_and_excludes_protected_artifacts() -> None:
    identities = candidate.candidate_identities()
    assert identities["count"] == 20
    assert len(identities["paths"]) == len(set(identities["paths"]))
    assert all("validator-candidate" not in path for path in identities["paths"])
    assert all(not path.startswith(("data/", "results/", "configs/")) for path in identities["paths"])


def test_full_coverage_is_exact_and_unique() -> None:
    proposal = _object(RESTRICTED / "proposal.json")
    keys = [control._unit_key(row) for row in proposal["dispositions"]]
    assert len(keys) == len(set(keys)) == 4_159
    assert sum(row["unit_type"] == "query" for row in proposal["dispositions"]) == 120
    assert sum(row["unit_type"] == "gold" for row in proposal["dispositions"]) == 1_235
    assert sum(row["unit_type"] == "candidate" for row in proposal["dispositions"]) == 2_804
    assert proposal["proposed_qrels_rows"] == []


def test_membership_changes_have_independent_semantic_review_provenance() -> None:
    proposal = _object(RESTRICTED / "proposal.json")
    changes = [row for row in proposal["dispositions"] if row["unit_type"] == "gold"]
    assert len(changes) == 1_235
    assert {row["disposition"] for row in changes} == {
        "quarantine_official_positive",
        "downgrade_to_uncertain",
    }
    assert all(row["provenance"]["session_id"] == candidate.REVIEW_SESSION for row in changes)
    assert all(row["provenance"]["agent_role"] == candidate.REVIEW_ROLE for row in changes)
    assert all(row["provenance"]["judgment_origin"] == "agent_judged" for row in changes)
    assert all(row["provenance"]["round"] == 1 for row in changes)
    assert all("task_semantics_mismatch" not in row["taxonomy"] for row in changes)
    assert sum("gold_support_defect" in row["taxonomy"] for row in changes) == 1_034
    assert sum("gold_ambiguity" in row["taxonomy"] for row in changes) == 10
    assert sum(row["taxonomy"] == ["no_qrels_defect"] for row in changes) == 199


def test_evidence_ledger_preserves_primary_and_adjudicated_judgments() -> None:
    ledger = _rows(RESTRICTED / "evidence-ledger.jsonl")
    assert len(ledger) == 4_159
    assert sum(row["adjudicated_judgment"] is not None for row in ledger) == 241
    for row in ledger:
        assert row["primary_judgment_sha256"] == candidate._row_sha256(row["primary_judgment"])
        assert row["effective_judgment_sha256"] == candidate._row_sha256(row["effective_judgment"])
        if row["adjudicated_judgment"] is not None:
            assert row["effective_judgment"] == row["adjudicated_judgment"]


def test_diff_is_complete_nonnegative_and_consistent() -> None:
    proposal = _object(RESTRICTED / "proposal.json")
    diff = _rows(RESTRICTED / "original-vs-proposed.jsonl")
    gold = [row for row in proposal["dispositions"] if row["unit_type"] == "gold"]
    assert len(diff) == len(gold) == 1_235
    assert all(row["operation"] == "quarantine" for row in diff)
    assert all(row["original_grade"] >= 1 and row["proposed_grade"] is None for row in diff)
    assert all(row["negative_synthesized"] is False for row in diff)
    assert {(row["track"], row["query_id"], row["document_id"]) for row in diff} == {
        (row["track"], row["query_id"], row["document_id"]) for row in gold
    }


def test_query_and_track_risk_coverage_recomputes() -> None:
    aggregate = _object(TRACKED / "candidate-aggregate.json")
    risks = _rows(RESTRICTED / "query-risk.jsonl")
    assert len(risks) == 120
    assert sum(row["coverage"]["query"] for row in risks) == 120
    assert sum(row["coverage"]["gold"] for row in risks) == 1_235
    assert sum(row["coverage"]["candidate"] for row in risks) == 2_804
    assert all(row["verdict"] == "task_semantics_mismatch" for row in risks)
    assert aggregate["tracks"]["economics"]["reviewed_queries_stopped"] == 60
    assert aggregate["tracks"]["psychology"]["reviewed_queries_stopped"] == 60


def test_accepted_item_level_root_causes_are_preserved() -> None:
    aggregate = _object(TRACKED / "candidate-aggregate.json")
    assert aggregate["accepted_evidence"] == {
        "adjudicated_units": 241,
        "candidates": {
            "credible_missing_positive_by_track": {"economics": 53, "psychology": 44},
            "relevance": {"not_relevant": 2_544, "relevant": 260},
            "relevance_by_track": {
                "economics": {"not_relevant": 1_198, "relevant": 147},
                "psychology": {"not_relevant": 1_346, "relevant": 113},
            },
            "suspected_missing_positive": {
                "credible": 97,
                "not_applicable": 291,
                "not_credible": 2_416,
            },
            "suspected_missing_positive_by_track": {
                "economics": {"credible": 53, "not_applicable": 167, "not_credible": 1_125},
                "psychology": {"credible": 44, "not_applicable": 124, "not_credible": 1_291},
            },
        },
        "gold": {
            "ambiguity": {"ambiguous": 10, "unambiguous": 1_225},
            "ambiguity_by_track": {
                "economics": {"ambiguous": 1, "unambiguous": 664},
                "psychology": {"ambiguous": 9, "unambiguous": 561},
            },
            "support": {"does_not_support": 744, "partially_supports": 290, "supports": 201},
            "support_by_track": {
                "economics": {"does_not_support": 390, "partially_supports": 144, "supports": 131},
                "psychology": {"does_not_support": 354, "partially_supports": 146, "supports": 70},
            },
        },
        "queries": {
            "credible_missing_positive": {"no": 78, "yes": 42},
            "credible_missing_positive_queries_by_track": {"economics": 21, "psychology": 21},
            "decision_status": {"decided": 119, "uncertain": 1},
            "qrels_completeness": {"complete_enough": 41, "incomplete": 78, "uncertain": 1},
        },
        "unresolved_queries": {"by_track": {"economics": 1, "psychology": 0}, "total": 1},
    }
    assert aggregate["dispositions"]["by_taxonomy"] == {
        "candidate_pool_coverage_gap": 36,
        "credible_missing_positive": 139,
        "gold_ambiguity": 10,
        "gold_support_defect": 1_034,
        "insufficient_existing_evidence": 1,
        "no_qrels_defect": 2_906,
        "task_semantics_mismatch": 120,
    }


def test_resigned_unknown_diff_identity_fails_formal_entry(tmp_path: Path) -> None:
    repository = _isolated_materialization(tmp_path)
    diff_path = repository / RESTRICTED / "original-vs-proposed.jsonl"
    rows = _rows(diff_path)
    rows[0]["document_id"] = "unknown-resigned-document"
    _write_rows(diff_path, rows)
    _resign_materialization(repository)
    with pytest.raises(control.BrightQrelsRemediationError, match="deterministic recomputation mismatch"):
        candidate.validate_candidate(repository=repository)


def test_resigned_accepted_judgment_tamper_fails_formal_entry(tmp_path: Path) -> None:
    repository = _isolated_materialization(tmp_path)
    ledger_path = repository / RESTRICTED / "evidence-ledger.jsonl"
    rows = _rows(ledger_path)
    rows[0]["primary_judgment"]["notes"] = "resigned tamper"
    rows[0]["effective_judgment"] = copy.deepcopy(rows[0]["primary_judgment"])
    rows[0]["primary_judgment_sha256"] = candidate._row_sha256(rows[0]["primary_judgment"])
    rows[0]["effective_judgment_sha256"] = candidate._row_sha256(rows[0]["effective_judgment"])
    _write_rows(ledger_path, rows)
    _resign_materialization(repository)
    with pytest.raises(control.BrightQrelsRemediationError, match="deterministic recomputation mismatch"):
        candidate.validate_candidate(repository=repository)


def test_resigned_duplicate_query_risk_identities_fail_formal_entry(tmp_path: Path) -> None:
    repository = _isolated_materialization(tmp_path)
    risk_path = repository / RESTRICTED / "query-risk.jsonl"
    rows = _rows(risk_path)
    economics = next(row for row in rows if row["track"] == "economics")
    psychology = next(row for row in rows if row["track"] == "psychology")
    _write_rows(
        risk_path, [copy.deepcopy(economics) for _ in range(60)] + [copy.deepcopy(psychology) for _ in range(60)]
    )
    _resign_materialization(repository)
    with pytest.raises(control.BrightQrelsRemediationError, match="deterministic recomputation mismatch"):
        candidate.validate_candidate(repository=repository)


def test_resigned_fake_positive_span_fails_formal_entry(tmp_path: Path) -> None:
    repository = _isolated_materialization(tmp_path)
    proposal_path = repository / RESTRICTED / "proposal.json"
    proposal = _object(proposal_path)
    row = next(
        item
        for item in proposal["dispositions"]
        if item["unit_type"] == "candidate" and item["source_judgment_status"] == "unjudged"
    )
    row["disposition"] = "propose_add_positive"
    row["evidence_level"] = "direct_supporting_span"
    pack = control._read_jsonl(control.DEFAULT_PATHS["audit_pack"], "frozen audit pack")
    canonical = control._canonical_documents(pack)[(row["track"], row["query_id"], row["document_id"])]
    fake_text = "x" if canonical[0] != "x" else "y"
    row["supporting_spans"] = [
        {
            "document_id": row["document_id"],
            "start_offset": 0,
            "end_offset": 1,
            "text": fake_text,
            "span_sha256": hashlib.sha256(fake_text.encode("utf-8")).hexdigest(),
        }
    ]
    proposal["proposed_qrels_rows"].append(
        {
            "track": row["track"],
            "query_id": row["query_id"],
            "document_id": row["document_id"],
            "grade": 1,
            "operation": "add",
        }
    )
    _write_object(proposal_path, proposal)
    ledger_path = repository / RESTRICTED / "evidence-ledger.jsonl"
    ledger = _rows(ledger_path)
    ledger_row = next(item for item in ledger if control._unit_key(item["remediation"]) == control._unit_key(row))
    ledger_row["remediation"] = copy.deepcopy(row)
    _write_rows(ledger_path, ledger)
    _resign_materialization(repository)
    with pytest.raises(control.BrightQrelsRemediationError, match="declared canonical offsets"):
        candidate.validate_candidate(repository=repository)


def _assert_build_and_validate_reject_symlink(repository: Path) -> None:
    for operation in (candidate.build_candidate, candidate.validate_candidate):
        with pytest.raises(control.BrightQrelsRemediationError, match="symlink"):
            operation(repository=repository)


def test_restricted_root_symlink_escape_fails_formal_build_and_validate(tmp_path: Path) -> None:
    repository = _isolated_materialization(tmp_path)
    root = repository / RESTRICTED
    relocated = root.parent / "relocated-restricted"
    root.rename(relocated)
    root.symlink_to(relocated, target_is_directory=True)
    _assert_build_and_validate_reject_symlink(repository)


def test_restricted_parent_symlink_escape_fails_formal_build_and_validate(tmp_path: Path) -> None:
    repository = _isolated_materialization(tmp_path)
    parent = (repository / RESTRICTED).parent
    relocated = parent.parent / "relocated-remediation-parent"
    parent.rename(relocated)
    parent.symlink_to(relocated, target_is_directory=True)
    _assert_build_and_validate_reject_symlink(repository)


def test_every_restricted_payload_and_sidecar_symlink_fails_formal_entries(tmp_path: Path) -> None:
    repository = _isolated_materialization(tmp_path)
    root = repository / RESTRICTED
    relocated = root.parent / "relocated-files"
    relocated.mkdir()
    for index, name in enumerate(candidate.RESTRICTED_BUNDLE_NAMES):
        path = root / name
        target = relocated / f"{index}-{path.name}"
        path.rename(target)
        path.symlink_to(target)
        _assert_build_and_validate_reject_symlink(repository)
        path.unlink()
        target.rename(path)


def _supporting_span_fixture() -> tuple[dict, dict, dict, dict]:
    control_value = _object(control.DEFAULT_PATHS["control"])
    audit = control_value["source_identities"]["audit"]
    input_row_sha256 = "1" * 64
    source_annotation_sha256 = audit["primary_annotations"]["sha256"]
    content = "repeat middle repeat"
    start_offset = content.rindex("repeat")
    text = "repeat"
    row = {
        "unit_type": "candidate",
        "track": "economics",
        "query_id": "offset-query",
        "document_id": "offset-document",
        "source_judgment_status": "unjudged",
        "taxonomy": ["credible_missing_positive"],
        "disposition": "propose_add_positive",
        "evidence_level": "direct_supporting_span",
        "supporting_spans": [
            {
                "document_id": "offset-document",
                "start_offset": start_offset,
                "end_offset": start_offset + len(text),
                "text": text,
                "span_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        ],
        "rationale": "Synthetic explicit-offset boundary fixture.",
        "provenance": candidate._provenance(input_row_sha256, source_annotation_sha256, audit),
    }
    expected = {
        "source_judgment_status": "unjudged",
        "input_row_sha256": input_row_sha256,
        "source_annotation_sha256": source_annotation_sha256,
    }
    canonical = {("economics", "offset-query", "offset-document"): content}
    return row, expected, control_value, canonical


def test_explicit_second_occurrence_offset_is_honored() -> None:
    row, expected, control_value, canonical = _supporting_span_fixture()
    control._validate_disposition_semantics(row, expected, control_value, canonical)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda span: span.update(start_offset=span["end_offset"]), "outside canonical document bounds"),
        (lambda span: span.update(end_offset=10_000), "outside canonical document bounds"),
        (lambda span: span.update(start_offset=1, end_offset=7), "declared canonical offsets"),
        (lambda span: span.update(text="middle"), "identity mismatch"),
    ),
)
def test_invalid_supporting_span_offset_mutations_fail_closed(mutation, message: str) -> None:
    row, expected, control_value, canonical = _supporting_span_fixture()
    mutation(row["supporting_spans"][0])
    with pytest.raises(control.BrightQrelsRemediationError, match=message):
        control._validate_disposition_semantics(row, expected, control_value, canonical)


def test_repeated_text_occurrence_cannot_ignore_declared_offsets() -> None:
    row, expected, control_value, canonical = _supporting_span_fixture()
    span = row["supporting_spans"][0]
    span["start_offset"] = 0
    span["end_offset"] = len(canonical[("economics", "offset-query", "offset-document")])
    with pytest.raises(control.BrightQrelsRemediationError, match="declared canonical offsets"):
        control._validate_disposition_semantics(row, expected, control_value, canonical)


def test_supporting_span_schema_requires_explicit_offsets() -> None:
    proposal = _object(RESTRICTED / "proposal.json")
    row = next(item for item in proposal["dispositions"] if item["unit_type"] == "candidate")
    row["supporting_spans"] = [
        {
            "document_id": row["document_id"],
            "text": "missing offsets",
            "span_sha256": hashlib.sha256(b"missing offsets").hexdigest(),
        }
    ]
    with pytest.raises(control.BrightQrelsRemediationError, match="start_offset"):
        control._validate_schema(
            proposal,
            "bright-qrels-remediation-proposal-v01.schema.json",
            "proposal missing explicit span offsets",
        )


def test_safe_outputs_contain_no_restricted_literals() -> None:
    aggregate = _object(TRACKED / "candidate-aggregate.json")
    report = candidate.REPORT_PATH.read_text(encoding="utf-8")
    pack = control._read_jsonl(control.DEFAULT_PATHS["audit_pack"], "frozen audit pack")
    candidate._validate_safe_output_privacy(aggregate, report, pack)


def test_privacy_scan_rejects_a_real_canonical_identity() -> None:
    aggregate = _object(TRACKED / "candidate-aggregate.json")
    pack = control._read_jsonl(control.DEFAULT_PATHS["audit_pack"], "frozen audit pack")
    leaked = next(
        item["document_id"]
        for row in pack
        for item in [*row["gold_documents"], *row["baseline_top10_union"]]
        if len(item["document_id"]) >= 8
    )
    with pytest.raises(control.BrightQrelsRemediationError, match="canonical identity"):
        candidate._validate_safe_output_privacy(aggregate, leaked, pack)


def test_public_exporter_rejects_restricted_candidate() -> None:
    with pytest.raises(ValueError, match="public export denied"):
        _reject_restricted_materializations(RESTRICTED)


def test_protected_registry_runner_and_export_surfaces_are_unchanged() -> None:
    candidate._validate_protected_surfaces(Path.cwd())


def test_protected_registry_drift_fails_closed(tmp_path: Path) -> None:
    relative_path = "src/mm_embed/benchmark/registry.py"
    drifted = tmp_path / "registry.py"
    drifted.write_bytes(Path(relative_path).read_bytes() + b"\n")
    with pytest.raises(control.BrightQrelsRemediationError, match="protected publication surface identity drift"):
        candidate._validate_protected_surfaces(Path.cwd(), overrides={relative_path: drifted})


def test_negative_diff_mutation_fails_closed() -> None:
    row = copy.deepcopy(_rows(RESTRICTED / "original-vs-proposed.jsonl")[0])
    row["negative_synthesized"] = True
    with pytest.raises(control.BrightQrelsRemediationError, match="negative_synthesized"):
        control._validate_schema(row, "bright-qrels-remediation-diff-v01.schema.json", "mutated diff row")


def test_recomputed_aggregate_rejects_drift() -> None:
    aggregate = _object(TRACKED / "candidate-aggregate.json")
    aggregate["diff"]["negative_rows"] = 1
    with pytest.raises(control.BrightQrelsRemediationError, match="negative_rows"):
        control._validate_schema(
            aggregate,
            "bright-qrels-remediation-safe-aggregate-v01.schema.json",
            "mutated safe aggregate",
        )
