import json

import pytest

from mm_embed.benchmark import bright_label_audit as contract
from mm_embed.benchmark import bright_label_audit_primary as primary
from mm_embed.benchmark import bright_label_audit_primary_gemini as gemini
from mm_embed.hf_publish.export import _restricted_bright_object_reason


def _source() -> dict:
    return {
        "track": "economics",
        "query_id": "query-a",
        "gold_documents": [
            {"document_id": "gold-a", "title": "Gold", "text": "Evidence."},
        ],
        "candidate_documents": [
            {"document_id": "gold-a", "title": "Gold", "text": "Evidence."},
            {"document_id": "candidate-a", "title": "Candidate", "text": "More evidence."},
            {"document_id": "candidate-b", "title": "Candidate", "text": "Related topic."},
            {"document_id": "candidate-c", "title": "Candidate", "text": "Insufficient excerpt."},
        ],
    }


def _model_output() -> dict:
    return {
        "gold_reviews": [
            {
                "document_id": "gold-a",
                "support": "partially_supports",
                "ambiguity": "ambiguous",
                "confidence": "medium",
                "decision_status": "decided",
                "notes": "Partial support.",
            }
        ],
        "candidate_reviews": [
            {
                "document_id": "gold-a",
                "relevance": "relevant",
                "likely_hard_negative": "yes",
                "suspected_missing_positive": "credible",
                "sensitive_information_risk": "none",
                "sensitive_information_types": ["other"],
                "confidence": "high",
                "decision_status": "decided",
                "notes": "",
            },
            {
                "document_id": "candidate-a",
                "relevance": "relevant",
                "likely_hard_negative": "yes",
                "suspected_missing_positive": "not_credible",
                "sensitive_information_risk": "low",
                "sensitive_information_types": [],
                "confidence": "medium",
                "decision_status": "decided",
                "notes": "",
            },
            {
                "document_id": "candidate-b",
                "relevance": "not_relevant",
                "likely_hard_negative": "yes",
                "suspected_missing_positive": "credible",
                "sensitive_information_risk": "none",
                "sensitive_information_types": [],
                "confidence": "high",
                "decision_status": "decided",
                "notes": "",
            },
            {
                "document_id": "candidate-c",
                "relevance": "uncertain",
                "likely_hard_negative": "no",
                "suspected_missing_positive": "credible",
                "sensitive_information_risk": "uncertain",
                "sensitive_information_types": [],
                "confidence": "medium",
                "decision_status": "decided",
                "notes": "Insufficient evidence.",
            },
        ],
        "query_review": {
            "credible_missing_positive": "no",
            "credible_missing_positive_ids": [],
            "likely_hard_negative_present": "no",
            "likely_hard_negative_ids": [],
            "qrels_completeness": "complete_enough",
            "answerability": "answerable",
            "label_incompleteness_risk": "none",
            "sensitive_information_risk": "none",
            "confidence": "high",
            "decision_status": "decided",
            "notes": "",
        },
    }


def test_normalization_derives_cross_item_semantics_and_preserves_uncertainty() -> None:
    reviewed = primary._normalize_model_output(_source(), _model_output())
    candidates = {item["document_id"]: item for item in reviewed["candidate_reviews"]}
    assert candidates["gold-a"]["suspected_missing_positive"] == "not_applicable"
    assert candidates["gold-a"]["likely_hard_negative"] == "no"
    assert candidates["gold-a"]["sensitive_information_types"] == []
    assert candidates["candidate-a"]["suspected_missing_positive"] == "credible"
    assert candidates["candidate-a"]["likely_hard_negative"] == "no"
    assert candidates["candidate-a"]["sensitive_information_types"] == ["other"]
    assert candidates["candidate-b"]["suspected_missing_positive"] == "not_credible"
    assert candidates["candidate-c"]["suspected_missing_positive"] == "uncertain"
    assert candidates["candidate-c"]["likely_hard_negative"] == "uncertain"
    assert candidates["candidate-c"]["confidence"] == "low"
    assert candidates["candidate-c"]["decision_status"] == "uncertain"
    assert reviewed["query_review"]["credible_missing_positive"] == "yes"
    assert reviewed["query_review"]["credible_missing_positive_ids"] == ["candidate-a"]
    assert reviewed["query_review"]["likely_hard_negative_present"] == "yes"
    assert reviewed["query_review"]["likely_hard_negative_ids"] == ["candidate-b"]
    assert reviewed["query_review"]["qrels_completeness"] == "incomplete"


def test_normalization_rejects_incomplete_or_duplicate_model_coverage() -> None:
    output = _model_output()
    output["candidate_reviews"].pop()
    with pytest.raises(primary.PrimaryAuditRunError, match="candidate coverage"):
        primary._normalize_model_output(_source(), output)

    output = _model_output()
    output["candidate_reviews"][-1]["document_id"] = "candidate-b"
    with pytest.raises(primary.PrimaryAuditRunError, match="candidate coverage"):
        primary._normalize_model_output(_source(), output)


def test_provenance_is_attached_to_every_judgment_without_human_claim() -> None:
    source = _source()
    reviewed = primary._normalize_model_output(source, _model_output())
    annotation = primary._attach_provenance(
        source,
        reviewed,
        provider=gemini.PROVIDER,
        model="gemini-test",
        model_revision="gemini-test-revision",
        session_id="executor-session",
        judged_at="2026-09-06T13:00:00Z",
        rubric_sha256="a" * 64,
        input_sha256="b" * 64,
    )
    judgments = [*annotation["gold_reviews"], *annotation["candidate_reviews"], annotation["query_review"]]
    assert annotation["input_row_sha256"] == contract._sha256_json(source)
    assert {item["provenance"]["judgment_origin"] for item in judgments} == {"llm_assisted"}
    assert {item["provenance"]["agent_role"] for item in judgments} == {"executor_primary"}
    assert {item["provenance"]["provider"] for item in judgments} == {"google_gemini"}
    assert {item["provenance"]["session_id"] for item in judgments} == {"executor-session"}


def test_gemini_sampling_contract_is_explicit_and_json_serializable() -> None:
    sampling = gemini._sampling(32768, "low")
    assert sampling == {
        "temperature": 0,
        "top_p": "server_default",
        "seed": "not_set",
        "max_output_tokens": 32768,
        "response_format": "application/json with response_json_schema",
        "thinking_level": "low",
        "retry_instruction_version": "coverage-repair-v1",
    }
    json.dumps(sampling)


def test_exact_prompt_identity_covers_base_and_repair_request_material() -> None:
    base_contents, base_variant, base_sha256 = gemini._request_material("prompt", "rubric", _source(), 1)
    retry_contents, retry_variant, retry_sha256 = gemini._request_material("prompt", "rubric", _source(), 2)
    assert base_variant == "base"
    assert retry_variant == "coverage-repair-v1"
    assert "coverage-repair retry" not in base_contents
    assert "coverage-repair retry" in retry_contents
    assert len(base_sha256) == len(retry_sha256) == 64
    assert base_sha256 != retry_sha256


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            {
                "audit_id": contract.AUDIT_ID,
                "judgment_origin": "llm_assisted",
                "provider": "provider",
                "session_id": "session",
                "requests": {},
                "raw_responses": {},
                "annotations": {},
            },
            "run provenance",
        ),
        (
            {
                "query_id": "query",
                "input_row_sha256": "a" * 64,
                "request_ordinal": 1,
                "session_id": "session",
                "prompt_recipe_sha256": "b" * 64,
                "raw_response_sha256": "c" * 64,
            },
            "request provenance",
        ),
        (
            {
                "query_id": "query",
                "input_row_sha256": "a" * 64,
                "request_ordinal": 1,
                "session_id": "session",
                "prompt_recipe_sha256": "b" * 64,
                "content": "restricted model output",
            },
            "raw response",
        ),
    ],
)
def test_public_export_rejects_renamed_primary_provenance(value: dict, expected: str) -> None:
    reason = _restricted_bright_object_reason(value, "bright-nontechnical-pilot-v0.2")
    assert reason is not None
    assert expected in reason
