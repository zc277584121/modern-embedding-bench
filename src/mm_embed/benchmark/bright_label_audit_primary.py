"""Bounded primary-label runner for the restricted BRIGHT label-quality audit."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from mm_embed.benchmark import bright_label_audit as contract
from mm_embed.benchmark.retrieval_v01 import json_dumps, write_jsonl

PROVIDER = "azure_openai_compatible"
API_SURFACE = "azure_chat_completions"
JUDGMENT_ORIGIN = "llm_assisted"


class PrimaryAuditRunError(RuntimeError):
    """A stable fail-closed primary-label execution error."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _output_schema() -> dict[str, Any]:
    string = {"type": "string"}
    confidence = {"type": "string", "enum": ["high", "medium", "low"]}
    decision = {"type": "string", "enum": ["decided", "uncertain", "abstain"]}
    risk = {"type": "string", "enum": ["none", "low", "moderate", "high", "uncertain"]}
    provenance_free_gold = {
        "type": "object",
        "additionalProperties": False,
        "required": ["document_id", "support", "ambiguity", "confidence", "decision_status", "notes"],
        "properties": {
            "document_id": string,
            "support": {
                "type": "string",
                "enum": ["supports", "partially_supports", "does_not_support", "uncertain"],
            },
            "ambiguity": {"type": "string", "enum": ["unambiguous", "ambiguous", "uncertain"]},
            "confidence": confidence,
            "decision_status": decision,
            "notes": string,
        },
    }
    provenance_free_candidate = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "document_id",
            "relevance",
            "likely_hard_negative",
            "suspected_missing_positive",
            "sensitive_information_risk",
            "sensitive_information_types",
            "confidence",
            "decision_status",
            "notes",
        ],
        "properties": {
            "document_id": string,
            "relevance": {"type": "string", "enum": ["relevant", "not_relevant", "uncertain"]},
            "likely_hard_negative": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "suspected_missing_positive": {
                "type": "string",
                "enum": ["credible", "not_credible", "not_applicable", "uncertain"],
            },
            "sensitive_information_risk": risk,
            "sensitive_information_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "personal_contact",
                        "public_contact",
                        "network_identifier",
                        "credential_or_secret",
                        "health",
                        "financial",
                        "legal",
                        "minor",
                        "reidentification",
                        "other",
                    ],
                },
            },
            "confidence": confidence,
            "decision_status": decision,
            "notes": string,
        },
    }
    query_review = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "credible_missing_positive",
            "credible_missing_positive_ids",
            "likely_hard_negative_present",
            "likely_hard_negative_ids",
            "qrels_completeness",
            "answerability",
            "label_incompleteness_risk",
            "sensitive_information_risk",
            "confidence",
            "decision_status",
            "notes",
        ],
        "properties": {
            "credible_missing_positive": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "credible_missing_positive_ids": {"type": "array", "items": string},
            "likely_hard_negative_present": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "likely_hard_negative_ids": {"type": "array", "items": string},
            "qrels_completeness": {
                "type": "string",
                "enum": ["complete_enough", "incomplete", "uncertain"],
            },
            "answerability": {"type": "string", "enum": ["answerable", "no_answer", "uncertain"]},
            "label_incompleteness_risk": risk,
            "sensitive_information_risk": risk,
            "confidence": confidence,
            "decision_status": decision,
            "notes": string,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["gold_reviews", "candidate_reviews", "query_review"],
        "properties": {
            "gold_reviews": {"type": "array", "items": provenance_free_gold},
            "candidate_reviews": {"type": "array", "items": provenance_free_candidate},
            "query_review": query_review,
        },
    }


def _request_body(deployment: str, prompt: str, rubric: str, source: dict[str, Any], max_tokens: int) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": f"{prompt}\n\nFROZEN RUBRIC:\n{rubric}",
            },
            {
                "role": "user",
                "content": "Review this one source-only audit row:\n" + json_dumps(source),
            },
        ],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "bright_primary_audit_row",
                "strict": True,
                "schema": _output_schema(),
            },
        },
        "model": deployment,
    }


def _normalize_judgment_status(item: dict[str, Any], uncertain: bool) -> None:
    if uncertain:
        item["confidence"] = "low"
        item["decision_status"] = "abstain" if item.get("decision_status") == "abstain" else "uncertain"
        if item["decision_status"] == "abstain" and not item.get("notes", "").strip():
            item["notes"] = "The supplied evidence is insufficient for a reliable judgment."
    else:
        item["decision_status"] = "decided"


def _normalize_model_output(source: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    gold_by_id = {item["document_id"]: item for item in value["gold_reviews"]}
    candidate_by_id = {item["document_id"]: item for item in value["candidate_reviews"]}
    expected_gold = [item["document_id"] for item in source["gold_documents"]]
    expected_candidates = [item["document_id"] for item in source["candidate_documents"]]
    if len(gold_by_id) != len(value["gold_reviews"]) or set(gold_by_id) != set(expected_gold):
        raise PrimaryAuditRunError("model gold coverage does not match the source row")
    if len(candidate_by_id) != len(value["candidate_reviews"]) or set(candidate_by_id) != set(expected_candidates):
        raise PrimaryAuditRunError("model candidate coverage does not match the source row")
    gold_reviews = []
    for document_id in expected_gold:
        item = gold_by_id[document_id]
        _normalize_judgment_status(item, "uncertain" in {item["support"], item["ambiguity"]})
        gold_reviews.append(item)
    gold_ids = set(expected_gold)
    candidate_reviews = []
    for document_id in expected_candidates:
        item = candidate_by_id[document_id]
        relevance = item["relevance"]
        if document_id in gold_ids:
            item["suspected_missing_positive"] = "not_applicable"
        elif relevance == "relevant":
            item["suspected_missing_positive"] = "credible"
        elif relevance == "not_relevant":
            item["suspected_missing_positive"] = "not_credible"
        else:
            item["suspected_missing_positive"] = "uncertain"
        if relevance == "relevant":
            item["likely_hard_negative"] = "no"
        elif relevance == "uncertain":
            item["likely_hard_negative"] = "uncertain"
        if item["sensitive_information_risk"] == "none":
            item["sensitive_information_types"] = []
        elif (
            item["sensitive_information_risk"] in {"low", "moderate", "high"}
            and not item["sensitive_information_types"]
        ):
            item["sensitive_information_types"] = ["other"]
        uncertain = "uncertain" in {
            item["relevance"],
            item["likely_hard_negative"],
            item["suspected_missing_positive"],
            item["sensitive_information_risk"],
        }
        _normalize_judgment_status(item, uncertain)
        candidate_reviews.append(item)
    credible_ids = sorted(
        item["document_id"] for item in candidate_reviews if item["suspected_missing_positive"] == "credible"
    )
    hard_ids = sorted(item["document_id"] for item in candidate_reviews if item["likely_hard_negative"] == "yes")
    uncertain_missing = any(item["suspected_missing_positive"] == "uncertain" for item in candidate_reviews)
    uncertain_hard = any(item["likely_hard_negative"] == "uncertain" for item in candidate_reviews)
    query_review = value["query_review"]
    query_review["credible_missing_positive_ids"] = credible_ids
    query_review["credible_missing_positive"] = "yes" if credible_ids else "uncertain" if uncertain_missing else "no"
    query_review["likely_hard_negative_ids"] = hard_ids
    query_review["likely_hard_negative_present"] = "yes" if hard_ids else "uncertain" if uncertain_hard else "no"
    if credible_ids:
        query_review["qrels_completeness"] = "incomplete"
        if query_review["label_incompleteness_risk"] in {"none", "low"}:
            query_review["label_incompleteness_risk"] = "moderate"
    query_uncertain = "uncertain" in {
        query_review["credible_missing_positive"],
        query_review["likely_hard_negative_present"],
        query_review["qrels_completeness"],
        query_review["answerability"],
        query_review["label_incompleteness_risk"],
        query_review["sensitive_information_risk"],
    }
    _normalize_judgment_status(query_review, query_uncertain)
    return {
        "gold_reviews": gold_reviews,
        "candidate_reviews": candidate_reviews,
        "query_review": query_review,
    }


def _attach_provenance(
    source: dict[str, Any],
    reviewed: dict[str, Any],
    *,
    provider: str,
    model: str,
    model_revision: str | None,
    session_id: str,
    judged_at: str,
    rubric_sha256: str,
    input_sha256: str,
) -> dict[str, Any]:
    input_row_sha256 = contract._sha256_json(source)
    provenance = {
        "judgment_origin": JUDGMENT_ORIGIN,
        "agent_role": "executor_primary",
        "provider": provider,
        "model": model,
        "model_revision": model_revision,
        "session_id": session_id,
        "round": 1,
        "judged_at": judged_at,
        "rubric_sha256": rubric_sha256,
        "input_artifact_sha256": input_sha256,
        "input_row_sha256": input_row_sha256,
    }
    for item in [*reviewed["gold_reviews"], *reviewed["candidate_reviews"], reviewed["query_review"]]:
        item["provenance"] = provenance
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "audit_id": contract.AUDIT_ID,
        "review_role": "primary",
        "track": source["track"],
        "query_id": source["query_id"],
        "input_row_sha256": input_row_sha256,
        "rubric_sha256": rubric_sha256,
        **reviewed,
    }


async def _review_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    endpoint: str,
    key: str,
    api_version: str,
    deployment: str,
    prompt: str,
    rubric: str,
    source: dict[str, Any],
    ordinal: int,
    total: int,
    raw_root: Path,
    session_id: str,
    input_sha256: str,
    rubric_sha256: str,
    prompt_sha256: str,
    max_tokens: int,
    retries: int,
    pacing_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = _request_body(deployment, prompt, rubric, source, max_tokens)
    raw_path = raw_root / f"{ordinal:04d}.json"
    input_row_sha256 = contract._sha256_json(source)
    if raw_path.is_file():
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes)
        expected = {
            "request_ordinal": ordinal,
            "input_row_sha256": input_row_sha256,
            "input_artifact_sha256": input_sha256,
            "prompt_recipe_sha256": prompt_sha256,
            "rubric_sha256": rubric_sha256,
            "session_id": session_id,
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise PrimaryAuditRunError("cached raw response identity mismatch")
        reviewed = _normalize_model_output(source, json.loads(raw["content"]))
        annotation = _attach_provenance(
            source,
            reviewed,
            provider=PROVIDER,
            model=raw["model_returned"],
            model_revision=raw["model_returned"],
            session_id=session_id,
            judged_at=raw["finished_at"],
            rubric_sha256=rubric_sha256,
            input_sha256=input_sha256,
        )
        log = {
            key: raw[key]
            for key in (
                "track",
                "query_id",
                "input_row_sha256",
                "request_ordinal",
                "attempts",
                "started_at",
                "finished_at",
                "provider",
                "api_surface",
                "deployment_requested",
                "model_returned",
                "session_id",
                "round",
                "prompt_recipe_sha256",
                "rubric_sha256",
                "input_artifact_sha256",
                "usage",
                "sampling",
            )
        }
        log["raw_response_sha256"] = _sha256_bytes(raw_bytes)
        print(f"resumed {ordinal}/{total}", flush=True)
        return annotation, log
    started_at = _utc_now()
    last_error = "request not attempted"
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(
                    endpoint.rstrip("/") + f"/openai/deployments/{deployment}/chat/completions",
                    params={"api-version": api_version},
                    headers={"api-key": key, "content-type": "application/json"},
                    json=body,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    last_error = f"retryable HTTP {response.status_code}"
                    retry_after = response.headers.get("retry-after")
                    wait_seconds = min(2**attempt, 20)
                    if retry_after:
                        with suppress(ValueError):
                            wait_seconds = max(wait_seconds, min(float(retry_after), 60))
                    await asyncio.sleep(wait_seconds)
                    continue
                if response.status_code != 200:
                    code = "unknown"
                    try:
                        code = str(response.json().get("error", {}).get("code") or "unknown")
                    except json.JSONDecodeError:
                        code = "non_json"
                    raise PrimaryAuditRunError(f"non-retryable API response {response.status_code}/{code}")
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                reviewed = _normalize_model_output(source, json.loads(content))
                finished_at = _utc_now()
                model = str(payload.get("model") or deployment)
                annotation = _attach_provenance(
                    source,
                    reviewed,
                    provider=PROVIDER,
                    model=model,
                    model_revision=model,
                    session_id=session_id,
                    judged_at=finished_at,
                    rubric_sha256=rubric_sha256,
                    input_sha256=input_sha256,
                )
                sampling = {
                    "temperature": 0,
                    "top_p": "server_default",
                    "seed": "not_set",
                    "max_completion_tokens": max_tokens,
                    "response_format": "strict_json_schema",
                }
                raw = {
                    "request_ordinal": ordinal,
                    "attempts": attempt,
                    "track": source["track"],
                    "query_id": source["query_id"],
                    "input_row_sha256": input_row_sha256,
                    "input_artifact_sha256": input_sha256,
                    "prompt_recipe_sha256": prompt_sha256,
                    "rubric_sha256": rubric_sha256,
                    "session_id": session_id,
                    "round": 1,
                    "provider": PROVIDER,
                    "api_surface": API_SURFACE,
                    "deployment_requested": deployment,
                    "request_id": response.headers.get("x-request-id") or payload.get("id"),
                    "model_returned": model,
                    "created": payload.get("created"),
                    "finish_reason": payload["choices"][0].get("finish_reason"),
                    "usage": payload.get("usage"),
                    "sampling": sampling,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "content": content,
                }
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_bytes = (json_dumps(raw) + "\n").encode("utf-8")
                raw_path.write_bytes(raw_bytes)
                log = {
                    "track": source["track"],
                    "query_id": source["query_id"],
                    "input_row_sha256": contract._sha256_json(source),
                    "request_ordinal": ordinal,
                    "attempts": attempt,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "provider": PROVIDER,
                    "api_surface": API_SURFACE,
                    "deployment_requested": deployment,
                    "model_returned": model,
                    "session_id": session_id,
                    "round": 1,
                    "prompt_recipe_sha256": prompt_sha256,
                    "rubric_sha256": rubric_sha256,
                    "input_artifact_sha256": input_sha256,
                    "raw_response_sha256": _sha256_bytes(raw_bytes),
                    "usage": payload.get("usage"),
                    "sampling": sampling,
                }
                print(f"completed {ordinal}/{total}", flush=True)
                if pacing_seconds:
                    await asyncio.sleep(pacing_seconds)
                return annotation, log
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, PrimaryAuditRunError) as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt < retries:
                    await asyncio.sleep(min(2**attempt, 20))
                    continue
                break
    raise PrimaryAuditRunError(f"primary review failed after {retries} attempts: {last_error}")


async def run_primary_audit(
    *,
    input_path: Path,
    rubric_path: Path,
    prompt_path: Path,
    output_path: Path,
    log_path: Path,
    raw_root: Path,
    run_manifest_path: Path,
    endpoint_env: str,
    key_env: str,
    api_version_env: str,
    deployment: str,
    session_id: str,
    concurrency: int,
    max_tokens: int,
    retries: int,
    pacing_seconds: float,
) -> dict[str, Any]:
    endpoint = os.environ.get(endpoint_env)
    key = os.environ.get(key_env)
    api_version = os.environ.get(api_version_env)
    if not endpoint or not key or not api_version:
        raise PrimaryAuditRunError("required API environment variables are not set")
    sources = contract._read_jsonl(input_path)
    if len(sources) != contract.REVIEW_PLAN_ROWS:
        raise PrimaryAuditRunError("primary input must contain exactly 120 frozen rows")
    rubric = rubric_path.read_text(encoding="utf-8")
    prompt = prompt_path.read_text(encoding="utf-8")
    rubric_sha256 = _sha256_file(rubric_path)
    if rubric_sha256 != "cadf239c93ab40e583e524bfe8364c655eaad1d8dde07bc57892fa0776dd22b9":
        raise PrimaryAuditRunError("frozen rubric identity mismatch")
    input_sha256 = _sha256_file(input_path)
    prompt_sha256 = _sha256_file(prompt_path)
    started_at = _utc_now()
    timeout = httpx.Timeout(300, connect=30)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _review_one(
                client,
                semaphore,
                endpoint=endpoint,
                key=key,
                api_version=api_version,
                deployment=deployment,
                prompt=prompt,
                rubric=rubric,
                source=source,
                ordinal=ordinal,
                total=len(sources),
                raw_root=raw_root,
                session_id=session_id,
                input_sha256=input_sha256,
                rubric_sha256=rubric_sha256,
                prompt_sha256=prompt_sha256,
                max_tokens=max_tokens,
                retries=retries,
                pacing_seconds=pacing_seconds,
            )
            for ordinal, source in enumerate(sources, 1)
        ]
        completed = await asyncio.gather(*tasks)
    annotations = [item[0] for item in completed]
    logs = [item[1] for item in completed]
    annotation_entry = write_jsonl(output_path, annotations)
    log_entry = write_jsonl(log_path, logs)
    contract.validate_annotations(output_path, input_path, expected_role="primary")
    model_counts: dict[str, int] = {}
    for row in logs:
        model = row["model_returned"]
        model_counts[model] = model_counts.get(model, 0) + 1
    usage = {
        key: sum(int((row.get("usage") or {}).get(key) or 0) for row in logs)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    manifest = {
        "schema_version": "1",
        "audit_id": contract.AUDIT_ID,
        "judgment_origin": JUDGMENT_ORIGIN,
        "provider": PROVIDER,
        "api_surface": API_SURFACE,
        "endpoint_env_name": endpoint_env,
        "key_env_name": key_env,
        "api_version_env_name": api_version_env,
        "deployment_requested": deployment,
        "models_returned": dict(sorted(model_counts.items())),
        "session_id": session_id,
        "round": 1,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "prompt_recipe_sha256": prompt_sha256,
        "rubric_sha256": rubric_sha256,
        "input_artifact_sha256": input_sha256,
        "sampling": {
            "temperature": 0,
            "top_p": "server_default",
            "seed": "not_set",
            "max_completion_tokens": max_tokens,
            "response_format": "strict_json_schema",
        },
        "requests": {**log_entry, "path": log_path.name},
        "raw_responses": {
            "rows": len(list(raw_root.glob("*.json"))),
            "aggregate_sha256": _sha256_bytes(
                "".join(_sha256_file(path) for path in sorted(raw_root.glob("*.json"))).encode("ascii")
            ),
        },
        "annotations": {**annotation_entry, "path": output_path.name},
        "usage": usage,
        "credentials_recorded": False,
        "publication_allowed": False,
    }
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--endpoint-env", default="AZURE_OPENAI_ENDPOINT_BAK")
    parser.add_argument("--key-env", default="AZURE_OPENAI_API_KEY_BAK")
    parser.add_argument("--api-version-env", default="OPENAI_API_VERSION")
    parser.add_argument("--deployment", default="gpt-4o")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pacing-seconds", type=float, default=5)
    args = parser.parse_args(argv)
    manifest = asyncio.run(
        run_primary_audit(
            input_path=args.input,
            rubric_path=args.rubric,
            prompt_path=args.prompt,
            output_path=args.output,
            log_path=args.log,
            raw_root=args.raw_root,
            run_manifest_path=args.run_manifest,
            endpoint_env=args.endpoint_env,
            key_env=args.key_env,
            api_version_env=args.api_version_env,
            deployment=args.deployment,
            session_id=args.session_id,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            retries=args.retries,
            pacing_seconds=args.pacing_seconds,
        )
    )
    safe = {
        "rows": manifest["annotations"]["rows"],
        "annotations_sha256": manifest["annotations"]["sha256"],
        "models_returned": manifest["models_returned"],
        "usage": manifest["usage"],
    }
    print(json.dumps(safe, indent=2, sort_keys=True))
    return 0
