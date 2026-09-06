"""Bounded Gemini primary-label runner for the restricted BRIGHT audit."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mm_embed.benchmark import bright_label_audit as contract
from mm_embed.benchmark.bright_label_audit_primary import (
    PrimaryAuditRunError,
    _attach_provenance,
    _normalize_model_output,
    _output_schema,
    _sha256_bytes,
    _sha256_file,
    _utc_now,
)
from mm_embed.benchmark.retrieval_v01 import json_dumps, write_jsonl

PROVIDER = "google_gemini"
API_SURFACE = "google_genai_generate_content"
JUDGMENT_ORIGIN = "llm_assisted"
FROZEN_RUBRIC_SHA256 = "cadf239c93ab40e583e524bfe8364c655eaad1d8dde07bc57892fa0776dd22b9"
RETRY_INSTRUCTION_VERSION = "coverage-repair-v1"


def _sampling(max_tokens: int, thinking_level: str) -> dict[str, Any]:
    return {
        "temperature": 0,
        "top_p": "server_default",
        "seed": "not_set",
        "max_output_tokens": max_tokens,
        "response_format": "application/json with response_json_schema",
        "thinking_level": thinking_level,
        "retry_instruction_version": RETRY_INSTRUCTION_VERSION,
    }


def _request_material(
    prompt: str,
    rubric: str,
    source: dict[str, Any],
    attempt: int,
) -> tuple[str, str, str]:
    request_variant = "base" if attempt == 1 else RETRY_INSTRUCTION_VERSION
    retry_instruction = ""
    if attempt > 1:
        expected_gold = [item["document_id"] for item in source["gold_documents"]]
        expected_candidates = [item["document_id"] for item in source["candidate_documents"]]
        retry_instruction = (
            "\n\nThis is a deterministic coverage-repair retry. Return every ID in each of the "
            "following arrays exactly once and do not add any ID. Preserve the same rubric.\n"
            f"gold_document_ids={json_dumps(expected_gold)}\n"
            f"candidate_document_ids={json_dumps(expected_candidates)}"
        )
    system_instruction = f"{prompt}\n\nFROZEN RUBRIC:\n{rubric}"
    contents = "Review this one source-only audit row:\n" + json_dumps(source) + retry_instruction
    prompt_identity = _sha256_bytes(
        (json_dumps({"system_instruction": system_instruction, "contents": contents}) + "\n").encode()
    )
    return contents, request_variant, prompt_identity


def _usage(response: Any) -> dict[str, Any] | None:
    if response.usage_metadata is None:
        return None
    return response.usage_metadata.model_dump(mode="json", exclude_none=True)


def _log_from_raw(raw: dict[str, Any], raw_bytes: bytes) -> dict[str, Any]:
    keys = (
        "track",
        "query_id",
        "input_row_sha256",
        "request_ordinal",
        "attempts",
        "started_at",
        "finished_at",
        "provider",
        "api_surface",
        "model_requested",
        "model_returned",
        "model_revision",
        "session_id",
        "round",
        "prompt_recipe_sha256",
        "request_prompt_sha256",
        "rubric_sha256",
        "input_artifact_sha256",
        "usage",
        "sampling",
    )
    log = {key: raw[key] for key in keys}
    log["request_variant"] = raw.get("request_variant", "base")
    log["raw_response_sha256"] = _sha256_bytes(raw_bytes)
    return log


async def _review_one(
    client: Any,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
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
    thinking_level: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from google.genai import errors, types

    raw_path = raw_root / f"{ordinal:04d}.json"
    input_row_sha256 = contract._sha256_json(source)
    if raw_path.is_file():
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes)
        _, request_variant, request_prompt_sha256 = _request_material(
            prompt,
            rubric,
            source,
            int(raw["attempts"]),
        )
        expected = {
            "request_ordinal": ordinal,
            "input_row_sha256": input_row_sha256,
            "input_artifact_sha256": input_sha256,
            "prompt_recipe_sha256": prompt_sha256,
            "rubric_sha256": rubric_sha256,
            "session_id": session_id,
            "model_requested": model,
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise PrimaryAuditRunError("cached raw response identity mismatch")
        if raw.get("request_prompt_sha256") not in {None, request_prompt_sha256}:
            raise PrimaryAuditRunError("cached exact prompt identity mismatch")
        if "request_prompt_sha256" not in raw:
            raw["request_prompt_sha256"] = request_prompt_sha256
            raw["request_variant"] = request_variant
            raw_bytes = (json_dumps(raw) + "\n").encode()
            raw_path.write_bytes(raw_bytes)
        reviewed = _normalize_model_output(source, json.loads(raw["content"]))
        annotation = _attach_provenance(
            source,
            reviewed,
            provider=PROVIDER,
            model=raw["model_returned"],
            model_revision=raw["model_revision"],
            session_id=session_id,
            judged_at=raw["finished_at"],
            rubric_sha256=rubric_sha256,
            input_sha256=input_sha256,
        )
        print(f"resumed {ordinal}/{total}", flush=True)
        return annotation, _log_from_raw(raw, raw_bytes)

    started_at = _utc_now()
    last_error = "request not attempted"
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                contents, request_variant, request_prompt_sha256 = _request_material(
                    prompt,
                    rubric,
                    source,
                    attempt,
                )
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=f"{prompt}\n\nFROZEN RUBRIC:\n{rubric}",
                        temperature=0,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                        response_json_schema=_output_schema(),
                        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                    ),
                )
                content = response.text
                if not content:
                    raise PrimaryAuditRunError("model response did not contain JSON text")
                reviewed = _normalize_model_output(source, json.loads(content))
                finished_at = _utc_now()
                model_returned = str(response.model_version or model)
                annotation = _attach_provenance(
                    source,
                    reviewed,
                    provider=PROVIDER,
                    model=model_returned,
                    model_revision=model_returned,
                    session_id=session_id,
                    judged_at=finished_at,
                    rubric_sha256=rubric_sha256,
                    input_sha256=input_sha256,
                )
                raw = {
                    "request_ordinal": ordinal,
                    "attempts": attempt,
                    "track": source["track"],
                    "query_id": source["query_id"],
                    "input_row_sha256": input_row_sha256,
                    "input_artifact_sha256": input_sha256,
                    "prompt_recipe_sha256": prompt_sha256,
                    "request_prompt_sha256": request_prompt_sha256,
                    "rubric_sha256": rubric_sha256,
                    "session_id": session_id,
                    "round": 1,
                    "provider": PROVIDER,
                    "api_surface": API_SURFACE,
                    "model_requested": model,
                    "model_returned": model_returned,
                    "model_revision": model_returned,
                    "response_id": response.response_id,
                    "finish_reason": (
                        response.candidates[0].finish_reason.value
                        if response.candidates and response.candidates[0].finish_reason
                        else None
                    ),
                    "usage": _usage(response),
                    "sampling": _sampling(max_tokens, thinking_level),
                    "request_variant": request_variant,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "content": content,
                }
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_bytes = (json_dumps(raw) + "\n").encode()
                raw_path.write_bytes(raw_bytes)
                print(f"completed {ordinal}/{total}", flush=True)
                if pacing_seconds:
                    await asyncio.sleep(pacing_seconds)
                return annotation, _log_from_raw(raw, raw_bytes)
            except (
                errors.APIError,
                json.JSONDecodeError,
                KeyError,
                PrimaryAuditRunError,
                TypeError,
                ValueError,
            ) as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt < retries:
                    await asyncio.sleep(min(2**attempt, 60))
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
    key_env: str,
    model: str,
    session_id: str,
    concurrency: int,
    max_tokens: int,
    retries: int,
    pacing_seconds: float,
    thinking_level: str,
) -> dict[str, Any]:
    key = os.environ.get(key_env)
    if not key:
        raise PrimaryAuditRunError("required API key environment variable is not set")
    if concurrency < 1 or max_tokens < 1 or retries < 1 or pacing_seconds < 0:
        raise PrimaryAuditRunError("execution controls must be positive (pacing may be zero)")

    from google import genai

    sources = contract._read_jsonl(input_path)
    if len(sources) != contract.REVIEW_PLAN_ROWS:
        raise PrimaryAuditRunError("primary input must contain exactly 120 frozen rows")
    rubric = rubric_path.read_text(encoding="utf-8")
    prompt = prompt_path.read_text(encoding="utf-8")
    rubric_sha256 = _sha256_file(rubric_path)
    if rubric_sha256 != FROZEN_RUBRIC_SHA256:
        raise PrimaryAuditRunError("frozen rubric identity mismatch")
    input_sha256 = _sha256_file(input_path)
    prompt_sha256 = _sha256_file(prompt_path)
    started_at = _utc_now()
    semaphore = asyncio.Semaphore(concurrency)
    client = genai.Client(api_key=key)
    try:
        completed = await asyncio.gather(
            *(
                _review_one(
                    client,
                    semaphore,
                    model=model,
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
                    thinking_level=thinking_level,
                )
                for ordinal, source in enumerate(sources, 1)
            )
        )
    finally:
        await client.aio.aclose()

    annotations = [item[0] for item in completed]
    logs = [item[1] for item in completed]
    annotation_entry = write_jsonl(output_path, annotations)
    log_entry = write_jsonl(log_path, logs)
    contract.validate_annotations(output_path, input_path, expected_role="primary")
    model_counts: dict[str, int] = {}
    for row in logs:
        returned = row["model_returned"]
        model_counts[returned] = model_counts.get(returned, 0) + 1
    usage = {
        key: sum(int((row.get("usage") or {}).get(key) or 0) for row in logs)
        for key in (
            "prompt_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "total_token_count",
        )
    }
    raw_paths = sorted(raw_root.glob("*.json"))
    manifest = {
        "schema_version": "1",
        "audit_id": contract.AUDIT_ID,
        "judgment_origin": JUDGMENT_ORIGIN,
        "provider": PROVIDER,
        "api_surface": API_SURFACE,
        "key_env_name": key_env,
        "model_requested": model,
        "models_returned": dict(sorted(model_counts.items())),
        "session_id": session_id,
        "round": 1,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "prompt_recipe_sha256": prompt_sha256,
        "rubric_sha256": rubric_sha256,
        "input_artifact_sha256": input_sha256,
        "sampling": _sampling(max_tokens, thinking_level),
        "requests": {**log_entry, "path": log_path.name},
        "raw_responses": {
            "rows": len(raw_paths),
            "aggregate_sha256": _sha256_bytes("".join(_sha256_file(path) for path in raw_paths).encode("ascii")),
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
    parser.add_argument("--key-env", default="GEMINI_API_KEY")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--pacing-seconds", type=float, default=1)
    parser.add_argument("--thinking-level", choices=("low", "medium", "high"), default="low")
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
            key_env=args.key_env,
            model=args.model,
            session_id=args.session_id,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            retries=args.retries,
            pacing_seconds=args.pacing_seconds,
            thinking_level=args.thinking_level,
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
