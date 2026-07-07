"""Upload prepared folders to Hugging Face Hub."""

from __future__ import annotations

import os
import socket
from pathlib import Path


TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


def get_hf_token() -> str | None:
    for name in TOKEN_ENV_NAMES:
        token = os.environ.get(name)
        if token:
            return token
    return None


def upload_hf_folder(
    *,
    folder: str | Path,
    repo_id: str,
    repo_type: str,
    private: bool = False,
    space_sdk: str = "gradio",
    commit_message: str = "Update benchmark artifacts",
    dataset_repo_id_for_space: str | None = None,
    resolve_hf_ip: str | None = None,
) -> str:
    """Create or update a Hugging Face repo from a local folder."""
    if repo_type not in {"dataset", "space"}:
        raise ValueError("repo_type must be 'dataset' or 'space'")

    if resolve_hf_ip:
        patch_huggingface_dns(resolve_hf_ip)

    token = get_hf_token()
    if not token:
        names = ", ".join(TOKEN_ENV_NAMES)
        raise RuntimeError(f"Missing Hugging Face token. Set one of: {names}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    create_kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "private": private,
        "exist_ok": True,
    }
    if repo_type == "space":
        create_kwargs["space_sdk"] = space_sdk

    api.create_repo(**create_kwargs)
    commit_info = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        ignore_patterns=["__pycache__/*", "**/__pycache__/*", "*.pyc", ".DS_Store"],
    )

    if repo_type == "space" and dataset_repo_id_for_space:
        _set_space_variable(api, repo_id, "DATASET_REPO_ID", dataset_repo_id_for_space)

    return getattr(commit_info, "commit_url", str(commit_info))


def _set_space_variable(api, repo_id: str, key: str, value: str) -> None:
    """Set a Space variable when the installed hub client supports it."""
    if hasattr(api, "add_space_variable"):
        api.add_space_variable(repo_id=repo_id, key=key, value=value)


def patch_huggingface_dns(ip_address: str) -> None:
    """Resolve huggingface.co to a known IP for this Python process only."""
    original_getaddrinfo = socket.getaddrinfo

    def patched_getaddrinfo(host, port, *args, **kwargs):
        if host == "huggingface.co":
            return original_getaddrinfo(ip_address, port, *args, **kwargs)
        return original_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = patched_getaddrinfo
