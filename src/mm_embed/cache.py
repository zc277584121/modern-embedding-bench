"""Disk-based embedding cache.

Caches embedding results keyed by (model, input_hash, dimensions, task_type)
to avoid redundant API calls and GPU computation.

Cache layout:
    data/embedding_cache/<provider>/<model_hash>/<input_hash>.npy
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "embedding_cache"


def _sanitize(name: str, max_len: int = 60) -> str:
    """Make a string safe for use as a directory name."""
    safe = name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    if len(safe) > max_len:
        h = hashlib.md5(name.encode()).hexdigest()[:8]
        safe = safe[:max_len - 9] + "_" + h
    return safe


def _hash_content(content: Any) -> str:
    """Hash input content (text string, bytes, or Path) to a stable hex string."""
    h = hashlib.sha256()
    if isinstance(content, str):
        h.update(content.encode("utf-8"))
    elif isinstance(content, bytes):
        h.update(content)
    elif isinstance(content, Path):
        if content.exists():
            h.update(content.read_bytes())
        else:
            h.update(str(content).encode("utf-8"))
    else:
        h.update(str(content).encode("utf-8"))
    return h.hexdigest()


def make_cache_key(
    model_name: str,
    inputs_content: list[Any],
    modalities: list[str],
    dimensions: int | None,
    task_type: str | None,
) -> str:
    """Build a single hash key for a batch of inputs."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(json.dumps(dimensions).encode())
    h.update(json.dumps(task_type).encode())

    for content, modality in zip(inputs_content, modalities):
        h.update(modality.encode("utf-8"))
        h.update(_hash_content(content).encode("utf-8"))

    return h.hexdigest()


def get_cache_dir(provider_name: str, model_name: str) -> Path:
    """Return the cache directory for a provider/model combo."""
    d = _CACHE_ROOT / _sanitize(provider_name) / _sanitize(model_name)
    return d


def cache_get(
    provider_name: str,
    model_name: str,
    cache_key: str,
) -> np.ndarray | None:
    """Look up a cached embedding array. Returns None on miss."""
    path = get_cache_dir(provider_name, model_name) / f"{cache_key}.npy"
    if path.exists():
        try:
            arr = np.load(path)
            logger.info("Cache HIT: %s/%s [%s...] shape=%s",
                        provider_name, model_name, cache_key[:12], arr.shape)
            return arr
        except Exception as e:
            logger.warning("Cache read error for %s: %s", path, e)
            return None
    return None


def cache_put(
    provider_name: str,
    model_name: str,
    cache_key: str,
    embeddings: np.ndarray,
) -> None:
    """Store an embedding array in the cache."""
    d = get_cache_dir(provider_name, model_name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{cache_key}.npy"
    try:
        fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".npy.tmp")
        with os.fdopen(fd, "wb") as f:
            np.save(f, embeddings)
        os.rename(tmp_path, path)
        logger.debug("Cache STORE: %s shape=%s", path, embeddings.shape)
    except Exception as e:
        logger.warning("Cache write error for %s: %s", path, e)
        # Clean up temp file on failure
        try:
            os.remove(tmp_path)
        except Exception:
            pass
