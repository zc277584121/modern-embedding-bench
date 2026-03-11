"""Evaluation metrics for embedding quality assessment."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def cosine_similarity_matrix(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        queries: shape (n_queries, dim)
        candidates: shape (n_candidates, dim)

    Returns:
        Similarity matrix of shape (n_queries, n_candidates)
    """
    queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    candidates_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10)
    return queries_norm @ candidates_norm.T


def recall_at_k(sim_matrix: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """Recall@K — fraction of queries where the true match is in top-K results.

    Args:
        sim_matrix: shape (n_queries, n_candidates)
        ground_truth: shape (n_queries,) — index of correct candidate for each query
        k: Number of top results to consider

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    n_queries = sim_matrix.shape[0]
    top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    hits = sum(1 for i in range(n_queries) if ground_truth[i] in top_k_indices[i])
    return hits / n_queries


def ndcg_at_k(sim_matrix: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """NDCG@K — Normalized Discounted Cumulative Gain.

    Assumes binary relevance: only one relevant document per query.

    Args:
        sim_matrix: shape (n_queries, n_candidates)
        ground_truth: shape (n_queries,) — index of correct candidate for each query
        k: Number of top results to consider

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    n_queries = sim_matrix.shape[0]
    top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    ndcg_sum = 0.0
    for i in range(n_queries):
        for rank, idx in enumerate(top_k_indices[i]):
            if idx == ground_truth[i]:
                ndcg_sum += 1.0 / np.log2(rank + 2)  # +2 because rank is 0-indexed
                break
    return ndcg_sum / n_queries


def mrr(sim_matrix: np.ndarray, ground_truth: np.ndarray) -> float:
    """Mean Reciprocal Rank.

    Args:
        sim_matrix: shape (n_queries, n_candidates)
        ground_truth: shape (n_queries,) — index of correct candidate for each query

    Returns:
        MRR score (0.0 to 1.0)
    """
    n_queries = sim_matrix.shape[0]
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    rr_sum = 0.0
    for i in range(n_queries):
        for rank, idx in enumerate(sorted_indices[i]):
            if idx == ground_truth[i]:
                rr_sum += 1.0 / (rank + 1)
                break
    return rr_sum / n_queries


def dimension_retention_score(
    full_embeddings: np.ndarray,
    truncated_embeddings: np.ndarray,
) -> float:
    """Measure how well truncated (MRL) embeddings preserve similarity structure.

    Computes Spearman rank correlation between pairwise similarities at full
    dimensions vs truncated dimensions.

    Args:
        full_embeddings: shape (n, full_dim)
        truncated_embeddings: shape (n, reduced_dim)

    Returns:
        Spearman correlation (0.0 to 1.0)
    """
    from scipy import stats

    full_sim = cosine_similarity_matrix(full_embeddings, full_embeddings)
    trunc_sim = cosine_similarity_matrix(truncated_embeddings, truncated_embeddings)

    # Extract upper triangle (exclude diagonal)
    mask = np.triu_indices(full_sim.shape[0], k=1)
    corr, _ = stats.spearmanr(full_sim[mask], trunc_sim[mask])
    return float(corr)


def modality_gap(text_embeddings: np.ndarray, image_embeddings: np.ndarray) -> float:
    """Measure the modality gap between text and image embeddings.

    Computed as the L2 distance between the centroids of text and image clusters.

    Args:
        text_embeddings: shape (n_text, dim)
        image_embeddings: shape (n_image, dim)

    Returns:
        L2 distance between centroids (lower = better alignment)
    """
    text_centroid = text_embeddings.mean(axis=0)
    image_centroid = image_embeddings.mean(axis=0)
    return float(np.linalg.norm(text_centroid - image_centroid))
