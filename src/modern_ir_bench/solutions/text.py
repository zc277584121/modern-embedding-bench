"""Small deterministic text retrieval solutions used by the preview."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit

TOKEN_PATTERN = re.compile(r"[a-z0-9_./-]+")
TextTokenizer = Callable[[str], list[str]]


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _character_ngrams(text: str, size: int) -> Counter[str]:
    normalized = " ".join(_tokens(text))
    if len(normalized) < size:
        return Counter([normalized]) if normalized else Counter()
    return Counter(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm)


def _top_hits(scores: Mapping[str, float], top_k: int) -> list[SearchHit]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [SearchHit(id=item_id, score=float(score)) for item_id, score in ordered[:top_k]]


class BM25Searcher:
    def __init__(
        self,
        resources: Iterable[RetrievalResource[str]],
        *,
        k1: float,
        b: float,
        tokenizer: TextTokenizer,
        tokenizer_id: str,
    ) -> None:
        documents = [(resource.id, tokenizer(resource.value)) for resource in resources]
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id
        self.average_length = sum(len(tokens) for _, tokens in documents) / len(documents)
        frequencies: Counter[str] = Counter()
        for _, tokens in documents:
            frequencies.update(set(tokens))
        self.idf = {
            token: math.log(1 + (len(documents) - count + 0.5) / (count + 0.5)) for token, count in frequencies.items()
        }

    def search_batch(self, queries: list[str], *, top_k: int) -> list[list[SearchHit]]:
        results = []
        for query in queries:
            query_tokens = self.tokenizer(query)
            scores: dict[str, float] = {}
            for item_id, document_tokens in self.documents:
                term_frequency = Counter(document_tokens)
                document_length = len(document_tokens)
                score = 0.0
                for token in query_tokens:
                    frequency = term_frequency[token]
                    if not frequency:
                        continue
                    denominator = frequency + self.k1 * (1 - self.b + self.b * document_length / self.average_length)
                    score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
                scores[item_id] = score
            results.append(_top_hits(scores, top_k))
        return results

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "bm25",
            "tokenizer": self.tokenizer_id,
            "k1": self.k1,
            "b": self.b,
        }


@dataclass(frozen=True, kw_only=True)
class BM25Solution(Solution):
    k1: float = 1.2
    b: float = 0.75
    tokenizer: TextTokenizer = _tokens
    tokenizer_id: str = "ascii-retrieval-v1"

    def prepare(
        self,
        resources: Iterable[RetrievalResource[str]],
    ) -> BM25Searcher:
        return BM25Searcher(
            resources,
            k1=self.k1,
            b=self.b,
            tokenizer=self.tokenizer,
            tokenizer_id=self.tokenizer_id,
        )


class CharacterNGramSearcher:
    def __init__(
        self,
        resources: Iterable[RetrievalResource[str]],
        *,
        size: int,
    ) -> None:
        self.size = size
        self.documents = [
            (resource.id, _character_ngrams(resource.value, size))
            for resource in resources
        ]

    def search_batch(self, queries: list[str], *, top_k: int) -> list[list[SearchHit]]:
        results = []
        for query in queries:
            query_vector = _character_ngrams(query, self.size)
            scores = {item_id: _cosine(query_vector, document_vector) for item_id, document_vector in self.documents}
            results.append(_top_hits(scores, top_k))
        return results

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"backend": "character-ngram", "size": self.size}


@dataclass(frozen=True, kw_only=True)
class CharacterNGramSolution(Solution):
    size: int = 3

    def prepare(
        self,
        resources: Iterable[RetrievalResource[str]],
    ) -> CharacterNGramSearcher:
        return CharacterNGramSearcher(
            resources,
            size=self.size,
        )


class HybridSearcher:
    def __init__(self, primary: Any, secondary: Any, *, alpha: float) -> None:
        self.primary = primary
        self.secondary = secondary
        self.alpha = alpha

    def search_batch(self, queries: list[str], *, top_k: int) -> list[list[SearchHit]]:
        depth = max(top_k, 20)
        primary_results = self.primary.search_batch(queries, top_k=depth)
        secondary_results = self.secondary.search_batch(queries, top_k=depth)
        combined_results = []
        for primary_hits, secondary_hits in zip(
            primary_results,
            secondary_results,
            strict=True,
        ):
            scores: dict[str, float] = {}
            for rank, hit in enumerate(primary_hits, start=1):
                scores[hit.id] = scores.get(hit.id, 0.0) + self.alpha / (60 + rank)
            for rank, hit in enumerate(secondary_hits, start=1):
                scores[hit.id] = scores.get(hit.id, 0.0) + (1 - self.alpha) / (60 + rank)
            combined_results.append(_top_hits(scores, top_k))
        return combined_results

    def close(self) -> None:
        self.primary.close()
        self.secondary.close()

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "hybrid-rrf",
            "alpha": self.alpha,
            "primary": dict(self.primary.metadata),
            "secondary": dict(self.secondary.metadata),
        }


@dataclass(frozen=True, kw_only=True)
class HybridSolution(Solution):
    primary: Solution
    secondary: Solution
    alpha: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between zero and one")

    def prepare(
        self,
        resources: Iterable[RetrievalResource[str]],
    ) -> HybridSearcher:
        return HybridSearcher(
            self.primary.prepare(resources),
            self.secondary.prepare(resources),
            alpha=self.alpha,
        )
