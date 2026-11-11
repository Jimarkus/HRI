"""Picks the book that best fits a spoken request, from the catalog in books.json.

Two backends behind one `recommend(text)` call:
  * "embedding" - sentence-embedding similarity between the request and each book description.
  * "hybrid"    - embeddings shortlist the closest few books, then Laya (attention-based
                  decision model) makes the final choice among them.

    python recommender.py "something about money"     # try it without the mic or camera
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("LAYA_CUDA_AMP", "fp16")  # halves Laya's GPU memory; RTX 20-series has no bf16

CATALOG_PATH = Path(__file__).resolve().parent / "books.json"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Book:
    title: str
    author: str = ""
    description: str = ""
    # Optional fallback: place on the shelf counted from the left (1 = leftmost). Only used
    # when the camera can't read this book's spine, or with --no-vision.
    position: int | None = None

    @property
    def text(self) -> str:
        """What the models read about this book."""
        by = f" by {self.author}" if self.author else ""
        return f"{self.title}{by}. {self.description}".strip()


@dataclass
class Recommendation:
    book: Book
    score: float                 # embedding: cosine similarity; hybrid: Laya's probability
    ranking: list[tuple[Book, float]] = field(default_factory=list)  # best first, includes `book`
    backend: str = ""
    latency_ms: float = 0.0


def load_catalog(path: Path = CATALOG_PATH) -> list[Book]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    books = [Book(**b) for b in data["books"]]
    positions = [b.position for b in books if b.position is not None]
    if len(set(positions)) != len(positions):
        raise ValueError(f"{path}: two books share the same position")
    missing = [b.title for b in books if not b.description.strip()]
    if missing:
        print(f"WARNING: no description for {', '.join(missing)}; matching on title only.")
    return books


class EmbeddingRecommender:
    name = "embedding"

    def __init__(self, books: list[Book]):
        from sentence_transformers import SentenceTransformer
        self.books = books
        # CPU on purpose: ~20 ms per request, and it leaves the GPU memory for Whisper and Laya.
        self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.book_vectors = self.model.encode([b.text for b in books], normalize_embeddings=True)

    def rank(self, text: str) -> list[tuple[Book, float]]:
        query = self.model.encode([QUERY_PREFIX + text], normalize_embeddings=True)[0]
        scores = self.book_vectors @ query
        order = scores.argsort()[::-1]
        return [(self.books[i], float(scores[i])) for i in order]

    def recommend(self, text: str) -> Recommendation:
        t0 = time.perf_counter()
        ranking = self.rank(text)
        return Recommendation(ranking[0][0], ranking[0][1], ranking, self.name,
                              (time.perf_counter() - t0) * 1000)


class HybridRecommender:
    name = "hybrid"

    def __init__(self, books: list[Book], shortlist: int = 5, device: str | None = None):
        from laya import Router
        self.embedder = EmbeddingRecommender(books)
        self.shortlist = shortlist
        self.router = Router(device=device, max_loaded=1, default="english")
        self.router.load("english")

    def recommend(self, text: str) -> Recommendation:
        t0 = time.perf_counter()
        candidates = [book for book, _ in self.embedder.rank(text)[:self.shortlist]]
        by_label = {f"book_{i}": b for i, b in enumerate(candidates, start=1)}
        question = {"book": {
            "type": "choice",
            "instructions": "A person asked a librarian for a book. Which book fits their request best?",
            "criteria": {label: b.text for label, b in by_label.items()},
        }}
        answer = self.router.predict(text, question, model="english")["answers"]["book"]
        ranking = sorted(((by_label[label], float(p)) for label, p in answer["probabilities"].items()),
                         key=lambda pair: pair[1], reverse=True)
        return Recommendation(ranking[0][0], ranking[0][1], ranking, self.name,
                              (time.perf_counter() - t0) * 1000)


def build_recommender(backend: str = "embedding", books: list[Book] | None = None):
    books = books or load_catalog()
    if backend == "hybrid":
        return HybridRecommender(books)
    return EmbeddingRecommender(books)


if __name__ == "__main__":
    request = " ".join(sys.argv[1:]) or "I want to learn about money"
    for backend in ("embedding", "hybrid"):
        rec = build_recommender(backend).recommend(request)
        print(f"[{backend}] {rec.book.title}  (score {rec.score:.2f}, {rec.latency_ms:.0f} ms)")
        for book, score in rec.ranking[1:3]:
            print(f"      then {book.title} ({score:.2f})")
