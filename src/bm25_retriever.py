"""BM25 baseline — module 4 của
docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md.

Nhận corpus đã build sẵn (không tự dedup), tokenize trên `normalized_text`
(đã chuẩn hóa ở module Data), trả `RetrievalResult`-shape (passage_id, score).
"""
import bm25s

from src.corpus import normalize_text

DEFAULT_TOP_N = 50


class BM25Retriever:
    def __init__(self, passage_ids: list[str], normalized_texts: list[str]):
        self.passage_ids = passage_ids
        tokens = bm25s.tokenize(normalized_texts, stopwords=None, show_progress=False)
        self.vocab = set(tokens.vocab.keys())
        self._retriever = bm25s.BM25()
        self._retriever.index(tokens, show_progress=False)

    def search(self, query_text: str, top_n: int = DEFAULT_TOP_N) -> list[tuple[str, float]]:
        norm_query = normalize_text(query_text)
        query_tokens = bm25s.tokenize(
            [norm_query], stopwords=None, return_ids=False, show_progress=False
        )[0]

        if not query_tokens:
            # quy ước 4.5: query rỗng sau normalize -> trả rỗng, không raise lỗi
            return []

        if not any(t in self.vocab for t in query_tokens):
            # quy ước 4.5: toàn bộ query OOV -> điểm 0 đều nhau, tie-break ổn định
            # theo passage_id (KHÔNG dựa vào tie-break nội bộ không đảm bảo của bm25s)
            ranked = sorted(self.passage_ids)[:top_n]
            return [(pid, 0.0) for pid in ranked]

        k = min(top_n, len(self.passage_ids))
        results, scores = self._retriever.retrieve(
            [query_tokens], corpus=self.passage_ids, k=k, show_progress=False
        )
        return list(zip(results[0].tolist(), scores[0].tolist()))
