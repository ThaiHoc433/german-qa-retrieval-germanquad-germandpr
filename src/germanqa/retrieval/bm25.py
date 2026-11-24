
"""BM25 Okapi tự cài đặt (GĐ3)."""
import math
from collections import Counter

import numpy as np

from germanqa.retrieval.tokenizer import GERMAN_STOPWORDS, tokenize


class BM25:
    """Chỉ mục nghịch đảo: với mỗi từ, lưu danh sách (đoạn nào, điểm đóng góp).

    Điểm của đoạn d với truy vấn q:
        sum_{t in q}  idf(t) * tf(t,d) * (k1+1) / ( tf(t,d) + k1 * (1 - b + b * |d|/avgdl) )
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    @classmethod
    def from_parts(cls, k1, b, doc_ids, weights):
        """Dựng lại từ phần đã lưu: weights = {từ: (mảng chỉ số đoạn, mảng điểm đóng góp)}."""
        self = cls(k1, b)
        self.doc_ids, self.weights, self.n = list(doc_ids), weights, len(doc_ids)
        return self

    def fit(self, doc_ids, token_lists):
        self.doc_ids = list(doc_ids)
        n = len(self.doc_ids)
        lengths = np.array([len(t) for t in token_lists], dtype=np.float64)
        avgdl = lengths.mean() if n else 0.0

        postings = {}                                   # từ -> ([chỉ số đoạn], [tf])
        for i, tokens in enumerate(token_lists):
            for term, tf in Counter(tokens).items():
                idx, tfs = postings.setdefault(term, ([], []))
                idx.append(i)
                tfs.append(tf)

        self.weights = {}                               # từ -> (mảng chỉ số, mảng điểm)
        for term, (idx, tfs) in postings.items():
            idx = np.asarray(idx, dtype=np.int64)
            tf = np.asarray(tfs, dtype=np.float64)
            df = len(idx)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            norm = self.k1 * (1.0 - self.b + self.b * lengths[idx] / avgdl)
            self.weights[term] = (idx, idf * tf * (self.k1 + 1.0) / (tf + norm))
        self.n = n
        return self

    def scores(self, query_tokens):
        """Điểm BM25 của mọi đoạn (mảng độ dài n)."""
        s = np.zeros(self.n, dtype=np.float64)
        for term, qtf in Counter(query_tokens).items():
            entry = self.weights.get(term)
            if entry is not None:
                s[entry[0]] += qtf * entry[1]
        return s

    def top_k(self, query_tokens, k):
        """Trả về [(doc_id, score)] giảm dần theo điểm; đồng điểm thì theo thứ tự doc_ids.
        Chỉ trả đoạn có điểm > 0; truy vấn rỗng -> []."""
        s = self.scores(query_tokens)
        order = np.lexsort((np.arange(self.n), -s))     # khóa chính: -điểm, khóa phụ: chỉ số
        out = []
        for i in order[:max(k, 0)]:
            if s[i] <= 0:
                break
            out.append((self.doc_ids[i], float(s[i])))
        return out


class BM25Retriever:
    """Bọc BM25 thành giao diện retrieve(question, top_k) dùng chung với dense ở GĐ12."""

    def __init__(self, corpus, k1=1.5, b=0.75, use_stopwords=False):
        self.stopwords = GERMAN_STOPWORDS if use_stopwords else None
        ids = [r["passage_id"] for r in corpus]
        tokens = [tokenize(r["retrieval_text"], self.stopwords) for r in corpus]
        self.index = BM25(k1, b).fit(ids, tokens)

    @classmethod
    def from_index(cls, index, use_stopwords=False):
        """Bọc một chỉ mục BM25 đã dựng/nạp sẵn (GĐ11.2), không tách từ lại cả corpus."""
        self = cls.__new__(cls)
        self.stopwords = GERMAN_STOPWORDS if use_stopwords else None
        self.index = index
        return self

    def retrieve(self, question, top_k=20):
        q = tokenize(question, self.stopwords)
        return [{"passage_id": pid, "score": sc, "rank": r}
                for r, (pid, sc) in enumerate(self.index.top_k(q, top_k), start=1)]
