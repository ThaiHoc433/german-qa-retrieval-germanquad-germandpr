"""Hybrid retriever (GĐ13): BM25 + dense hợp nhất bằng RRF.

Giao diện giống BM25Retriever và DenseRetriever: retrieve(question, top_k) -> [{passage_id, score, rank}], nên
QAPipeline không đổi. Mỗi dòng có thêm:
  sources      {'bm25': hạng, 'dense': hạng}  (nhánh vắng thì KHÔNG có khóa)
  bm25_score   điểm BM25 gốc, None nếu đoạn không nằm trong top bm25_top_k của BM25
  dense_score  tích vô hướng dense gốc, None nếu đoạn không nằm trong top dense_top_k của dense
`score` là điểm RRF: KHÔNG so được với bm25_score hay dense_score và KHÔNG phải xác suất."""
import time

from germanqa.retrieval.rrf import rrf

BRANCHES = ("bm25", "dense")


class HybridRetriever:
    def __init__(self, bm25, dense, bm25_top_k=50, dense_top_k=50, fusion_top_k=20, rrf_c=60, w_dense=1.0):
        if min(bm25_top_k, dense_top_k, fusion_top_k) < 1:
            raise ValueError("bm25_top_k, dense_top_k, fusion_top_k phải >= 1")
        if rrf_c < 0 or w_dense < 0:
            raise ValueError("rrf_c và w_dense phải >= 0")
        self.bm25, self.dense = bm25, dense                 # đối tượng ĐÃ dựng sẵn: không nạp encoder lần nữa
        self.bm25_top_k, self.dense_top_k, self.fusion_top_k = bm25_top_k, dense_top_k, fusion_top_k
        self.rrf_c, self.w_dense = rrf_c, w_dense
        self.last_timing = {"bm25": 0.0, "dense": 0.0, "fusion": 0.0}

    @classmethod
    def from_config(cls, bm25, dense, cfg):
        keys = ("bm25_top_k", "dense_top_k", "fusion_top_k", "rrf_c", "w_dense")
        return cls(bm25, dense, **{k: cfg[k] for k in keys if k in cfg})

    def config(self):
        return {"bm25_top_k": self.bm25_top_k, "dense_top_k": self.dense_top_k, "fusion_top_k": self.fusion_top_k,
                "rrf_c": self.rrf_c, "w_dense": self.w_dense}

    def retrieve(self, question, top_k=None):
        k = self.fusion_top_k if top_k is None else top_k
        t = time.perf_counter()
        b_rows = self.bm25.retrieve(question, self.bm25_top_k)
        t_b = time.perf_counter() - t
        t = time.perf_counter()
        d_rows = self.dense.retrieve(question, self.dense_top_k)
        t_d = time.perf_counter() - t
        t = time.perf_counter()
        fused = rrf({"bm25": [r["passage_id"] for r in b_rows], "dense": [r["passage_id"] for r in d_rows]},
                    c=self.rrf_c, top_k=k, weights={"dense": self.w_dense})
        b_score = {r["passage_id"]: r["score"] for r in b_rows}
        d_score = {r["passage_id"]: r["score"] for r in d_rows}
        out = [{"passage_id": f["passage_id"], "score": f["score"], "rank": f["rank"], "sources": f["sources"],
                "bm25_score": b_score.get(f["passage_id"]), "dense_score": d_score.get(f["passage_id"])} for f in fused]
        self.last_timing = {"bm25": t_b, "dense": t_d, "fusion": time.perf_counter() - t}
        return out
