"""Pool ứng viên cho reranker (GĐ14): top-20 của H0 tái dựng từ danh sách top-100 đã lưu của BM25 và dense.

Dùng đúng hàm rrf() và cấu hình đã chốt ở GĐ13 (results/13_selection.json -> final), nên giống hệt H0 mà pipeline
hybrid trả về (đã kiểm ở GĐ13: xác nhận trên 234 câu bằng đúng cách này). Không gọi GPU."""
from germanqa.config import RESULTS_DIR
from germanqa.data.io_utils import read_json
from germanqa.evaluation.run_cache import as_ranked, read_run
from germanqa.retrieval.rrf import rrf

BM25_RUN = "B0_select_top100.jsonl"
DENSE_RUN = "B1_select_top100_384.jsonl"


def final_config():
    return read_json(RESULTS_DIR / "13_selection.json")["final"]


def hybrid_pool(qids, final=None, bm25_run=None, dense_run=None):
    """{question_id: [passage_id theo hạng RRF]} (đủ fusion_top_k đoạn). Câu thiếu trong file top-100 -> KeyError."""
    final = final or final_config()
    bm = bm25_run if bm25_run is not None else as_ranked(read_run(RESULTS_DIR / BM25_RUN))
    dn = dense_run if dense_run is not None else as_ranked(read_run(RESULTS_DIR / DENSE_RUN))
    d = final["depth"]
    return {q: [r["passage_id"] for r in rrf({"bm25": bm[q][:d], "dense": dn[q][:d]}, c=final["rrf_c"],
                                             top_k=final["fusion_top_k"], weights={"dense": final["w_dense"]})]
            for q in qids}
