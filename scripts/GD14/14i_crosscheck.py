"""GĐ14.6b: kiểm lại phép so 'đầu-cuối so với điểm đã lưu đệm' của 14f, vốn ghép cặp theo hạng nên ghép nhầm khi pool thật khác pool dựng lại.
Ở đây ghép theo passage_id, và điều tra nguyên nhân pool khác bằng cách so điểm BM25/dense thật với file top-100 đã lưu.
Ghi results/14i_crosscheck.json (không ghi đè). Chỉ dev-unseen, không test."""
import numpy as np

from germanqa.config import REPORT_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.rerank import analysis as A
from germanqa.rerank.pool import BM25_RUN, DENSE_RUN, hybrid_pool


def main():
    out_path = RESULTS_DIR / "14i_crosscheck.json"
    if out_path.exists():
        raise SystemExit(f"{out_path.name} đã tồn tại")
    unseen = list(read_json(RESULTS_DIR / "select_question_ids.json")["dev"])
    sc = A.load_scores(RESULTS_DIR / "14c_scores_bge_m3.jsonl", unseen)
    rows = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_dev_unseen_H1_predictions.jsonl")}
    files = {n: {r["question_id"]: dict((p, s) for p, s in r["hits"]) for r in read_jsonl(RESULTS_DIR / f)} for n, f in (("bm25", BM25_RUN), ("dense", DENSE_RUN))}
    pool = hybrid_pool(unseen)
    same, diff_order_only, diff_set = [], [], []
    score_diff, bm_diff, dn_diff = [], [], []
    for q in unseen:
        live = [h["passage_id"] for h in rows[q]["retrieved"]]
        if live == pool[q]:
            same.append(q)
            new = {x["passage_id"]: x["rerank_score"] for x in rows[q]["reranked"]}
            score_diff += [abs(new[p] - s) for p, s in zip(sc[q]["passage_ids"], sc[q]["score"])]
        else:
            (diff_order_only if set(live) == set(pool[q]) else diff_set).append(q)
            for h in rows[q]["retrieved"]:
                if h.get("bm25_score") is not None and h["passage_id"] in files["bm25"][q]:
                    bm_diff.append(abs(h["bm25_score"] - files["bm25"][q][h["passage_id"]]))
                if h.get("dense_score") is not None and h["passage_id"] in files["dense"][q]:
                    dn_diff.append(abs(h["dense_score"] - files["dense"][q][h["passage_id"]]))
    def stat(x):
        return {"n": len(x), "max": float(max(x)) if x else None, "n_nonzero": int(sum(v > 0 for v in x))}
    res = {"n": len(unseen), "same_pool": len(same), "pool_order_differs_same_set": len(diff_order_only), "pool_set_differs": len(diff_set),
           "rerank_score_abs_diff_on_same_pool": stat(score_diff),
           "live_vs_file_bm25_score_diff_on_differing_pools": stat(bm_diff),
           "live_vs_file_dense_score_diff_on_differing_pools": stat(dn_diff),
           "note": "Thay cho trường end_to_end_vs_cached_scores của 14f (ghép theo hạng nên sai ở các câu pool khác). "
                   "Hit@5 live = offline cho cả H0 và H1 (xem 14f/14e)."}
    write_json(out_path, res)
    print(res)


if __name__ == "__main__":
    main()
