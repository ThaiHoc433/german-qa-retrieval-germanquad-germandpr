"""GĐ11.5: chạy lại B0 trên TOÀN BỘ dev, so với kết quả đã lưu (results/eval_dev_20261103... + reports/pipeline_dev_predictions.jsonl).
--index fresh : BM25 dựng mới (không dùng tệp)   --index loaded : BM25 nạp từ artifacts/bm25 (dấu vân tay phải khớp).
Không đọc test. Ghi results/11e_B0_rerun_<index>.json (không ghi đè)."""
import argparse
import json
import time

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl
from germanqa.evaluation.report import layer_summary, retrieval_summary
from germanqa.experiment import load_config
from germanqa.experiment_rules import write_once
from germanqa.pipeline import build_pipeline
from germanqa.reader.features import load_examples
from germanqa.retrieval import bm25_store


def main(index, limit):
    ev = load_config("evaluation")["evaluation"]
    t0 = time.perf_counter()
    pipe = build_pipeline(None, PROCESSED_DIR, "bm25", bm25_store_dir=False if index == "fresh" else bm25_store.DEFAULT_DIR)
    t_build = time.perf_counter() - t0
    examples = load_examples(PROCESSED_DIR, "dev")[:limit]
    pipe.answer(examples[0]["question"])
    rows = []
    for ex in examples:
        r = pipe.answer(ex["question"], trace=True)
        rows.append({"question_id": ex["id"], "answer": r.answer, "passage_id": r.passage_id, "retrieved": r.trace["retrieved"],
                     "reader_score": r.reader_score})
    boot = dict(n_boot=ev["bootstrap_samples"], confidence=ev["confidence"])
    e2e = layer_summary({r["question_id"]: {"text": r["answer"]} for r in rows}, examples, **boot)["all_refs"]
    retr = retrieval_summary(rows, read_json(PROCESSED_DIR / "qrels_dev.json"), ev["ks"])

    saved = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_dev_predictions.jsonl")}
    sm = json.loads((RESULTS_DIR / "eval_dev_20261103T163556Z_cb4749bc.json").read_text(encoding="utf-8-sig"))["metrics"]
    d_ans = d_pid = d_ret = 0
    max_rs = 0.0
    for r in rows:
        s = saved[r["question_id"]]
        d_ans += r["answer"] != s["answer"]
        d_pid += r["passage_id"] != s["passage_id"]
        d_ret += [h["passage_id"] for h in r["retrieved"]] != [h["passage_id"] for h in s["retrieved"]]
        max_rs = max(max_rs, abs(r["reader_score"] - s["reader_score"]))
    out = {"index": index, "n": len(rows), "limit": limit, "pipeline_build_seconds": round(t_build, 2),
           "now": {"em": e2e["em"], "f1": e2e["f1"], "hit@5": retr["hit@5"], "hit@20": retr["hit@20"]},
           "saved": {"file": "results/eval_dev_20261103T163556Z_cb4749bc.json", "em": sm["end_to_end"]["all_refs"]["em"],
                     "f1": sm["end_to_end"]["all_refs"]["f1"], "hit@5": sm["retrieval"]["hit@5"], "hit@20": sm["retrieval"]["hit@20"]},
           "question_level": {"answer_differs": d_ans, "selected_passage_differs": d_pid, "retrieved_passage_list_differs": d_ret,
                              "max_abs_reader_score_diff": max_rs}}
    out["metric_diff"] = {k: out["now"][k] - out["saved"][k] for k in out["now"]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if limit is None:
        write_once(RESULTS_DIR / f"11e_B0_rerun_{index}.json", out)
    else:
        print("(--limit: không ghi kết quả)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", choices=("fresh", "loaded"), required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.index, a.limit)
