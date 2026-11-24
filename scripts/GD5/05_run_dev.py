
"""GĐ5.6: chạy pipeline trên TOÀN BỘ dev, lưu dự đoán (kèm top-20 của BM25) để GĐ7 phân loại lỗi.
Hàm answer() chỉ nhận câu hỏi; gold (qrels, đáp án) chỉ được đọc ở bước chấm điểm bên dưới, sau khi dự đoán xong."""
import argparse
import time

from germanqa.config import PROCESSED_DIR, REPORT_DIR
from germanqa.data.io_utils import read_json, write_jsonl
from germanqa.evaluation.retrieval_metrics import evaluate_run, latency_summary
from germanqa.experiment import save_run
from germanqa.pipeline import build_pipeline
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import evaluate_answers


def main(checkpoint, limit):
    pipe = build_pipeline(checkpoint)
    examples = load_examples(PROCESSED_DIR, "dev")[:limit]
    pipe.answer(examples[0]["question"])                       # khởi động, không tính vào thời gian

    rows, t0 = [], time.perf_counter()
    for i, ex in enumerate(examples, 1):
        r = pipe.answer(ex["question"], trace=True)            # chỉ truyền câu hỏi
        rows.append({"question_id": ex["id"], "question": r.question, "answer": r.answer, "status": r.status,
                     "passage_id": r.passage_id, "answer_start": r.answer_start, "answer_end": r.answer_end,
                     "retrieval_score": r.retrieval_score, "reader_score": r.reader_score,
                     "retrieval_time": r.retrieval_time, "reader_time": r.reader_time, "total_time": r.total_time,
                     "retrieved": r.trace["retrieved"] if r.trace else [],
                     "candidates": r.trace["candidates"] if r.trace else []})
        if i % 200 == 0:
            print(f"{i}/{len(examples)}  ({time.perf_counter() - t0:.0f} s)", flush=True)

    # ---- chấm điểm: lúc này mới dùng gold ----
    preds = {r["question_id"]: {"text": r["answer"]} for r in rows}
    qa = evaluate_answers(preds, examples)
    qrels = {q: v for q, v in read_json(PROCESSED_DIR / "qrels_dev.json").items() if q in preds}
    retrieval = evaluate_run({r["question_id"]: [h["passage_id"] for h in r["retrieved"]] for r in rows}, qrels)
    timing = {name: latency_summary([r[name] for r in rows]) for name in ("retrieval_time", "reader_time", "total_time")}
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print("END-TO-END dev:", {k: round(v, 4) for k, v in qa.items()})
    print("Retrieval top-20:", {k: round(v, 4) for k, v in retrieval.items() if k.startswith("hit")})
    print("Thời gian (ms):", {k: {a: round(b, 1) for a, b in v.items()} for k, v in timing.items()})
    print("Trạng thái:", statuses)

    write_jsonl(REPORT_DIR / "pipeline_dev_predictions.jsonl", rows)
    path = save_run(pipe.experiment_id, {"checkpoint": pipe.model_version, "corpus_version": pipe.corpus_version,
                                         "retrieval_top_k": pipe.retrieval_top_k, "reader_top_k": pipe.reader_top_k},
                    {"end_to_end": qa, "retrieval": retrieval, "timing": timing, "statuses": statuses})
    print("đã ghi", REPORT_DIR / "pipeline_dev_predictions.jsonl", "và", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N câu đầu (chạy thử)")
    a = ap.parse_args()
    main(a.checkpoint, a.limit)