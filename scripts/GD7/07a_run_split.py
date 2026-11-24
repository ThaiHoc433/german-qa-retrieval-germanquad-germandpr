"""GĐ7.1/7.7: đánh giá cả ba tầng trên một tập (dev hoặc test) với cấu hình ĐÃ ĐÓNG BĂNG ở configs/*.yaml.
Test chỉ chạy khi có --confirm-test và chỉ một lần (file dự đoán đã có thì từ chối, trừ khi --overwrite)."""
import argparse
import sys
import time

from germanqa.config import PROCESSED_DIR, REPORT_DIR
from germanqa.data.io_utils import read_json, write_jsonl
from germanqa.evaluation.paired import retrieval_ci
from germanqa.evaluation.report import (hardware_info, layer_summary, markdown_tables,
                                        retrieval_summary, timing_summary)
from germanqa.experiment import load_config, new_experiment_id, save_run
from germanqa.experiment_rules import guard_or_exit
from germanqa.pipeline import build_pipeline
from germanqa.reader.features import load_examples


def main(split, checkpoint, limit, confirm_test, overwrite):
    pred_path = REPORT_DIR / f"pipeline_{split}_predictions.jsonl"
    if split == "test":
        if pred_path.exists() and not overwrite:
            sys.exit(f"{pred_path} đã có: test đã chạy rồi. Chạy lại là tinh chỉnh theo test; chỉ làm khi có --overwrite.")
        guard_or_exit("07a_run_split.py", "pipeline end-to-end trên test với cấu hình đã đóng băng", confirm_test,
                      files=["qa_test.jsonl", "qrels_test.json", pred_path.name])
    ev = load_config("evaluation")["evaluation"]
    pipe = build_pipeline(checkpoint)
    examples = load_examples(PROCESSED_DIR, split)[:limit]
    pipe.answer(examples[0]["question"])                       # khởi động, không tính thời gian

    rows = []
    for i, ex in enumerate(examples, 1):
        r = pipe.answer(ex["question"], trace=True)            # chỉ truyền câu hỏi; gold chỉ đọc ở bước chấm bên dưới
        rows.append({"question_id": ex["id"], "question": r.question, "answer": r.answer, "status": r.status,
                     "passage_id": r.passage_id, "answer_start": r.answer_start, "answer_end": r.answer_end,
                     "retrieval_score": r.retrieval_score, "reader_score": r.reader_score,
                     "retrieval_time": r.retrieval_time, "reader_time": r.reader_time, "total_time": r.total_time,
                     "retrieved": r.trace["retrieved"], "candidates": r.trace["candidates"]})
        if i % 500 == 0:
            print(f"{i}/{len(examples)}", flush=True)

    # ---- chấm điểm ----
    boot = dict(n_boot=ev["bootstrap_samples"], confidence=ev["confidence"])
    qrels = read_json(PROCESSED_DIR / f"qrels_{split}.json")
    summary = {"retrieval": retrieval_summary(rows, qrels, ev["ks"]),
               "retrieval_ci": retrieval_ci(rows, qrels, ev["ks"], **boot),      # thêm 2026-11-13
               "timing": timing_summary(rows)}
    summary["end_to_end"] = layer_summary({r["question_id"]: {"text": r["answer"]} for r in rows}, examples, **boot)
    reader = pipe.reader
    feats, s, e = reader.logits(examples)                      # reader nhận ĐÚNG đoạn đúng (gold-context)
    summary["reader_gold_context"] = layer_summary(reader.decode(examples, feats, s, e), examples, **boot)
    summary["statuses"] = {}
    for r in rows:
        summary["statuses"][r["status"]] = summary["statuses"].get(r["status"], 0) + 1

    for layer in ("reader_gold_context", "end_to_end"):
        for ref in ("all_refs", "first_ref"):
            m = summary[layer][ref]
            print(f"{layer:20} {ref:9} EM {m['em']:.4f} [{m['em_ci'][0]:.4f}; {m['em_ci'][1]:.4f}]  "
                  f"F1 {m['f1']:.4f} [{m['f1_ci'][0]:.4f}; {m['f1_ci'][1]:.4f}]  (n={m['queries']})")
    print("Retrieval:", {k: round(v, 4) for k, v in summary["retrieval"].items() if k.startswith(("hit", "mrr@10"))})
    print("Thời gian (ms):", {k: {a: round(b, 1) for a, b in v.items()} for k, v in summary["timing"].items()})
    print("Trạng thái:", summary["statuses"])

    write_jsonl(pred_path, rows)
    cfg = {"split": split, "checkpoint": pipe.model_version, "corpus_version": pipe.corpus_version,
           "retrieval_top_k": pipe.retrieval_top_k, "reader_top_k": pipe.reader_top_k,
           "pipeline_config": load_config("pipeline"), "evaluation": ev}
    exp_id = new_experiment_id(f"eval_{split}", cfg)
    path = save_run(exp_id, cfg, summary, extra={"hardware": hardware_info(), "limit": limit,
                                                 "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    (REPORT_DIR / f"summary_{split}.md").write_text(markdown_tables(split, summary), encoding="utf-8", newline="\n")
    print("đã ghi", pred_path.name, path.name, f"summary_{split}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ N câu đầu (chạy thử; không dùng với test)")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    main(a.split, a.checkpoint, a.limit, a.confirm_test, a.overwrite)