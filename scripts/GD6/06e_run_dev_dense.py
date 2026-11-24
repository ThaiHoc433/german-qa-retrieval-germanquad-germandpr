"""GĐ6.7: end-to-end trên DEV với dense thay BM25, giữ nguyên reader và mọi cấu hình khác.
So với BM25 (reports/pipeline_dev_predictions.jsonl của GĐ7) trên cùng câu hỏi, có kiểm định bootstrap ghép cặp."""
import argparse
import time

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json, write_jsonl
from germanqa.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from germanqa.evaluation.report import hardware_info, layer_summary, retrieval_summary, timing_summary
from germanqa.experiment import load_config, new_experiment_id, save_run
from germanqa.pipeline import build_pipeline
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import score_answers


def main(artifact_dir, checkpoint, limit):
    ev = load_config("evaluation")["evaluation"]
    pipe = build_pipeline(checkpoint, retriever="dense", artifact_dir=artifact_dir)
    examples = load_examples(PROCESSED_DIR, "dev")[:limit]
    pipe.answer(examples[0]["question"])                       # khởi động, không tính thời gian
    rows = []
    for i, ex in enumerate(examples, 1):
        r = pipe.answer(ex["question"], trace=True)
        rows.append({"question_id": ex["id"], "question": r.question, "answer": r.answer, "status": r.status,
                     "passage_id": r.passage_id, "answer_start": r.answer_start, "answer_end": r.answer_end,
                     "retrieval_score": r.retrieval_score, "reader_score": r.reader_score,
                     "retrieval_time": r.retrieval_time, "reader_time": r.reader_time, "total_time": r.total_time,
                     "retrieved": r.trace["retrieved"], "candidates": r.trace["candidates"]})
        if i % 300 == 0:
            print(f"{i}/{len(examples)}", flush=True)

    unseen = set(read_json(RESULTS_DIR / "dev_unseen_question_ids.json"))
    bm = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_dev_predictions.jsonl")}
    boot = dict(n_boot=ev["bootstrap_samples"], confidence=ev["confidence"])
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    out = {"timing": timing_summary(rows), "statuses": {}}
    for r in rows:
        out["statuses"][r["status"]] = out["statuses"].get(r["status"], 0) + 1
    for name, keep in (("dev", None), ("dev_unseen", unseen)):
        exs = [e for e in examples if keep is None or e["id"] in keep]
        dense_pred = {r["question_id"]: {"text": r["answer"]} for r in rows}
        bm_pred = {q: {"text": bm[q]["answer"]} for q in (e["id"] for e in exs)}
        sd, sb = score_answers(dense_pred, exs), score_answers(bm_pred, exs)
        sub_rows = [r for r in rows if keep is None or r["question_id"] in keep]
        out[name] = {"dense": layer_summary(dense_pred, exs, **boot), "bm25": layer_summary(bm_pred, exs, **boot),
                     "paired_em_dense_minus_bm25": dict(zip(("diff", "lo", "hi"), paired_bootstrap([x[0] for x in sd], [x[0] for x in sb]))),
                     "paired_f1_dense_minus_bm25": dict(zip(("diff", "lo", "hi"), paired_bootstrap([x[1] for x in sd], [x[1] for x in sb]))),
                     "retrieval_dense": retrieval_summary(sub_rows, qrels, ev["ks"]),
                     "retrieval_bm25": retrieval_summary([bm[r["question_id"]] for r in sub_rows], qrels, ev["ks"])}
        d, b = out[name]["dense"]["first_ref"], out[name]["bm25"]["first_ref"]
        p = out[name]["paired_em_dense_minus_bm25"]
        print(f"{name:11} n={d['queries']:4}  EM dense {d['em']:.4f} [{d['em_ci'][0]:.4f}; {d['em_ci'][1]:.4f}]  "
              f"BM25 {b['em']:.4f} [{b['em_ci'][0]:.4f}; {b['em_ci'][1]:.4f}]  chênh {p['diff']:+.4f} [{p['lo']:+.4f}; {p['hi']:+.4f}]")
        print(f"            F1 dense {d['f1']:.4f}  BM25 {b['f1']:.4f}   Hit@5 dense {out[name]['retrieval_dense']['hit@5']:.3f} BM25 {out[name]['retrieval_bm25']['hit@5']:.3f}")
    print("Thời gian (ms):", {k: {a: round(b, 1) for a, b in v.items()} for k, v in out["timing"].items()})
    write_jsonl(REPORT_DIR / "pipeline_dev_dense_predictions.jsonl", rows)
    cfg = {"split": "dev", "retriever": "dense", "artifact_dir": artifact_dir, "checkpoint": pipe.model_version,
           "corpus_version": pipe.corpus_version, "dense": load_config("dense"), "pipeline": load_config("pipeline")}
    path = save_run(new_experiment_id("eval_dev_dense", cfg), cfg, out,
                    extra={"hardware": hardware_info(), "limit": limit, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print("đã ghi", path.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.artifact_dir, a.checkpoint, a.limit)
