"""GĐ13.4: (1) xác nhận MỘT lần cấu hình đã chọn trên 234 câu dev-unseen bằng danh sách top-100 đã lưu;
(2) chạy H0 end-to-end (pipeline hybrid + reader giữ nguyên) trên dev, so ghép cặp với B0 (BM25) và B1 (dense 384).
Không mở test. Bước (1) chỉ chạy một lần: results/13_confirmation.json tồn tại thì chỉ đọc lại, không tính lại."""
import argparse
import time

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, read_jsonl, write_json, write_jsonl
from germanqa.evaluation.bootstrap import paired_bootstrap
from germanqa.evaluation.fusion_select import require_prereg
from germanqa.evaluation.report import hardware_info, layer_summary, retrieval_summary, timing_summary
from germanqa.evaluation.run_cache import as_ranked, per_question, read_run
from germanqa.experiment import load_config, new_experiment_id, save_run
from germanqa.pipeline import build_pipeline
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import score_answers
from germanqa.retrieval.rrf import rrf

METRICS = ("hit@1", "hit@5", "hit@20", "recall@20", "mrr@10")


def confirm(pre, final):
    path = RESULTS_DIR / "13_confirmation.json"
    if path.exists():
        print("13_confirmation.json đã có: đọc lại, KHÔNG tính lại (xác nhận chỉ một lần)")
        return read_json(path)
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    ids = list(sel["dev"])
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    bm = as_ranked(read_run(RESULTS_DIR / "B0_select_top100.jsonl"))
    dn = as_ranked(read_run(RESULTS_DIR / "B1_select_top100_384.jsonl"))
    d = final["depth"]
    hy = {q: [r["passage_id"] for r in rrf({"bm25": bm[q][:d], "dense": dn[q][:d]}, c=final["rrf_c"], top_k=final["fusion_top_k"],
                                           weights={"dense": final["w_dense"]})] for q in ids}
    boot = dict(n_boot=pre["bootstrap"]["n_boot"], confidence=pre["bootstrap"]["confidence"], seed=pre["bootstrap"]["seed"])
    runs = {"hybrid": hy, "dense384": {q: dn[q] for q in ids}, "bm25": {q: bm[q] for q in ids}}
    pq = {n: per_question(r, qrels, ids, ks=(1, 5, 20)) for n, r in runs.items()}
    out = {"split": "dev-unseen", "n": len(ids), "final_config": final,
           "metrics": {n: {m: sum(p[m]) / len(ids) for m in METRICS} for n, p in pq.items()}, "paired_hybrid_minus": {}}
    for other in ("dense384", "bm25"):
        out["paired_hybrid_minus"][other] = {}
        for m in METRICS:
            diff, lo, hi = paired_bootstrap(pq["hybrid"][m], pq[other][m], **boot)
            out["paired_hybrid_minus"][other][m] = {"diff": diff, "lo": lo, "hi": hi}
    h5 = out["paired_hybrid_minus"]["dense384"]["hit@5"]
    out["verdict"] = ("H0 hơn B1 về Hit@5 (cận dưới > 0)" if h5["lo"] > 0 else "chưa đủ bằng chứng H0 hơn B1 về Hit@5")
    out["confirmed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(path, out)
    return out


def main(limit, tag):
    pre = require_prereg(RESULTS_DIR / "13_prereg.json", ROOT)
    final = read_json(RESULTS_DIR / "13_selection.json")["final"]
    conf = confirm(pre, final)
    print("XÁC NHẬN dev-unseen (n=%d):" % conf["n"], {n: round(m["hit@5"], 4) for n, m in conf["metrics"].items()},
          "| hybrid - dense hit@5:", {k: round(v, 4) for k, v in conf["paired_hybrid_minus"]["dense384"]["hit@5"].items()}, "|", conf["verdict"])

    ev = load_config("evaluation")["evaluation"]
    pipe = build_pipeline(retriever="hybrid")
    assert pipe.retriever.config()["rrf_c"] == final["rrf_c"] and pipe.retriever.bm25_top_k == final["depth"], "hybrid.yaml lệch lựa chọn"
    examples = load_examples(PROCESSED_DIR, "dev")[:limit]
    pipe.answer(examples[0]["question"])                       # khởi động, không tính thời gian
    rows, parts = [], []
    for i, ex in enumerate(examples, 1):
        r = pipe.answer(ex["question"], trace=True)
        parts.append(dict(pipe.retriever.last_timing))
        rows.append({"question_id": ex["id"], "question": r.question, "answer": r.answer, "status": r.status,
                     "passage_id": r.passage_id, "answer_start": r.answer_start, "answer_end": r.answer_end,
                     "retrieval_score": r.retrieval_score, "reader_score": r.reader_score,
                     "retrieval_time": r.retrieval_time, "reader_time": r.reader_time, "total_time": r.total_time,
                     "retrieved": r.trace["retrieved"], "candidates": r.trace["candidates"]})
        if i % 300 == 0:
            print(f"{i}/{len(examples)}", flush=True)

    unseen = set(read_json(RESULTS_DIR / "dev_unseen_question_ids.json"))
    b0 = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_dev_predictions.jsonl")}
    b1 = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_dev_dense_predictions.jsonl")}
    boot = dict(n_boot=ev["bootstrap_samples"], confidence=ev["confidence"])
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    n = len(parts)
    out = {"timing": timing_summary(rows), "statuses": {}, "timing_parts_ms": {k: round(1000 * sum(p[k] for p in parts) / n, 2) for k in parts[0]}}
    for r in rows:
        out["statuses"][r["status"]] = out["statuses"].get(r["status"], 0) + 1
    for name, keep in (("dev", None), ("dev_unseen", unseen)):
        exs = [e for e in examples if keep is None or e["id"] in keep]
        pred = {r["question_id"]: {"text": r["answer"]} for r in rows}
        sc = score_answers(pred, exs)
        out[name] = {"hybrid": layer_summary(pred, exs, **boot)}
        sub = [r for r in rows if keep is None or r["question_id"] in keep]
        out[name]["retrieval_hybrid"] = retrieval_summary(sub, qrels, ev["ks"])
        for label, base in (("B0_bm25", b0), ("B1_dense384", b1)):
            bp = {e["id"]: {"text": base[e["id"]]["answer"]} for e in exs}
            sb = score_answers(bp, exs)
            out[name][label] = layer_summary(bp, exs, **boot)
            out[name][f"retrieval_{label}"] = retrieval_summary([base[r["question_id"]] for r in sub], qrels, ev["ks"])
            out[name][f"paired_em_hybrid_minus_{label}"] = dict(zip(("diff", "lo", "hi"), paired_bootstrap([x[0] for x in sc], [x[0] for x in sb])))
            out[name][f"paired_f1_hybrid_minus_{label}"] = dict(zip(("diff", "lo", "hi"), paired_bootstrap([x[1] for x in sc], [x[1] for x in sb])))
        h = out[name]["hybrid"]["first_ref"]
        for label in ("B0_bm25", "B1_dense384"):
            b = out[name][label]["first_ref"]
            p = out[name][f"paired_em_hybrid_minus_{label}"]
            print(f"{name:11} n={h['queries']:4} EM H0 {h['em']:.4f} [{h['em_ci'][0]:.4f}; {h['em_ci'][1]:.4f}]  {label} {b['em']:.4f}  "
                  f"chênh {p['diff']:+.4f} [{p['lo']:+.4f}; {p['hi']:+.4f}]  F1 H0 {h['f1']:.4f} vs {b['f1']:.4f}  Hit@5 {out[name]['retrieval_hybrid']['hit@5']:.3f} vs {out[name]['retrieval_'+label]['hit@5']:.3f}")
    print("Thời gian (ms):", {k: {a: round(b, 1) for a, b in v.items()} for k, v in out["timing"].items()}, out["timing_parts_ms"])
    suffix = f"_{tag}" if tag else ""
    write_jsonl(REPORT_DIR / f"pipeline_dev_hybrid_predictions{suffix}.jsonl", rows)
    cfg = {"split": "dev", "retriever": "hybrid", "hybrid": load_config("hybrid"), "dense": load_config("dense"),
           "checkpoint": pipe.model_version, "corpus_version": pipe.corpus_version, "pipeline": load_config("pipeline")}
    path = save_run(new_experiment_id("eval_dev_hybrid", cfg), cfg, out,
                    extra={"hardware": hardware_info(), "limit": limit, "tag": tag, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print("đã ghi", path.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.limit, a.tag)
