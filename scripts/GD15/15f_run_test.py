"""GĐ15.6b: chạy TEST MỘT LẦN cho ma trận. B1, H0, H1 chạy trực tiếp; B0 DÙNG LẠI kết quả đã chạy ngày 2026-11-04 (cấu hình khớp manifest B0 hiện tại, đã kiểm ở 15e).
Điều kiện: có results/15_freeze.json còn khớp từng tệp, preflight sạch, và cờ --confirm-test. Từ chối nếu tệp dự đoán hoặc kết quả đã có (test chỉ một lần).
Guard của sổ phơi bày ghi một dòng metric_run cho B1, H0, H1 NGAY LÚC GỌI (kể cả khi script sau đó hỏng), nên --dry-run bắt buộc phải qua trước.

--dry-run: chạy ĐÚNG đường mã này trên 30 câu DEV-unseen (B0 lấy từ dự đoán dev), KHÔNG mở test, KHÔNG ghi gì. Dùng để bắt lỗi trước khi mở test thật.

Phán định ba giả thuyết trong results/15_hypotheses.json bằng hàm đã khóa gd15_hypotheses.py; EM chính là first_ref theo phụ lục 15_hypotheses_addendum.json,
all_refs báo thêm. Dự đoán test của B0 chỉ được đọc để tính chỉ số/so ghép cặp, không để xem lỗi hay thiết kế."""
import argparse
import time

import numpy as np

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, read_jsonl, write_json, write_jsonl
from germanqa.evaluation import gd15_hypotheses as H
from germanqa.evaluation.bootstrap import paired_bootstrap
from germanqa.evaluation.gd15_freeze import preflight_problems, verify_freeze
from germanqa.evaluation.report import first_reference_only, hardware_info, layer_summary, retrieval_summary, timing_summary
from germanqa.experiment import load_config
from germanqa.experiment_rules import guard_or_exit
from germanqa.pipeline import QAPipeline, build_pipeline
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import score_answers

CODES = ("B0", "B1", "H0", "H1")
PAIRS = [("B1", "B0"), ("H0", "B0"), ("H0", "B1"), ("H1", "B0"), ("H1", "B1"), ("H1", "H0")]
METRICS = ("hit@5", "hit@20", "mrr@10", "em_first", "f1_first", "em_all", "f1_all")
BOOT = dict(n_boot=1000, confidence=0.95, seed=42)


def run_system(pipe, examples, tag):
    pipe.answer(examples[0]["question"])                                 # khởi động, không tính
    rows, t0 = [], time.perf_counter()
    for i, ex in enumerate(examples, 1):
        r = pipe.answer(ex["question"], trace=True)                      # chỉ truyền câu hỏi; đáp án chuẩn chỉ đọc ở bước chấm
        rows.append({"question_id": ex["id"], "question": r.question, "answer": r.answer, "status": r.status, "passage_id": r.passage_id,
                     "answer_start": r.answer_start, "answer_end": r.answer_end, "retrieval_score": r.retrieval_score, "rerank_score": r.rerank_score,
                     "reader_score": r.reader_score, "retrieval_time": r.retrieval_time, "rerank_time": r.rerank_time, "reader_time": r.reader_time,
                     "total_time": r.total_time, "retrieved": r.trace["retrieved"], "candidates": r.trace["candidates"], "reranked": r.trace.get("reranked")})
        if i % 400 == 0:
            print(f"[{tag}] {i}/{len(examples)} | {time.perf_counter() - t0:.0f}s", flush=True)
    return rows


def ranked(row, code):
    return [x["passage_id"] for x in (row["reranked"] if code == "H1" and row.get("reranked") else row["retrieved"])]


def vectors(rows_by_code, examples, qrels):
    ids = [e["id"] for e in examples]
    first = first_reference_only(examples)
    out = {}
    for c, rows in rows_by_code.items():
        by = {r["question_id"]: r for r in rows}
        pred = {q: {"text": by[q]["answer"]} for q in ids}
        sf, sa = score_answers(pred, first), score_answers(pred, examples)
        v = {m: [] for m in METRICS}
        for i, q in enumerate(ids):
            gold, full, pool = set(qrels[q]), ranked(by[q], c), [x["passage_id"] for x in by[q]["retrieved"]]
            v["hit@5"].append(float(bool(gold & set(full[:5]))))
            v["hit@20"].append(float(bool(gold & set(pool[:20]))))
            v["mrr@10"].append(next((1.0 / (k + 1) for k, p in enumerate(full[:10]) if p in gold), 0.0))
            v["em_first"].append(sf[i][0]), v["f1_first"].append(sf[i][1]), v["em_all"].append(sa[i][0]), v["f1_all"].append(sa[i][1])
        out[c] = v
    return out


def compute(rows_by_code, examples, qrels, hyp):
    ids = [e["id"] for e in examples]
    vec = vectors(rows_by_code, examples, qrels)
    res = {"n": len(ids), "systems": {}, "paired": {}}
    for c, rows in rows_by_code.items():
        by = {r["question_id"]: r for r in rows}
        rows_ordered = [by[q] for q in ids]
        view = [{**r, "retrieved": [{"passage_id": p} for p in ranked(r, c)]} for r in rows_ordered]          # H1: thứ tự cuối cùng
        stat = {}
        for r in rows_ordered:
            stat[r["status"]] = stat.get(r["status"], 0) + 1
        res["systems"][c] = {"retrieval": retrieval_summary(view, qrels, [1, 5, 10, 20]),
                             "end_to_end": layer_summary({q: {"text": by[q]["answer"]} for q in ids}, examples, **{k: BOOT[k] for k in ("n_boot", "confidence")}),
                             "statuses": stat, "timing": timing_summary(rows_ordered),
                             "valid_span_rate": float(np.mean([r["status"] in ("ok", "question_truncated") and r["answer_start"] >= 0 for r in rows_ordered]))}
    for a, b in PAIRS:
        res["paired"][f"{a}-{b}"] = {m: dict(zip(("diff", "lo", "hi"), paired_bootstrap(vec[a][m], vec[b][m], **BOOT))) for m in METRICS}
    em = lambda mode: {c: res["systems"][c]["end_to_end"][mode]["em"] for c in ("B0", "B1", "H0")}
    full, unseen = hyp["reference_dev_em"]["full_dev_1176"], hyp["reference_dev_em"]["dev_unseen_234"]
    p = res["paired"]
    res["hypotheses"] = {
        "primary_reference_mode": "first_ref",
        "HA": {"primary_first_ref": H.judge_ha(em("first_ref"), unseen, full), "secondary_all_refs": H.judge_ha(em("all_refs"), unseen, full),
               "test_em_first_ref": em("first_ref"), "test_em_all_refs": em("all_refs")},
        "HB": {"primary_first_ref": H.judge_hb(p["H0-B1"]["em_first"]["diff"]), "secondary_all_refs": H.judge_hb(p["H0-B1"]["em_all"]["diff"]),
               "em_diff_h0_minus_b1": {"first_ref": p["H0-B1"]["em_first"], "all_refs": p["H0-B1"]["em_all"]}},
        "HC": {"primary_first_ref": H.judge_hc(tuple(p["H1-H0"]["hit@5"][k] for k in ("diff", "lo", "hi")), tuple(p["H1-H0"]["em_first"][k] for k in ("diff", "lo", "hi"))),
               "secondary_all_refs": H.judge_hc(tuple(p["H1-H0"]["hit@5"][k] for k in ("diff", "lo", "hi")), tuple(p["H1-H0"]["em_all"][k] for k in ("diff", "lo", "hi")))}}
    return res


def show(res, label):
    print(f"\n=== {label} (n = {res['n']}) ===")
    print(f"{'':4} {'Hit@20':>7} {'Hit@5':>7} {'MRR@10':>7} {'EM1':>7} {'F1(1)':>7} {'EM*':>7} {'F1*':>7} {'ms/câu':>8}")
    for c, s in res["systems"].items():
        r, e = s["retrieval"], s["end_to_end"]
        print(f"{c:4} {r['hit@20']:7.4f} {r['hit@5']:7.4f} {r['mrr@10']:7.4f} {e['first_ref']['em']:7.4f} {e['first_ref']['f1']:7.4f} {e['all_refs']['em']:7.4f} {e['all_refs']['f1']:7.4f} "
              f"{1000 * s['timing']['total_time']['p50_ms'] / 1000 if 'p50_ms' in s['timing']['total_time'] else float('nan'):8.1f}")
    for k, v in res["paired"].items():
        print(f"{k:7} Hit@5 {v['hit@5']['diff']:+.3f}[{v['hit@5']['lo']:+.3f};{v['hit@5']['hi']:+.3f}]  EM(first) {v['em_first']['diff']:+.3f}[{v['em_first']['lo']:+.3f};{v['em_first']['hi']:+.3f}]  "
              f"EM(all) {v['em_all']['diff']:+.3f}[{v['em_all']['lo']:+.3f};{v['em_all']['hi']:+.3f}]")
    h = res["hypotheses"]
    print("HA:", h["HA"]["primary_first_ref"], "| all_refs:", h["HA"]["secondary_all_refs"]["verdict"])
    print("HB:", h["HB"]["primary_first_ref"], "| all_refs:", h["HB"]["secondary_all_refs"]["verdict"])
    print("HC:", h["HC"]["primary_first_ref"], "| all_refs:", h["HC"]["secondary_all_refs"]["verdict"])


def main(confirm_test, dry_run):
    hyp = read_json(RESULTS_DIR / "15_hypotheses.json")
    split = "dev" if dry_run else "test"
    out_json = RESULTS_DIR / "final_test_matrix.json"
    pred_paths = {c: REPORT_DIR / f"pipeline_test_{c}_predictions.jsonl" for c in ("B1", "H0", "H1")}
    if not dry_run:
        fz_path = RESULTS_DIR / "15_freeze.json"
        if not fz_path.exists():
            raise SystemExit("chưa có results/15_freeze.json: chạy 15e_freeze.py trước")
        bad = verify_freeze(read_json(fz_path), ROOT)
        if bad:
            raise SystemExit("cấu hình đã đổi sau khi đóng băng:\n  - " + "\n  - ".join(bad))
        pf = preflight_problems(ROOT)
        if pf:
            raise SystemExit("preflight không sạch:\n  - " + "\n  - ".join(pf))
        existing = [p.name for p in list(pred_paths.values()) + [out_json] if p.exists()]
        if existing:
            raise SystemExit(f"test đã chạy rồi ({existing}). Chạy lại là tinh chỉnh theo test.")
        guard_or_exit("15f_run_test.py", "test MỘT lần cho B1, H0, H1 với cấu hình đã đóng băng (B0 dùng lại số đã chạy)", confirm_test, kind="metric_run",
                      files=["qa_test.jsonl", "qrels_test.json"] + [p.name for p in pred_paths.values()], codes=["B1", "H0", "H1"])
    examples = load_examples(PROCESSED_DIR, split)
    qrels = read_json(PROCESSED_DIR / f"qrels_{split}.json")
    if dry_run:
        unseen = set(read_json(RESULTS_DIR / "select_question_ids.json")["dev"])
        examples = [e for e in examples if e["id"] in unseen][:30]
        b0 = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_dev_predictions.jsonl")}
    else:
        b0 = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_test_predictions.jsonl")}
    ids = [e["id"] for e in examples]
    assert all(q in b0 for q in ids), "thiếu dự đoán B0 cho một số câu"
    cfg = load_config("pipeline")["pipeline"]
    h1 = build_pipeline(retriever="hybrid", reranker=True)
    mk = lambda retriever, rr: QAPipeline(retriever, h1.reader, h1.passages, h1.model_version, h1.corpus_version, "final", cfg["retrieval_top_k"], cfg["reader_top_k"], reranker=rr)
    systems = {"B1": mk(h1.retriever.dense, None), "H0": mk(h1.retriever, None), "H1": h1}
    rows_by_code = {"B0": [b0[q] for q in ids]}
    for c, pipe in systems.items():
        t = time.perf_counter()
        rows_by_code[c] = run_system(pipe, examples, c)
        if not dry_run:
            write_jsonl(pred_paths[c], rows_by_code[c])                  # ghi ngay từng hệ để lỗi sau đó không làm mất kết quả đã tính
        print(f"[{c}] xong {len(examples)} câu trong {time.perf_counter() - t:.0f}s", flush=True)
    res = compute(rows_by_code, examples, qrels, hyp)
    show(res, f"{'DRY-RUN trên dev-unseen' if dry_run else 'TEST'}")
    if dry_run:
        print("\nDRY-RUN xong: không ghi gì, không mở test.")
        return
    res.update({"split": "test", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "hardware": hardware_info(),
                "b0": "dùng lại kết quả test ngày 2026-11-04 (reports/pipeline_test_predictions.jsonl, results/eval_test_20261103T170542Z_93098dcb.json)",
                "freeze": "results/15_freeze.json", "hypotheses_source": "results/15_hypotheses.json + 15_hypotheses_addendum.json"})
    write_json(out_json, res)
    print("đã ghi", out_json.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="chạy đúng đường mã này trên 30 câu dev-unseen, không mở test, không ghi gì")
    a = ap.parse_args()
    main(a.confirm_test, a.dry_run)
