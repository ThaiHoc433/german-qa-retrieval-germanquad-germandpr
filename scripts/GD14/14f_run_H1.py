"""GĐ14.6: chạy H0 và H1 đầu-cuối (cùng reader, cùng D5 = 80, cùng pool top-20) trên 234 câu dev-unseen, trong MỘT tiến trình.
Mục đích: (1) EM/F1 kèm khoảng tin cậy ghép cặp (chỉ báo cáo; chỉ số quyết định vẫn là Hit@5 ở 14e);
(2) chứng minh reranker=None cho kết quả GIỐNG HỆT H0 đã lưu (hồi quy); (3) đối chiếu thứ hạng đầu-cuối với điểm đã lưu đệm;
(4) đo thời gian từng tầng (trung vị) và VRAM đỉnh của cả tiến trình. Không mở test.
Ghi results/14f_H1_dev_unseen.json (không ghi đè) và reports/pipeline_dev_unseen_{H0,H1}_predictions.jsonl."""
import statistics
import time

import numpy as np
import torch

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, read_jsonl, write_json, write_jsonl
from germanqa.evaluation.bootstrap import paired_bootstrap
from germanqa.evaluation.fusion_select import require_prereg
from germanqa.evaluation.report import hardware_info, layer_summary
from germanqa.experiment import load_config
from germanqa.pipeline import build_pipeline
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import score_answers
from germanqa.rerank import analysis as A
from germanqa.rerank.pool import hybrid_pool


def run_pass(pipe, examples, reranker):
    pipe.reranker = reranker
    rows = []
    for ex in examples:
        r = pipe.answer(ex["question"], trace=True)
        rows.append({"question_id": ex["id"], "question": r.question, "answer": r.answer, "status": r.status,
                     "passage_id": r.passage_id, "answer_start": r.answer_start, "answer_end": r.answer_end,
                     "retrieval_score": r.retrieval_score, "rerank_score": r.rerank_score, "reader_score": r.reader_score,
                     "retrieval_time": r.retrieval_time, "rerank_time": r.rerank_time, "reader_time": r.reader_time,
                     "total_time": r.total_time, "retrieved": r.trace["retrieved"], "candidates": r.trace["candidates"],
                     "reranked": r.trace.get("reranked"), "rerank_stats": r.trace.get("rerank_stats")})
    return rows


def med(rows, key):
    return round(1000 * statistics.median(r[key] for r in rows), 1)


def main():
    out_path = RESULTS_DIR / "14f_H1_dev_unseen.json"
    if out_path.exists():
        raise SystemExit(f"{out_path.name} đã tồn tại: không ghi đè")
    pre = require_prereg(RESULTS_DIR / "14_prereg.json", ROOT)
    conf = read_json(RESULTS_DIR / "14_confirmation.json")
    ev = load_config("evaluation")["evaluation"]
    boot = dict(n_boot=ev["bootstrap_samples"], confidence=ev["confidence"])
    unseen = list(read_json(RESULTS_DIR / "select_question_ids.json")["dev"])
    keep = set(unseen)
    examples = [e for e in load_examples(PROCESSED_DIR, "dev") if e["id"] in keep]
    assert len(examples) == 234

    pipe = build_pipeline(retriever="hybrid", reranker=True)
    rr = pipe.reranker
    assert conf["config"]["model"] == load_config("rerank")["rerank"]["model"] == "bge_m3"
    for e in examples[:3]:                                       # khởi động cả hai đường, không tính thời gian
        run_pass(pipe, [e], None)
        run_pass(pipe, [e], rr)
    res = {"n": len(examples), "hardware": hardware_info()}
    passes = {}
    for name, reranker in (("H0", None), ("H1", rr)):
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        passes[name] = run_pass(pipe, examples, reranker)
        res[f"{name}_wall_s"] = round(time.perf_counter() - t, 1)
        res[f"{name}_peak_vram_mb"] = {"allocated": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                                       "reserved": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
    # lượt đo thời gian thứ hai (100 câu đầu, xen kẽ) để xem độ ổn định
    again = {"H0": run_pass(pipe, examples[:100], None), "H1": run_pass(pipe, examples[:100], rr)}

    # (2) hồi quy: reranker=None phải giống H0 đã lưu từ GĐ13
    saved = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_dev_hybrid_predictions.jsonl")}
    diff = {"answer": 0, "passage_id": 0, "reader_score": 0.0, "retrieved_ids": 0}
    for r in passes["H0"]:
        s = saved[r["question_id"]]
        diff["answer"] += int(r["answer"] != s["answer"])
        diff["passage_id"] += int(r["passage_id"] != s["passage_id"])
        diff["reader_score"] = max(diff["reader_score"], abs(r["reader_score"] - s["reader_score"]))
        diff["retrieved_ids"] += int([h["passage_id"] for h in r["retrieved"]] != [h["passage_id"] for h in s["retrieved"]])
    res["regression_H0_vs_saved_GD13"] = diff

    # (3) thứ hạng đầu-cuối so với điểm đã lưu đệm (cùng model, cùng cấu hình)
    sc = A.load_scores(RESULTS_DIR / "14c_scores_bge_m3.jsonl", unseen)
    off = A.reranked_run(sc, unseen, "windows")
    top5_diff = sum(set(off[r["question_id"]][:5]) != {x["passage_id"] for x in r["reranked"][:5]} for r in passes["H1"])
    order_diff = sum(off[r["question_id"]] != [x["passage_id"] for x in r["reranked"]] for r in passes["H1"])
    maxdiff = max(abs(a - b) for r in passes["H1"] for a, b in zip([x["rerank_score"] for x in sorted(r["reranked"], key=lambda x: x["rank_before"])],
                                                                    sc[r["question_id"]]["score"]))
    res["end_to_end_vs_cached_scores"] = {"top5_set_differs": top5_diff, "full_order_differs": order_diff, "max_abs_score_diff": maxdiff}

    # (1) chất lượng
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    q = {}
    for name in ("H0", "H1"):
        pred = {r["question_id"]: {"text": r["answer"]} for r in passes[name]}
        q[name] = (pred, score_answers(pred, examples))
        res[name] = layer_summary(pred, examples, **boot)
        top5 = [{x["passage_id"] for x in (r["reranked"] if name == "H1" else r["retrieved"])[:5]} for r in passes[name]]
        res[name]["evidence_in_top5"] = float(np.mean([bool(set(qrels[r["question_id"]]) & t) for r, t in zip(passes[name], top5)]))
    res["paired_em_H1_minus_H0"] = dict(zip(("diff", "lo", "hi"), paired_bootstrap([x[0] for x in q["H1"][1]], [x[0] for x in q["H0"][1]])))
    res["paired_f1_H1_minus_H0"] = dict(zip(("diff", "lo", "hi"), paired_bootstrap([x[1] for x in q["H1"][1]], [x[1] for x in q["H0"][1]])))
    em0, em1 = [x[0] for x in q["H0"][1]], [x[0] for x in q["H1"][1]]
    res["em_flips"] = {"H0_right_H1_wrong": sum(a > b for a, b in zip(em0, em1)), "H0_wrong_H1_right": sum(b > a for a, b in zip(em0, em1))}

    # (4) thời gian từng tầng (ms, trung vị) cho cả hai lượt đo
    def stage(rows):
        return {"retrieval": med(rows, "retrieval_time"), "rerank": med(rows, "rerank_time"), "reader": med(rows, "reader_time"),
                "total": med(rows, "total_time")}
    res["timing_median_ms"] = {"H0": stage(passes["H0"]), "H1": stage(passes["H1"]),
                               "H0_second_pass_100q": stage(again["H0"]), "H1_second_pass_100q": stage(again["H1"])}
    res["rerank_pairs_windows_mean"] = {"pairs": 20.0, "windows": float(np.mean([r["rerank_stats"]["windows"] for r in passes["H1"]]))}
    for name in ("H0", "H1"):
        write_jsonl(REPORT_DIR / f"pipeline_dev_unseen_{name}_predictions.jsonl", passes[name])
    write_json(out_path, res)
    h0, h1 = res["H0"]["first_ref"], res["H1"]["first_ref"]
    print("hồi quy H0 so với đã lưu:", diff)
    print("đầu-cuối so với điểm đệm:", res["end_to_end_vs_cached_scores"])
    print(f"EM H0 {h0['em']:.4f} -> H1 {h1['em']:.4f} | chênh {res['paired_em_H1_minus_H0']['diff']:+.4f} "
          f"[{res['paired_em_H1_minus_H0']['lo']:+.4f}; {res['paired_em_H1_minus_H0']['hi']:+.4f}] | F1 {h0['f1']:.4f} -> {h1['f1']:.4f} "
          f"chênh {res['paired_f1_H1_minus_H0']['diff']:+.4f} [{res['paired_f1_H1_minus_H0']['lo']:+.4f}; {res['paired_f1_H1_minus_H0']['hi']:+.4f}]")
    print("evidence-in-top5:", res["H0"]["evidence_in_top5"], "->", res["H1"]["evidence_in_top5"], "| lật EM:", res["em_flips"])
    print("thời gian (ms):", res["timing_median_ms"])
    print("VRAM đỉnh:", res["H0_peak_vram_mb"], res["H1_peak_vram_mb"], "| cặp/cửa sổ:", res["rerank_pairs_windows_mean"])


if __name__ == "__main__":
    main()
