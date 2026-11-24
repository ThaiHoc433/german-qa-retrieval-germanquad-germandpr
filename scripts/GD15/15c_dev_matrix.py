"""GĐ15.1-15.5: ma trận B0/B1/H0/H1 trên 234 câu dev-unseen, ba tầng, so ghép cặp có khoảng tin cậy, và phân tích ai cứu ai.
Chỉ dùng DEV (các tệp dự đoán đã lưu ở GĐ7/12/13/14); KHÔNG đọc bất kỳ tệp test nào. Ghi results/15_dev_matrix.json (không ghi đè).

Ba tầng (kế hoạch GĐ15):
  1. Ứng viên : Hit@20 (đoạn đúng có trong 20 đoạn đưa vào bước chọn không).
  2. Chọn     : Hit@5 (đoạn đúng có trong 5 đoạn reader nhận không), MRR@10 theo thứ tự cuối cùng của hệ.
  3. QA       : EM, F1 (first_ref; dev chỉ có một đáp án), tỉ lệ span hợp lệ, tỉ lệ reader chọn đúng đoạn đúng.
Hệ H1: top-5 lấy theo thứ tự xếp lại ('reranked'), còn 20 đoạn là pool của H0."""
import time

import numpy as np

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import score_answers

SOURCES = {"B0": "pipeline_dev_predictions.jsonl", "B1": "pipeline_dev_dense_predictions.jsonl",
           "H0": "pipeline_dev_hybrid_predictions.jsonl", "H1": "pipeline_dev_unseen_H1_predictions.jsonl"}
PAIRS = [("B1", "B0"), ("H0", "B0"), ("H0", "B1"), ("H1", "B0"), ("H1", "B1"), ("H1", "H0")]
METRICS = ("hit@5", "hit@20", "mrr@10", "em", "f1")
BOOT = dict(n_boot=1000, confidence=0.95, seed=42)


def ranked_ids(row, code):
    """Thứ tự đoạn cuối cùng của hệ: H1 theo 'reranked', còn lại theo 'retrieved'."""
    lst = row["reranked"] if code == "H1" else row["retrieved"]
    return [x["passage_id"] for x in lst]


def main():
    out_path = RESULTS_DIR / "15_dev_matrix.json"
    if out_path.exists():
        raise SystemExit(f"{out_path.name} đã tồn tại: không ghi đè")
    ids = list(read_json(RESULTS_DIR / "select_question_ids.json")["dev"])
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    exs = {e["id"]: e for e in load_examples(PROCESSED_DIR, "dev")}
    rows = {c: {r["question_id"]: r for r in read_jsonl(REPORT_DIR / f)} for c, f in SOURCES.items()}
    for c in rows:
        missing = [q for q in ids if q not in rows[c]]
        assert not missing, f"{c}: thiếu {len(missing)} câu"
    em_f1 = {c: dict(zip(ids, score_answers({q: {"text": rows[c][q]["answer"]} for q in ids}, [exs[q] for q in ids]))) for c in rows}
    pq = {c: {m: [] for m in METRICS} for c in rows}
    extra = {c: {"valid_span": [], "picks_gold_given_gold_in_top5": [], "gold_in_top5": []} for c in rows}
    for c in rows:
        for q in ids:
            gold, full, r = set(qrels[q]), ranked_ids(rows[c][q], c), rows[c][q]
            pool = [x["passage_id"] for x in r["retrieved"]]
            top5 = full[:5]
            pq[c]["hit@5"].append(float(bool(gold & set(top5))))
            pq[c]["hit@20"].append(float(bool(gold & set(pool[:20]))))
            rr = next((1.0 / (i + 1) for i, p in enumerate(full[:10]) if p in gold), 0.0)
            pq[c]["mrr@10"].append(rr)
            pq[c]["em"].append(em_f1[c][q][0])
            pq[c]["f1"].append(em_f1[c][q][1])
            extra[c]["valid_span"].append(float(r["status"] in ("ok", "question_truncated") and r["answer_start"] >= 0))
            extra[c]["gold_in_top5"].append(float(bool(gold & set(top5))))
            if gold & set(top5):
                extra[c]["picks_gold_given_gold_in_top5"].append(float(r["passage_id"] in gold))
    res = {"split": "dev-unseen", "n": len(ids), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": SOURCES,
           "note": "Chỉ dev. Không đọc test. EM/F1 là first_ref (dev chỉ có một đáp án).", "systems": {}, "paired": {}, "analysis": {}}
    for c in rows:
        s = {}
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(pq[c][m], **BOOT)
            s[m] = {"mean": mean, "lo": lo, "hi": hi}
        s["layers"] = {"candidates": {"hit@20": s["hit@20"]["mean"]}, "selection": {"hit@5": s["hit@5"]["mean"], "mrr@10": s["mrr@10"]["mean"]},
                       "qa": {"em": s["em"]["mean"], "f1": s["f1"]["mean"], "valid_span_rate": float(np.mean(extra[c]["valid_span"])),
                              "reader_picks_gold_given_gold_in_top5": float(np.mean(extra[c]["picks_gold_given_gold_in_top5"])),
                              "n_gold_in_top5": int(sum(extra[c]["gold_in_top5"]))}}
        res["systems"][c] = s
    for a, b in PAIRS:
        res["paired"][f"{a}-{b}"] = {m: dict(zip(("diff", "lo", "hi"), paired_bootstrap(pq[a][m], pq[b][m], **BOOT))) for m in METRICS}

    # ---- 15.5 ai cứu ai (Hit@5 trên 234 câu) ----
    h5 = {c: np.array(pq[c]["hit@5"], dtype=bool) for c in rows}
    two = lambda x, y: {"both": int((x & y).sum()), "only_first": int((x & ~y).sum()), "only_second": int((~x & y).sum()), "neither": int((~x & ~y).sum())}
    res["analysis"]["bm25_vs_dense_hit5"] = {"first": "B0 (BM25)", "second": "B1 (dense)", **two(h5["B0"], h5["B1"])}
    union = h5["B0"] | h5["B1"]
    res["analysis"]["rrf_effect_hit5"] = {
        "H0_has_gold_in_top5": int(h5["H0"].sum()),
        "lifted_by_RRF (H0 có, B0 và B1 đều không)": int((h5["H0"] & ~h5["B0"] & ~h5["B1"]).sum()),
        "lowered_by_RRF (B0 hoặc B1 có, H0 không)": int((union & ~h5["H0"]).sum()),
        "ceiling_union_B0_B1": int(union.sum())}
    res["analysis"]["reranker_effect_hit5"] = {"rescued (H1 có, H0 không)": int((h5["H1"] & ~h5["H0"]).sum()),
                                               "hurt (H0 có, H1 không)": int((h5["H0"] & ~h5["H1"]).sum()),
                                               "gold_not_in_pool (trần của reranker)": int((1 - np.array(pq["H0"]["hit@20"])).sum())}
    res["analysis"]["em_flips_H1_vs_H0"] = {"H0_right_H1_wrong": int(sum(a > b for a, b in zip(pq["H0"]["em"], pq["H1"]["em"]))),
                                            "H0_wrong_H1_right": int(sum(b > a for a, b in zip(pq["H0"]["em"], pq["H1"]["em"])))}
    write_json(out_path, res)

    print(f"n = {len(ids)} câu dev-unseen")
    print(f"{'':4} {'Hit@20':>7} {'Hit@5':>7} {'MRR@10':>7} {'EM':>7} {'F1':>7} {'span ok':>8} {'chọn đúng đoạn':>15}")
    for c in rows:
        s, l = res["systems"][c], res["systems"][c]["layers"]["qa"]
        print(f"{c:4} {s['hit@20']['mean']:7.4f} {s['hit@5']['mean']:7.4f} {s['mrr@10']['mean']:7.4f} {s['em']['mean']:7.4f} {s['f1']['mean']:7.4f} "
              f"{l['valid_span_rate']:8.3f} {l['reader_picks_gold_given_gold_in_top5']:15.3f}")
    for k, v in res["paired"].items():
        print(f"{k:7} " + "  ".join(f"{m} {v[m]['diff']:+.3f}[{v[m]['lo']:+.3f};{v[m]['hi']:+.3f}]" for m in ("hit@5", "mrr@10", "em")))
    print(res["analysis"])


if __name__ == "__main__":
    main()
