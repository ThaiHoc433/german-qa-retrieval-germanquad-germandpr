"""GĐ14.5: chọn cấu hình reranker TRÊN TRAIN-SELECT (1.989 câu) theo quy tắc trong results/14_prereg.json.
Chỉ đọc điểm của câu thuộc train-select (analysis.load_scores lọc theo ID): điểm dev-unseen không vào script này.
Ghi results/14_selection.json một lần (không ghi đè). Không mở test."""
import argparse
import time

from germanqa.config import PROCESSED_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import file_sha1, read_json, read_jsonl, write_json
from germanqa.evaluation.fusion_select import require_prereg
from germanqa.rerank import analysis as A
from germanqa.rerank.exposure import GROUPS, question_group
from germanqa.rerank.pool import hybrid_pool
from germanqa.rerank.select import choose_config, decide

MODELS = ("mmarco", "gbert_dpr", "bge_m3")
OVERFLOWS = ("truncate", "windows")


def main(smoke):
    pre = require_prereg(RESULTS_DIR / "14_prereg.json", ROOT)
    out_path = RESULTS_DIR / "14_selection.json"
    if out_path.exists() and not smoke:
        raise SystemExit(f"{out_path.name} đã tồn tại: không chạy lại (chọn một lần)")
    boot = {k: pre["bootstrap"][k] for k in ("n_boot", "confidence", "seed")}
    margin = pre["adoption_rule"]["margin"]
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    qids = list(sel["train"])
    qrels = read_json(PROCESSED_DIR / "qrels_train.json")
    corpus = {r["passage_id"]: r for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}
    pool = hybrid_pool(qids)

    rows = {m: A.load_scores(RESULTS_DIR / f"14c_scores_{m}.jsonl", qids) for m in MODELS}
    for m in MODELS:
        bad = A.integrity_problems(rows[m], pool, qids)
        if bad:
            raise SystemExit(f"{m}: dữ liệu điểm không toàn vẹn ({len(bad)} lỗi), ví dụ: {bad[:3]}")
    groups = {q: question_group(qrels[q], corpus) for q in qids}
    idx = {g: [i for i, q in enumerate(qids) if groups[q] == g] for g in GROUPS}
    assert {g: len(v) for g, v in idx.items()} == pre["exposure_analysis"]["counts_in_selection"]["train-select"], "số nhóm lệch prereg"

    pq0, m0 = A.metrics(A.h0_run(pool, qids), qrels, qids)
    table, pqs = [], {}
    for m in MODELS:
        for ov in OVERFLOWS:
            pq, mt = A.metrics(A.reranked_run(rows[m], qids, ov), qrels, qids)
            pqs[(m, ov)] = pq
            win_mean = sum(sum(rows[m][q]["windows"]) for q in qids) / len(qids)
            table.append({"model": m, "overflow": ov, "hit@1": mt["hit@1"], "hit@5": mt["hit@5"], "hit@20": mt["hit@20"],
                          "mrr@10": mt["mrr@10"], "recall@20": mt["recall@20"],
                          "windows_per_question": win_mean if ov == "windows" else 20.0})
    assert len(table) == pre["max_configs_tried"]
    chosen = choose_config(table)
    key = (chosen["model"], chosen["overflow"])

    def ci(metric, sub=None):
        return A.gain_ci(pqs[key], pq0, metric, idx=sub, **boot)

    overall5, overall20, never5 = ci("hit@5"), ci("hit@20"), ci("hit@5", idx["never"])
    adopted, why = decide(overall5, overall20, never5, margin)
    # báo cáo hiệu ứng 'đã thấy đoạn' cho TẤT CẢ cấu hình (không dùng để chọn)
    exposure = {}
    for (m, ov), pq in pqs.items():
        gains = {g: [pq["hit@5"][i] - pq0["hit@5"][i] for i in idx[g]] for g in GROUPS}
        exposure[f"{m}/{ov}"] = {
            "gain_hit5": {g: dict(zip(("diff", "lo", "hi"), A.gain_ci(pq, pq0, "hit@5", idx=idx[g], **boot))) for g in GROUPS},
            "positive_minus_never": dict(zip(("diff", "lo", "hi"), A.did_ci(gains["positive"], gains["never"], **boot)))}
    res = {"split": "train-select", "n": len(qids), "prereg_created_at": pre["created_at"], "h0": m0, "table": table,
           "chosen": chosen, "chosen_vs_h0": {"hit@5": dict(zip(("diff", "lo", "hi"), overall5)),
                                              "hit@20": dict(zip(("diff", "lo", "hi"), overall20)),
                                              "hit@5_never_group": dict(zip(("diff", "lo", "hi"), never5))},
           "adopted": adopted, "reason": why, "exposure": exposure, "group_sizes": {g: len(v) for g, v in idx.items()},
           "configs_tried": len(table),
           "unlocked_code_sha1": {p: file_sha1(ROOT / p) for p in ("src/germanqa/rerank/analysis.py", "scripts/GD14/14d_select.py",
                                                                  "scripts/GD14/14c_score_pool.py")},
           "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"H0 hit@5 {m0['hit@5']:.4f} hit@20 {m0['hit@20']:.4f} mrr@10 {m0['mrr@10']:.4f}")
    for r in table:
        print(f"{r['model']:10} {r['overflow']:9} hit@5 {r['hit@5']:.4f} hit@1 {r['hit@1']:.4f} mrr@10 {r['mrr@10']:.4f} win/câu {r['windows_per_question']:.1f}")
    print("CHỌN:", chosen["model"], chosen["overflow"], "| nhận:", adopted, "|", why)
    if not smoke:
        write_json(out_path, res)
        print("đã ghi", out_path.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="chỉ in, không ghi 14_selection.json")
    main(ap.parse_args().smoke)
