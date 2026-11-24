"""GĐ14 bổ sung: chấm ml6team/cross-encoder-mmarco-german-distilbert-base như thí nghiệm THĂM DÒ, NGOÀI prereg, SAU KHI đã thấy kết quả của 3 model.
Vì sao ngoài prereg: prereg khóa đúng 6 cấu hình; thêm model sau khi thấy kết quả vào phần chọn sẽ thiên lệch. Kết quả ở đây KHÔNG đổi H1; nếu nó ngang hoặc hơn
bge_m3 thì chỉ báo để mở lại prereg có ghi chú.
Quy tắc truy cập: chỉ chấm và chỉ đọc 1.989 câu train-select. Không chạm 234 câu dev-unseen. Không mở test.
Dùng lại CrossEncoderReranker (không sửa file đã khóa), hybrid_pool và các hàm trong analysis.py. Ghi results/14k_ml6team_scores.jsonl (+ .meta.json)
và results/14k_ml6team_exploratory.json (không ghi đè)."""
import importlib.util
import json
import statistics
import time

import torch

from germanqa.config import PROCESSED_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.rerank import analysis as A
from germanqa.rerank.cross_encoder import CrossEncoderReranker
from germanqa.rerank.exposure import GROUPS, question_group
from germanqa.rerank.hub import local_snapshot
from germanqa.rerank.pool import hybrid_pool

REPO = "ml6team/cross-encoder-mmarco-german-distilbert-base"
BOOT = dict(n_boot=1000, confidence=0.95, seed=42)          # cùng thiết lập với prereg GĐ14


def hand_pairs():
    spec = importlib.util.spec_from_file_location("probe14a", ROOT / "scripts" / "GD14" / "14a_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HAND


def main():
    out_path = RESULTS_DIR / "14k_ml6team_exploratory.json"
    if out_path.exists():
        raise SystemExit(f"{out_path.name} đã tồn tại: không ghi đè")
    snap = local_snapshot(REPO)
    rr = CrossEncoderReranker(snap, max_length=512, batch_size=16, fp16=True, relevant_label=None, overflow="windows", stride=128)
    fp = rr.fingerprint()
    # 1) kiểm hướng điểm bằng 5 cặp viết tay (đoạn đúng phải cao hơn đoạn lạc đề)
    wins = sum(int(s[0] > s[1]) for s in (rr.score_texts(q, [g, b]) for q, g, b in hand_pairs()))
    if wins != 5:
        raise SystemExit(f"điểm không đi đúng hướng trên cặp biết trước ({wins}/5): dừng, không chấm dữ liệu thật")
    # 2) chấm 1.989 câu train-select
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    qids = list(sel["train"])
    qa = {r["question_id"]: r["question"] for r in read_jsonl(PROCESSED_DIR / "qa_train.jsonl")}
    corpus = {r["passage_id"]: r for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}
    pool = hybrid_pool(qids)
    scores_path = RESULTS_DIR / "14k_ml6team_scores.jsonl"
    meta_path = scores_path.with_suffix(".meta.json")
    if scores_path.exists():
        raise SystemExit(f"{scores_path.name} đã tồn tại: xóa tay nếu muốn chấm lại")
    write_json(meta_path, {"model": "ml6team", "repo": REPO, "commit": snap.replace("\\", "/").split("/")[-1], "fingerprint": fp,
                           "exploratory_after_results": True, "split": "train-select", "pool_size": 20})
    times, wins_total = [], 0
    with open(scores_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(qids, 1):
            pids = pool[q]
            d = rr.score_detail(qa[q], [corpus[p]["original_text"] for p in pids])
            times.append(rr.last_stats["seconds"])
            wins_total += rr.last_stats["windows"]
            f.write(json.dumps({"question_id": q, "passage_ids": pids, "score": d["score"], "first": d["first"], "windows": d["windows"]}, ensure_ascii=False) + "\n")
            if i % 400 == 0:
                print(f"{i}/{len(qids)} | trung vị {1000 * statistics.median(times):.0f} ms/câu", flush=True)
    stats = {"questions": len(qids), "median_ms_per_question": round(1000 * statistics.median(times), 1),
             "mean_windows_per_question": round(wins_total / len(qids), 2), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    # 3) phân tích: cùng hàm với 14d
    qrels = read_json(PROCESSED_DIR / "qrels_train.json")
    rows = A.load_scores(scores_path, qids)
    bad = A.integrity_problems(rows, pool, qids)
    if bad:
        raise SystemExit(f"dữ liệu điểm không toàn vẹn: {bad[:3]}")
    pq0, m0 = A.metrics(A.h0_run(pool, qids), qrels, qids)
    bge_rows = A.load_scores(RESULTS_DIR / "14c_scores_bge_m3.jsonl", qids)
    pq_bge, m_bge = A.metrics(A.reranked_run(bge_rows, qids, "windows"), qrels, qids)
    groups = {q: question_group(qrels[q], corpus) for q in qids}
    idx = {g: [i for i, q in enumerate(qids) if groups[q] == g] for g in GROUPS}
    res = {"exploratory_after_results": True, "not_part_of_prereg": True, "split": "train-select", "n": len(qids), "model": REPO,
           "commit": snap.replace("\\", "/").split("/")[-1], "stats": stats, "h0": m0, "bge_m3_windows": m_bge, "variants": {}}
    for ov in ("truncate", "windows"):
        pq, mt = A.metrics(A.reranked_run(rows, qids, ov), qrels, qids)
        gains = {g: dict(zip(("diff", "lo", "hi"), A.gain_ci(pq, pq0, "hit@5", idx=idx[g], **BOOT))) for g in GROUPS}
        res["variants"][ov] = {"metrics": mt,
                               "vs_h0_hit5": dict(zip(("diff", "lo", "hi"), A.gain_ci(pq, pq0, "hit@5", **BOOT))),
                               "vs_bge_m3_hit5": dict(zip(("diff", "lo", "hi"), A.gain_ci(pq, pq_bge, "hit@5", **BOOT))),
                               "hit5_gain_vs_h0_by_group": gains}
    res["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(out_path, res)
    print(json.dumps({k: res[k] for k in ("stats", "h0", "bge_m3_windows")}, ensure_ascii=False)[:600])
    for ov, v in res["variants"].items():
        print(ov, "Hit@5 %.4f | so với H0 %+.4f [%+.4f; %+.4f] | so với bge_m3 %+.4f [%+.4f; %+.4f]" % (
            v["metrics"]["hit@5"], *(v["vs_h0_hit5"][k] for k in ("diff", "lo", "hi")), *(v["vs_bge_m3_hit5"][k] for k in ("diff", "lo", "hi"))))
        print("   theo nhóm:", {g: round(x["diff"], 4) for g, x in v["hit5_gain_vs_h0_by_group"].items()})


if __name__ == "__main__":
    main()
