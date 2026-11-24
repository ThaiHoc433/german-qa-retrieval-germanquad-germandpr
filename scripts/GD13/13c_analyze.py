"""GĐ13.5: phân tích lỗi XUẤT PHÁT TỪ LÝ DO chọn hybrid/RRF. Mỗi mục kiểm một giả thuyết đã nêu khi chọn phương pháp.
Chỉ dùng tập chọn (train + dev-unseen) và dev; không mở test. Ghi results/13c_analysis.json và results/13c_cases.json.
Các số trên 2.223 câu có chứa 1.989 câu đã dùng để chọn cấu hình: mô tả, không phải bằng chứng xác nhận."""
import re

import numpy as np

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.evaluation.retrieval_metrics import hit_at_k, mrr_at_k
from germanqa.evaluation.run_cache import as_ranked, read_run
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import score_answers
from germanqa.retrieval.rrf import rrf
from germanqa.retrieval.tokenizer import GERMAN_STOPWORDS, tokenize


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def qtype(question):
    w = re.findall(r"\w+", question.lower())
    w = w[0] if w else ""
    if w.startswith("wie"):
        return "wie*"
    if w.startswith("welch"):
        return "welch*"
    if w in ("warum", "wieso", "weshalb", "wozu", "weswegen"):
        return "warum"
    if w.startswith("wo"):
        return "wo*"
    return w if w in ("was", "wer", "wann", "wem", "wen", "wessen") else "andere"


def rate(x):
    return float(np.mean(x)) if len(x) else None


def gold_rank(run, q, rel):
    for i, p in enumerate(run[q], 1):
        if p in rel:
            return i
    return None


def fuse_all(bm, dn, ids, c, depth, w=1.0, top_k=20):
    return {q: [r["passage_id"] for r in rrf({"bm25": bm[q][:depth], "dense": dn[q][:depth]}, c=c, top_k=top_k, weights={"dense": w})] for q in ids}


def minmax_sum(rows_b, rows_d, depth=100, top_k=20):
    """Đối chứng: cộng điểm đã chuẩn hóa min-max từng nhánh (thay vì cộng 1/(c+hạng))."""
    out = {}
    for q in rows_b:
        tot = {}
        for rows in (rows_b[q][:depth], rows_d[q][:depth]):
            if not rows:
                continue
            s = [x[1] for x in rows]
            lo, hi = min(s), max(s)
            for pid, sc in rows:
                tot[pid] = tot.get(pid, 0.0) + ((sc - lo) / (hi - lo) if hi > lo else 1.0)
        out[q] = [p for p, _ in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]
    return out


def main():
    final = read_json(RESULTS_DIR / "13_selection.json")["final"]
    sweep = read_json(RESULTS_DIR / "rrf_sweep.json")
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    train_ids, dev_ids = list(sel["train"]), list(sel["dev"])
    ids = train_ids + dev_ids
    corpus = {r["passage_id"]: r for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}
    qa = {q["question_id"]: q for q in read_jsonl(PROCESSED_DIR / "qa_train.jsonl") + read_jsonl(PROCESSED_DIR / "qa_dev.jsonl")}
    qrels = {**read_json(PROCESSED_DIR / "qrels_train.json"), **read_json(PROCESSED_DIR / "qrels_dev.json")}
    rows_b = {r["question_id"]: r["hits"] for r in read_run(RESULTS_DIR / "B0_select_top100.jsonl")}
    rows_d = {r["question_id"]: r["hits"] for r in read_run(RESULTS_DIR / "B1_select_top100_384.jsonl")}
    bm, dn = ({q: [h[0] for h in v] for q, v in rows.items()} for rows in (rows_b, rows_d))
    hy = fuse_all(bm, dn, ids, final["rrf_c"], final["depth"], final["w_dense"], final["fusion_top_k"])
    rel = {q: set(qrels[q]) for q in ids}
    H = lambda run, k: {q: bool(hit_at_k(run[q], rel[q], k)) for q in ids}
    hb5, hd5, hh5 = H(bm, 5), H(dn, 5), H(hy, 5)
    out = {"final_config": final, "n_all": len(ids), "n_train": len(train_ids), "n_dev_unseen": len(dev_ids)}

    # ---- A. Hai nhánh bù nhau: RRF cứu được bao nhiêu câu "chỉ BM25 trúng" và làm mất bao nhiêu câu "chỉ dense trúng"?
    def cells(sub):
        res = {}
        for name, f in (("both_hit", lambda q: hb5[q] and hd5[q]), ("dense_only", lambda q: hd5[q] and not hb5[q]),
                        ("bm25_only", lambda q: hb5[q] and not hd5[q]), ("neither", lambda q: not hb5[q] and not hd5[q])):
            m = [q for q in sub if f(q)]
            res[name] = {"n": len(m), "hybrid_hit": sum(hh5[q] for q in m), "hybrid_miss": sum(not hh5[q] for q in m)}
        return res
    out["A_cells"] = {"all": cells(ids), "train": cells(train_ids), "dev_unseen": cells(dev_ids)}
    out["A_hit5"] = {s: {"bm25": rate([hb5[q] for q in sub]), "dense": rate([hd5[q] for q in sub]), "hybrid": rate([hh5[q] for q in sub])}
                     for s, sub in (("all", ids), ("train", train_ids), ("dev_unseen", dev_ids))}
    out["A_oracle_union_hit5"] = rate([hb5[q] or hd5[q] for q in ids])

    # ---- B. Theo mức trùng từ, độ dài câu hỏi, loại câu hỏi
    ptoks = {}
    overlap = {}
    for q in ids:
        pid = qa[q]["passage_id"]
        if pid not in ptoks:
            ptoks[pid] = set(tokenize(corpus[pid]["retrieval_text"], GERMAN_STOPWORDS))
        qt = set(tokenize(qa[q]["question"], GERMAN_STOPWORDS))
        overlap[q] = len(qt & ptoks[pid]) / len(qt) if qt else 0.0
    ob = lambda v: "0–25%" if v < .25 else "25–50%" if v < .5 else "50–75%" if v < .75 else "75–100%"
    nwords = {q: len(qa[q]["question"].split()) for q in ids}
    t1, t2 = np.percentile(list(nwords.values()), [33.3, 66.7])
    lb = lambda n: f"ngắn (≤{int(t1)} từ)" if n <= t1 else f"vừa (≤{int(t2)} từ)" if n <= t2 else f"dài (>{int(t2)} từ)"

    def table(label):
        rows = []
        for g in sorted({label[q] for q in ids}, key=str):
            m = [q for q in ids if label[q] == g]
            rows.append({"group": g, "n": len(m), "bm25_miss@5": 1 - rate([hb5[q] for q in m]), "dense_miss@5": 1 - rate([hd5[q] for q in m]),
                         "hybrid_miss@5": 1 - rate([hh5[q] for q in m]),
                         "hybrid_worse_than_dense": sum(hd5[q] and not hh5[q] for q in m), "hybrid_better_than_dense": sum(hh5[q] and not hd5[q] for q in m)})
        return rows
    out["B_overlap"] = table({q: ob(overlap[q]) for q in ids})
    out["B_qlen"] = table({q: lb(nwords[q]) for q in ids})
    out["B_qtype"] = table({q: qtype(qa[q]["question"]) for q in ids})

    # ---- C. Độ nhạy: lưới 12 cấu hình (train) + mở rộng c nhỏ HẬU NGHIỆM (thăm dò, không dùng để chọn)
    out["C_grid_train"] = [{k: g[k] for k in ("rrf_c", "depth", "hit@5", "hit@20", "recall@20", "mrr@10")} for g in sweep["grid"]]
    ext = []
    for depth in (50, 100):
        for c in (1, 3, 5):
            run = fuse_all(bm, dn, train_ids, c, depth)
            ext.append({"rrf_c": c, "depth": depth, "split": "train", "hit@5": rate([hit_at_k(run[q], rel[q], 5) for q in train_ids]),
                        "hit@20": rate([hit_at_k(run[q], rel[q], 20) for q in train_ids])})
    out["C_extension_posthoc_not_used"] = ext

    # ---- D. Đối chứng: cộng điểm chuẩn hóa thay vì hạng (train)
    ms = minmax_sum({q: rows_b[q] for q in train_ids}, {q: rows_d[q] for q in train_ids})
    out["D_minmax_control_train"] = {"hit@5": rate([hit_at_k(ms[q], rel[q], 5) for q in train_ids]),
                                    "hit@20": rate([hit_at_k(ms[q], rel[q], 20) for q in train_ids]),
                                    "mrr@10": rate([mrr_at_k(ms[q], rel[q], 10) for q in train_ids]),
                                    "rrf_final_hit@5": sweep["final_metrics"]["hit@5"]}

    # ---- E. Phân loại các lần hybrid trượt top-5 (a: đáp án có sẵn trong top-5, b: đúng bài sai đoạn, c: trượt thật)
    ntext = {pid: norm(r["original_text"]) for pid, r in corpus.items()}
    def kinds(run):
        miss = [q for q in ids if not hit_at_k(run[q], rel[q], 5)]
        k = {"n_miss@5": len(miss), "a_answer_in_top5": 0, "b_same_article_not_a": 0, "c_other": 0}
        for q in miss:
            ans = [norm(a["text"]) for a in qa[q]["answers"] if a.get("text")]
            gold_docs = {corpus[p]["document_id"] for p in rel[q]}
            a = any(x and x in ntext[p] for p in run[q][:5] for x in ans)
            b = any(corpus[p]["document_id"] in gold_docs for p in run[q][:5])
            k["a_answer_in_top5"] += a
            k["b_same_article_not_a"] += (b and not a)
            k["c_other"] += (not a and not b)
        return k
    out["E_miss_kinds"] = {"bm25": kinds(bm), "dense384": kinds(dn), "hybrid": kinds(hy)}

    # ---- F. Các câu hybrid kém hơn cả hai nhánh / kém hơn dense: đọc tay một mẫu (chỉ train + dev-unseen)
    worse_both = [q for q in ids if not hh5[q] and hb5[q] and hd5[q]]
    worse_dense = [q for q in ids if not hh5[q] and hd5[q]]
    out["F_worse"] = {"hybrid_miss_but_both_hit": len(worse_both), "hybrid_miss_but_dense_hit": len(worse_dense),
                      "hybrid_hit_but_dense_miss": sum(hh5[q] and not hd5[q] for q in ids)}
    def case(q):
        return {"question_id": q, "split": "dev-unseen" if q in set(dev_ids) else "train", "question": qa[q]["question"],
                "gold_ids": sorted(rel[q]), "gold_rank_bm25": gold_rank(bm, q, rel[q]), "gold_rank_dense": gold_rank(dn, q, rel[q]),
                "gold_rank_hybrid": gold_rank(hy, q, rel[q]), "overlap": round(overlap[q], 2),
                "hybrid_top3_titles": [corpus[p].get("title") for p in hy[q][:3]]}
    out_cases = {"worse_than_dense": [case(q) for q in worse_dense[:25]], "worse_than_both": [case(q) for q in worse_both[:25]]}
    write_json(RESULTS_DIR / "13c_cases.json", out_cases)

    # ---- G. Từ truy hồi sang EM: reader có biến mức tăng Hit@5 thành EM không? (dev 1.176 câu, end-to-end đã chạy)
    h0 = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_dev_hybrid_predictions.jsonl")}
    b1 = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / "pipeline_dev_dense_predictions.jsonl")}
    exs = load_examples(PROCESSED_DIR, "dev")
    qrels_dev = read_json(PROCESSED_DIR / "qrels_dev.json")
    unseen = set(dev_ids)
    g = {}
    for name, keep in (("dev", None), ("dev_unseen", unseen)):
        sub = [e for e in exs if keep is None or e["id"] in keep]
        res = {}
        for label, base in (("H0", h0), ("B1", b1)):
            sc = score_answers({e["id"]: {"text": base[e["id"]]["answer"]} for e in sub}, sub)
            em = {e["id"]: s[0] for e, s in zip(sub, sc)}
            hit = {e["id"]: any(r["passage_id"] in set(qrels_dev[e["id"]]) for r in base[e["id"]]["retrieved"][:5]) for e in sub}
            res[label] = {"em": rate(list(em.values())), "hit@5": rate(list(hit.values())),
                          "em_given_hit": rate([em[q] for q in em if hit[q]]), "em_given_nohit": rate([em[q] for q in em if not hit[q]]),
                          "hit_but_wrong": sum(1 for q in em if hit[q] and not em[q]), "n_hit": sum(hit.values()), "em_by_q": em, "hit_by_q": hit}
        a, b = res["H0"], res["B1"]
        flips = {"h0_right_b1_wrong": sum(1 for q in a["em_by_q"] if a["em_by_q"][q] and not b["em_by_q"][q]),
                 "b1_right_h0_wrong": sum(1 for q in a["em_by_q"] if b["em_by_q"][q] and not a["em_by_q"][q]),
                 "hit_gained": sum(1 for q in a["hit_by_q"] if a["hit_by_q"][q] and not b["hit_by_q"][q]),
                 "hit_lost": sum(1 for q in a["hit_by_q"] if b["hit_by_q"][q] and not a["hit_by_q"][q])}
        for r in (a, b):
            r.pop("em_by_q"); r.pop("hit_by_q")
        g[name] = {"H0": a, "B1": b, "flips": flips}
    out["G_end_to_end"] = g

    # ---- H. Phần online có giống phần offline không (kiểm tính nhất quán, 234 câu dev-unseen)
    same5 = sum([r["passage_id"] for r in h0[q]["retrieved"][:5]] == hy[q][:5] for q in dev_ids)
    out["H_online_vs_offline_top5_identical"] = {"same": same5, "n": len(dev_ids)}
    write_json(RESULTS_DIR / "13c_analysis.json", out)
    print("A cells (all):", out["A_cells"]["all"])
    print("Hit@5:", out["A_hit5"])
    print("F:", out["F_worse"], "| E:", out["E_miss_kinds"])
    print("D control:", out["D_minmax_control_train"])
    print("C ext:", [(e['rrf_c'], e['depth'], round(e['hit@5'], 4)) for e in ext])
    print("G:", {s: {k: v for k, v in d.items()} for s, d in g.items()})
    print("H:", out["H_online_vs_offline_top5_identical"])
    for t in ("B_overlap", "B_qlen", "B_qtype"):
        print(t)
        for r in out[t]:
            print("  ", r["group"], r["n"], "miss@5 bm25/dense/hybrid", round(r["bm25_miss@5"], 3), round(r["dense_miss@5"], 3), round(r["hybrid_miss@5"], 3), "worse/better vs dense", r["hybrid_worse_than_dense"], r["hybrid_better_than_dense"])


if __name__ == "__main__":
    main()
