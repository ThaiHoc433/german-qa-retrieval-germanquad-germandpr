"""GĐ6.6: đánh giá retrieval trên DEV, dense (một hoặc nhiều biến thể) so với BM25.
Báo cáo trên hai tập: dev đầy đủ (encoder đã thấy ~80% câu hỏi này lúc huấn luyện -> số bị thổi phồng)
và dev_unseen (câu hỏi encoder CHƯA thấy; n nhỏ nhưng là tập duy nhất không rò rỉ)."""
import argparse
import time

import numpy as np

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, read_jsonl, write_json, write_jsonl
from germanqa.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from germanqa.evaluation.retrieval_metrics import hit_at_k, mrr_at_k, recall_at_k
from germanqa.experiment import load_config
from germanqa.retrieval.bm25 import BM25Retriever
from germanqa.retrieval.dense import DenseRetriever, DPREncoder

KS = (1, 5, 10, 20, 50)


def per_question(runs, qrels, qids):
    """{'hit@5': [0/1 theo từng câu], 'mrr@10': [...], ...} theo thứ tự qids."""
    out = {}
    for k in KS:
        out[f"hit@{k}"] = [hit_at_k(runs[q], set(qrels[q]), k) for q in qids]
        out[f"recall@{k}"] = [recall_at_k(runs[q], set(qrels[q]), k) for q in qids]
    for k in (10, 20, 50):
        out[f"mrr@{k}"] = [mrr_at_k(runs[q], set(qrels[q]), k) for q in qids]
    return out


def summarize(scores):
    return {m: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v))) for m, v in scores.items()}


def main(names, detail, limit):
    cfg = load_config("dense")["dense"]
    pcfg = load_config("pipeline")["bm25"]
    corpus = read_jsonl(PROCESSED_DIR / "corpus.jsonl")
    qs = read_jsonl(PROCESSED_DIR / "qa_dev.jsonl")[:limit]
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    unseen = set(read_json(RESULTS_DIR / "dev_unseen_question_ids.json"))
    qids = [q["question_id"] for q in qs]
    subsets = {"dev": qids, "dev_unseen": [q for q in qids if q in unseen]}
    questions = [q["question"] for q in qs]

    bm25 = BM25Retriever(corpus, k1=pcfg["k1"], b=pcfg["b"], use_stopwords=pcfg["use_stopwords"])
    bm_run = {q["question_id"]: [h["passage_id"] for h in bm25.retrieve(q["question"], max(KS))] for q in qs}
    bm_scores = {s: per_question(bm_run, qrels, ids) for s, ids in subsets.items()}

    enc = DPREncoder(cfg["ctx_encoder"], cfg["question_encoder"], fp16=cfg["fp16"])
    result = {"bm25": {s: summarize(v) for s, v in bm_scores.items()}, "dense": {}, "paired_dense_minus_bm25": {}}
    all_dense = {}
    for name in names:
        r = DenseRetriever.from_dir(ROOT / "artifacts" / "dense" / name, enc, PROCESSED_DIR / "corpus.jsonl",
                                    cfg["question_max_length"])
        hits = r.retrieve_many(questions, max(KS))
        run = {qid: [h["passage_id"] for h in hs] for qid, hs in zip(qids, hits)}
        all_dense[name] = (r, hits, run)
        sc = {s: per_question(run, qrels, ids) for s, ids in subsets.items()}
        result["dense"][name] = {s: summarize(v) for s, v in sc.items()}
        result["paired_dense_minus_bm25"][name] = {
            s: {m: dict(zip(("diff", "lo", "hi"), paired_bootstrap(sc[s][m], bm_scores[s][m])))
                for m in ("hit@1", "hit@5", "hit@20", "mrr@10")} for s in subsets}
        print(f"\n{name}")
        for s in subsets:
            row = "  ".join(f"{m} {result['dense'][name][s][m]['mean']:.3f}" for m in ("hit@1", "hit@5", "hit@20", "hit@50", "mrr@10"))
            print(f"  {s:11} n={len(subsets[s]):4}  {row}")
    for s in subsets:
        row = "  ".join(f"{m} {result['bm25'][s][m]['mean']:.3f}" for m in ("hit@1", "hit@5", "hit@20", "hit@50", "mrr@10"))
        print(f"  BM25 {s:11} n={len(subsets[s]):4}  {row}")

    if detail:
        r, hits, run = all_dense[detail]
        # BM25 đúng / dense đúng: bảng 2x2 ở Hit@5 và Hit@20, và trần của phép hợp hai danh sách
        agree = {}
        for s, ids in subsets.items():
            agree[s] = {}
            for k in (5, 20):
                b = np.array([hit_at_k(bm_run[q], set(qrels[q]), k) for q in ids], bool)
                d = np.array([hit_at_k(run[q], set(qrels[q]), k) for q in ids], bool)
                u = np.array([hit_at_k(list(dict.fromkeys(bm_run[q][:k] + run[q][:k])), set(qrels[q]), 2 * k) for q in ids], bool)
                agree[s][f"hit@{k}"] = {"both": int((b & d).sum()), "dense_only": int((d & ~b).sum()),
                                        "bm25_only": int((b & ~d).sum()), "neither": int((~b & ~d).sum()),
                                        "union_hit": float(u.mean())}
        result["agreement"] = agree
        # độ trễ trực tuyến: từng câu một, sau khởi động
        r.retrieve(questions[0], 20)
        enc_t, srch_t = [], []
        for qn in questions[:200]:
            r.retrieve(qn, 20)
            enc_t.append(r.last_timing["encode"])
            srch_t.append(r.last_timing["search"])
        t = time.perf_counter()
        for qn in questions[:200]:
            bm25.retrieve(qn, 20)
        bm_ms = (time.perf_counter() - t) / 200 * 1000
        result["latency_ms"] = {"dense_encode_mean": float(np.mean(enc_t) * 1000), "dense_encode_p95": float(np.percentile(enc_t, 95) * 1000),
                                "dense_search_mean": float(np.mean(srch_t) * 1000), "bm25_mean": bm_ms}
        # trần do cắt cụt: đáp án vàng nằm sau điểm cắt của context encoder?
        man = read_json(ROOT / "artifacts" / "dense" / detail / "manifest.json")
        tok = enc.ctx[1]
        by_id = {c["passage_id"]: c for c in corpus}
        beyond = 0
        for q in qs:
            c = by_id[q["passage_id"]]
            a0 = q["answers"][0]["start"]
            text = c[man["passage_field"]]
            n_title = len(tok(c["title"] or "", add_special_tokens=False)["input_ids"]) if man["passage_input"] == "title_text" else 0
            n_before = len(tok(text[:a0], add_special_tokens=False)["input_ids"])
            beyond += (n_title + n_before + 3) > man["truncation"]["max_length"]
        result["answer_beyond_truncation"] = {"questions": len(qs), "count": int(beyond), "fraction": beyond / len(qs),
                                              "max_length": man["truncation"]["max_length"]}
        write_jsonl(REPORT_DIR / "dense_dev_run.jsonl", [{"question_id": qid, "retrieved": hs} for qid, hs in zip(qids, hits)])
        print("\nagreement", agree["dev"], agree["dev_unseen"], "\nlatency", result["latency_ms"],
              "\nbeyond truncation", result["answer_beyond_truncation"])
    out = RESULTS_DIR / ("dense_retrieval_dev.json" if detail else "dense_variants_dev.json")
    write_json(out, result)
    print("đã ghi", out.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", nargs="+", required=True, help="tên thư mục trong artifacts/dense/")
    ap.add_argument("--detail", default=None, help="biến thể chi tiết (độ trễ, phân tích 2x2, cắt cụt)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.artifacts, a.detail, a.limit)
