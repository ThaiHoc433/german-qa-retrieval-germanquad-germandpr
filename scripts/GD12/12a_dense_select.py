"""GĐ12.2: lưu top-100 của BM25 (B0) và dense (B1) cho tập chọn + toàn bộ dev, rồi đo Hit/Recall/MRR có khoảng tin cậy.
Ghi: results/B0_select_top100.jsonl, results/B1_select_top100[_<tag>].jsonl, results/12a_metrics[_<tag>].json"""
import argparse
import time
from pathlib import Path

from germanqa.config import PROCESSED_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, read_jsonl, write_json, write_jsonl
from germanqa.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from germanqa.evaluation.run_cache import per_question, to_run, validate_rows
from germanqa.experiment import load_config
from germanqa.retrieval.bm25 import BM25Retriever
from germanqa.retrieval.dense import DenseRetriever, DPREncoder

TOP_K = 100
SHOW = ("hit@1", "hit@5", "hit@20", "hit@100", "mrr@10", "recall@20")


def load_queries():
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    select_ids = sel["train"] + sel["dev"]
    questions, qrels, dev_ids = {}, {}, []
    for split in ("train", "dev"):
        for q in read_jsonl(PROCESSED_DIR / f"qa_{split}.jsonl"):
            questions[q["question_id"]] = q["question"]
            if split == "dev":
                dev_ids.append(q["question_id"])
        qrels.update(read_json(PROCESSED_DIR / f"qrels_{split}.json"))
    return questions, qrels, select_ids, list(dict.fromkeys(select_ids + dev_ids))


def to_rows(qids, hits_per_q):
    return [{"question_id": q, "hits": [[h["passage_id"], h["score"]] for h in hits]} for q, hits in zip(qids, hits_per_q)]


def write_checked(path, rows, corpus_ids, allowed):
    problems = validate_rows(rows, TOP_K, corpus_ids, allowed)
    if problems:                                   # KHÔNG ghi file hỏng: các bước sau tin file này
        raise SystemExit(f"{path.name}: {len(problems)} lỗi, ví dụ: {problems[:3]}")
    write_jsonl(path, rows)
    print("đã ghi", path.name, f"({len(rows)} câu)")


def main(a):
    cfg = load_config("dense")["dense"]
    pcfg = load_config("pipeline")["bm25"]
    corpus = read_jsonl(PROCESSED_DIR / "corpus.jsonl")
    corpus_ids = {r["passage_id"] for r in corpus}
    questions, qrels, select_ids, all_ids = load_queries()
    allowed = set(all_ids)                                       # chỉ train + dev; không có câu test
    suffix = f"_{a.tag}" if a.tag else ""

    b0_path = RESULTS_DIR / "B0_select_top100.jsonl"
    if a.redo_bm25 or not b0_path.exists():
        t = time.perf_counter()
        bm25 = BM25Retriever(corpus, k1=pcfg["k1"], b=pcfg["b"], use_stopwords=pcfg["use_stopwords"])
        rows = to_rows(all_ids, [bm25.retrieve(questions[q], TOP_K) for q in all_ids])
        write_checked(b0_path, rows, corpus_ids, allowed)
        print(f"BM25: {time.perf_counter() - t:.1f}s")

    art = ROOT / "artifacts" / "dense" / a.artifact
    enc = DPREncoder(cfg["ctx_encoder"], cfg["question_encoder"], fp16=cfg["fp16"])
    dense = DenseRetriever.from_dir(art, enc, PROCESSED_DIR / "corpus.jsonl", cfg["question_max_length"], expected_cfg=cfg)
    t = time.perf_counter()
    rows = to_rows(all_ids, dense.retrieve_many([questions[q] for q in all_ids], TOP_K))
    print(f"dense {a.artifact}: {time.perf_counter() - t:.1f}s")
    write_checked(RESULTS_DIR / f"B1_select_top100{suffix}.jsonl", rows, corpus_ids, allowed)

    b0, b1 = to_run(read_jsonl(b0_path)), to_run(rows)
    result = {"artifact": a.artifact, "n_select": len(select_ids), "metrics": {}, "paired_dense_minus_bm25": {}}
    pq0, pq1 = per_question(b0, qrels, select_ids), per_question(b1, qrels, select_ids)
    for m in SHOW:
        result["metrics"][m] = {"bm25": dict(zip(("mean", "lo", "hi"), bootstrap_ci(pq0[m]))),
                                "dense": dict(zip(("mean", "lo", "hi"), bootstrap_ci(pq1[m])))}
        result["paired_dense_minus_bm25"][m] = dict(zip(("diff", "lo", "hi"), paired_bootstrap(pq1[m], pq0[m])))
        r = result["metrics"][m]
        d = result["paired_dense_minus_bm25"][m]
        print(f"{m:10} BM25 {r['bm25']['mean']:.3f}  dense {r['dense']['mean']:.3f}  hiệu {d['diff']*100:+.1f} [{d['lo']*100:+.1f}; {d['hi']*100:+.1f}]")
    write_json(RESULTS_DIR / f"12a_metrics{suffix}.json", result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="text256", help="thư mục trong artifacts/dense/")
    ap.add_argument("--tag", default="", help="hậu tố tên file kết quả, ví dụ 384")
    ap.add_argument("--redo-bm25", action="store_true")
    main(ap.parse_args())
