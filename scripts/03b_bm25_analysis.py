
"""GĐ3 (phân tích): BM25 sai ở đâu trên DEV. Đọc cấu hình đã chọn từ results/bm25_*.json mới nhất."""
import json
import re

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.retrieval.bm25 import BM25Retriever
from germanqa.retrieval.tokenizer import GERMAN_STOPWORDS, tokenize

BUCKETS = [("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20)]


def latest_config():
    path = sorted(RESULTS_DIR.glob("bm25_*.json"))[-1]
    return path.name, json.loads(path.read_text(encoding="utf-8"))["config"]["bm25"]


def question_type(q):
    first = re.findall(r"\w+", q.lower())[0]
    return "welch-" if first.startswith("welch") else first if first in ("was", "wer", "wie", "wo", "wann", "wieviel", "warum", "wem", "wen", "wessen") else "khác"


def main():
    name, c = latest_config()
    corpus = read_jsonl(PROCESSED_DIR / "corpus.jsonl")
    rows = read_jsonl(PROCESSED_DIR / "qa_dev.jsonl")
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    retr = BM25Retriever(corpus, k1=c["k1"], b=c["b"], use_stopwords=c["use_stopwords"])
    text_of = {r["passage_id"]: r["original_text"] for r in corpus}

    ranks, by_type, fails = [], {}, []
    for r in rows:
        hits = [h["passage_id"] for h in retr.retrieve(r["question"], 20)]
        gold = set(qrels[r["question_id"]])
        rank = next((i for i, p in enumerate(hits, 1) if p in gold), None)
        ranks.append(rank)
        t = by_type.setdefault(question_type(r["question"]), [0, 0])
        t[0] += 1
        t[1] += rank is not None
        if rank is None:
            q_tokens = set(tokenize(r["question"], GERMAN_STOPWORDS))
            p_tokens = set(tokenize(text_of[r["passage_id"]]))
            fails.append({"question": r["question"], "gold_words_overlap": len(q_tokens & p_tokens),
                          "question_words": len(q_tokens)})

    n = len(rows)
    dist = {label: sum(1 for x in ranks if x and lo <= x <= hi) for label, lo, hi in BUCKETS}
    dist["không có trong top-20"] = sum(1 for x in ranks if x is None)
    print("config:", name, c)
    print("vị trí đoạn đúng đầu tiên (số câu, %):")
    for k, v in dist.items():
        print(f"  {k:<24}{v:>6}  {100 * v / n:5.1f}%")
    print("theo loại câu hỏi (số câu, % có đoạn đúng trong top-20):")
    for k, (tot, ok) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:<10}{tot:>6}  {100 * ok / tot:5.1f}%")
    zero = sum(1 for f in fails if f["gold_words_overlap"] == 0)
    print(f"câu thất bại: {len(fails)}; trong đó câu hỏi KHÔNG có từ nội dung nào trùng đoạn đúng: {zero}")

    write_json(REPORT_DIR / "bm25_dev_analysis.json", {
        "config": c, "n": n, "rank_distribution": dist,
        "by_question_type": {k: {"n": v[0], "found_in_top20": v[1]} for k, v in by_type.items()},
        "failures": len(fails), "failures_zero_overlap": zero})


if __name__ == "__main__":
    main()