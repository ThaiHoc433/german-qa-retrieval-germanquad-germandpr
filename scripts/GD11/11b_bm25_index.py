"""GĐ11.2: dựng, lưu và đo chỉ mục BM25; chứng minh nạp từ tệp cho kết quả GIỐNG HỆT dựng mới.
Chỉ dùng qa_train và qa_dev (không mở test). Ghi results/11b_bm25_store.json."""
import statistics
import time

from germanqa.config import PROCESSED_DIR, RESULTS_DIR
from germanqa.data.io_utils import file_sha1, read_json, read_jsonl, write_json
from germanqa.experiment import load_config
from germanqa.retrieval import bm25_store
from germanqa.retrieval.bm25 import BM25Retriever


def main():
    corpus_path = PROCESSED_DIR / "corpus.jsonl"
    corpus = read_jsonl(corpus_path)
    b = load_config("pipeline")["bm25"]
    t = time.perf_counter()
    fresh = BM25Retriever(corpus, k1=b["k1"], b=b["b"], use_stopwords=b["use_stopwords"])
    t_build = time.perf_counter() - t
    fp = bm25_store.compute_fingerprint(file_sha1(corpus_path), b["k1"], b["b"], b["use_stopwords"])
    t = time.perf_counter()
    d = bm25_store.save_index(fresh, fp, bm25_store.DEFAULT_DIR)
    t_save = time.perf_counter() - t
    size_mb = sum(p.stat().st_size for p in d.iterdir()) / 1024 ** 2
    loads = []
    for _ in range(3):
        t = time.perf_counter()
        loaded = bm25_store.load_index(d, fp)
        loads.append(time.perf_counter() - t)

    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    ids = set(sel["train"]) | set(sel["dev"])
    qs = [r for s in ("train", "dev") for r in read_jsonl(PROCESSED_DIR / f"qa_{s}.jsonl") if r["question_id"] in ids]
    diff = sum(1 for q in qs if fresh.retrieve(q["question"], 100) != loaded.retrieve(q["question"], 100))  # passage_id, điểm, hạng
    out = {"fingerprint": fp["sha1"], "fingerprint_parts": fp["parts"], "store_dir": "artifacts/bm25", "size_mb": round(size_mb, 1),
           "n_terms": len(fresh.index.weights), "n_docs": fresh.index.n,
           "seconds": {"build": round(t_build, 2), "save": round(t_save, 2), "load_each": [round(x, 3) for x in loads],
                       "load_median": round(statistics.median(loads), 3)},
           "equivalence": {"questions": len(qs), "top_k": 100, "questions_with_any_difference": diff,
                           "splits": "train+dev trong tập chọn; không mở test"}}
    write_json(RESULTS_DIR / "11b_bm25_store.json", out)
    print(out["seconds"], out["equivalence"], "MB", out["size_mb"])


if __name__ == "__main__":
    main()
