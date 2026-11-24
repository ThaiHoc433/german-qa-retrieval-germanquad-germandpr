"""GĐ12: các lần dense trượt ở top-5 thuộc loại nào? (chỉ tập chọn; không mở test)
 (a) đáp án nằm sẵn trong một đoạn top-5 dù đoạn đó không có trong qrels  -> nhãn thiếu, reader có thể vẫn trả đúng
 (b) top-5 có đoạn CÙNG BÀI với đoạn vàng nhưng không phải đoạn vàng        -> đúng bài, sai đoạn (rerank giúp được)
 (c) còn lại                                                               -> trượt thật"""
import argparse
import re

from germanqa.config import PROCESSED_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.evaluation.retrieval_metrics import hit_at_k
from germanqa.evaluation.run_cache import as_ranked, read_run


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def main(tag=''):
    corpus = {r["passage_id"]: r for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    ids = list(sel["train"]) + list(sel["dev"])
    qa = {q["question_id"]: q for q in read_jsonl(PROCESSED_DIR / "qa_train.jsonl") + read_jsonl(PROCESSED_DIR / "qa_dev.jsonl")}
    qrels = {**read_json(PROCESSED_DIR / "qrels_train.json"), **read_json(PROCESSED_DIR / "qrels_dev.json")}
    runs = {"dense": as_ranked(read_run(RESULTS_DIR / f"B1_select_top100{tag}.jsonl")), "bm25": as_ranked(read_run(RESULTS_DIR / "B0_select_top100.jsonl"))}
    ntext = {pid: norm(r["original_text"]) for pid, r in corpus.items()}
    out = {}
    for name, run in runs.items():
        miss = [q for q in ids if not hit_at_k(run[q], set(qrels[q]), 5)]
        kinds = {"a_answer_in_top5": 0, "a_specific_answer_in_top5(>=15 ký tự)": 0, "b_same_article": 0, "b_only(not a)": 0, "c_other": 0}
        for q in miss:
            top5 = run[q][:5]
            ans = [norm(a["text"]) for a in qa[q]["answers"] if a.get("text")]
            gold_docs = {corpus[p]["document_id"] for p in qrels[q]}
            a_any = any(a and a in ntext[p] for p in top5 for a in ans)
            a_spec = any(len(a) >= 15 and a in ntext[p] for p in top5 for a in ans)
            b = any(corpus[p]["document_id"] in gold_docs for p in top5)
            kinds["a_answer_in_top5"] += a_any
            kinds["a_specific_answer_in_top5(>=15 ký tự)"] += a_spec
            kinds["b_same_article"] += b
            kinds["b_only(not a)"] += (b and not a_any)
            kinds["c_other"] += (not a_any and not b)
        out[name] = {"n_miss@5": len(miss), **kinds}
    write_json(RESULTS_DIR / f"12c_miss_kinds{tag}.json", out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="vd _384 để phân tích B1_select_top100_384.jsonl")
    main(ap.parse_args().tag)
