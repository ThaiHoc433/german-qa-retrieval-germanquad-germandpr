"""GĐ6.2: encoder dense đã học trên GermanDPR train; dev/test của dự án có trùng với dữ liệu đó không?
Ghi reports/dense_leakage.json và results/dev_unseen_question_ids.json (dev mà encoder chưa thấy câu hỏi)."""
from datasets import load_from_disk

from germanqa.config import PROCESSED_DIR, RAW_DIR, REPORT_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.experiment_rules import log_test_metadata
from germanqa.evaluation.leakage import normalize_question, seen_question_flags, summarize, unseen_ids


def main():
    log_test_metadata("06a_leakage_check.py", "đối chiếu câu hỏi test với GermanDPR để kiểm tra rò rỉ; không tính chỉ số chất lượng",
                      files=["qa_test.jsonl", "qrels_test.json"])
    dpr = load_from_disk(str(RAW_DIR / "germandpr"))
    seen = {"dpr_train": {normalize_question(r["question"]) for r in dpr["train"]},
            "dpr_test": {normalize_question(r["question"]) for r in dpr["test"]}}
    corpus = {r["passage_id"]: r for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}

    report = {}
    for split in ("train", "dev", "test"):
        qs = read_jsonl(PROCESSED_DIR / f"qa_{split}.jsonl")
        qrels = read_json(PROCESSED_DIR / f"qrels_{split}.json")
        flags = seen_question_flags(qs, seen)
        gold_seen = {"as_dpr_train_positive": 0, "in_dpr_train_any_role": 0}
        for q in qs:
            srcs = [x for p in qrels[q["question_id"]] for x in corpus[p]["source"] if x["dataset"] == "germandpr" and x["split"] == "train"]
            gold_seen["as_dpr_train_positive"] += any(x["role"] == "positive" for x in srcs)
            gold_seen["in_dpr_train_any_role"] += bool(srcs)
        report[split] = {**summarize(flags), "gold_passage": gold_seen,
                         "unseen_question": len(unseen_ids(flags))}
        print(split, report[split])
        if split == "dev":
            write_json(REPORT_DIR.parent / "results" / "dev_unseen_question_ids.json", unseen_ids(flags))
    write_json(REPORT_DIR / "dense_leakage.json", report)


if __name__ == "__main__":
    main()
