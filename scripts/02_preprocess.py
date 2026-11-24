
"""GĐ2: dữ liệu thô -> corpus, qa_{train,dev,test}, qrels, báo cáo chất lượng."""
import argparse
from pathlib import Path

from datasets import load_from_disk

from germanqa.config import DATASETS, DEV_RATIO, PROCESSED_DIR, RAW_DIR, SEED
from germanqa.data.build_corpus import build_corpus
from germanqa.data.build_qa import build_qa
from germanqa.data.io_utils import file_sha1, write_json, write_jsonl
from germanqa.data.split import make_groups, remove_conflicts_with_test, split_train_dev
from germanqa.experiment import load_config


def by_qid(rows):
    return sorted(rows, key=lambda r: r["question_id"])


def qrels_json(qrels, rows):
    return {r["question_id"]: sorted(qrels[r["question_id"]]) for r in by_qid(rows)}


def main(out_dir, include_dpr_qa):
    out = Path(out_dir)
    raw = {name: load_from_disk(str(RAW_DIR / name)) for name in DATASETS}

    # 1. corpus và dữ liệu QA
    corpus, corpus_stats = build_corpus(raw)
    pools, qrels, hardneg, quarantine, fixlog, qa_stats = build_qa(raw, include_dpr_qa)
    doc_of = {r["passage_id"]: r["document_id"] for r in corpus}

    # 2. R3: loại khỏi train mẫu trùng test
    test_rows = list(pools["test"].values())
    train_all = list(pools["train"].values())
    train_kept, removed = remove_conflicts_with_test(train_all, test_rows, qrels["test"])

    # 3. R5: chia train/dev theo nhóm
    groups = make_groups(train_kept, qrels["train"], doc_of)
    train_rows, dev_rows = split_train_dev(train_kept, groups, DEV_RATIO, SEED)
    sizes = {}
    for g in groups.values():
        sizes[g] = sizes.get(g, 0) + 1

    # 4. ghi kết quả
    write_jsonl(out / "corpus.jsonl", corpus)
    for name, rows in (("train", train_rows), ("dev", dev_rows), ("test", test_rows)):
        write_jsonl(out / f"qa_{name}.jsonl", by_qid(rows))
        source = qrels["test"] if name == "test" else qrels["train"]   # dev lấy từ train
        write_json(out / f"qrels_{name}.json", qrels_json(source, rows))
    kept_ids = {r["question_id"] for r in train_rows}
    write_json(out / "hardneg_train.json",
               {q: hardneg[q] for q in sorted(kept_ids) if q in hardneg})
    write_jsonl(out / "quarantine.jsonl", quarantine)
    write_jsonl(out / "span_fix_log.jsonl", fixlog)
    write_jsonl(out / "conflicts_removed.jsonl", sorted(removed, key=lambda r: r["question_id"]))

    report = {
        "seed": SEED, "dev_ratio": DEV_RATIO, "include_dpr_qa": include_dpr_qa,
        "corpus": corpus_stats, "qa": qa_stats,
        "rows": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)},
        "removed_from_train": {k: sum(1 for r in removed if r["reason"] == k)
                               for k in ("QUESTION_IN_TEST", "PASSAGE_IN_TEST")},
        "groups": {"count": len(sizes), "largest": max(sizes.values())},
        "quarantine": len(quarantine), "span_fixed": len(fixlog),
        "hardneg_questions_in_train": sum(1 for q in kept_ids if q in hardneg),
    }
    write_json(out / "data_quality_report.json", report)

    files = ["corpus.jsonl", "qa_train.jsonl", "qa_dev.jsonl", "qa_test.jsonl",
             "qrels_train.json", "qrels_dev.json", "qrels_test.json", "hardneg_train.json"]
    write_json(out / "split_manifest.json", {
        "seed": SEED, "dev_ratio": DEV_RATIO,
        "question_ids": {"train": sorted(kept_ids),
                         "dev": sorted(r["question_id"] for r in dev_rows),
                         "test": sorted(r["question_id"] for r in test_rows)},
        "file_sha1": {f: file_sha1(out / f) for f in files},
    })
    print("rows", report["rows"], "| corpus", corpus_stats["passages_after_dedup"])
    print("removed", report["removed_from_train"], "| groups", report["groups"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    ap.add_argument("--dpr-qa", action="store_true",
                    help="đưa cả câu hỏi GermanDPR vào QA (mặc định: không)")
    a = ap.parse_args()
    cfg = load_config("baseline")
    main(a.out, a.dpr_qa or cfg["preprocess"]["include_dpr_qa"])