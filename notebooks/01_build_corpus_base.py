"""Milestone A — build corpus_base + qa/retrieval-training records + manifest.

Điều kiện chuyển bước (theo yêu cầu): gold tồn tại, offsets đúng, split đúng quy ước.
Chỉ build corpus_base (dataset-derived, không distractor Wikipedia) — corpus_mid/large
để dành cho vòng chạy sau, không cần cho vòng nhỏ dữ liệu→BM25→retrieval metrics.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import CorpusBuilder, find_span_chunks, passage_id
from src.hf_data import load_germandpr, load_germanquad
from src.splits import assign_splits, question_key

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_qa_records(gq_df, split_assignment, builder, coverage_counter):
    records = {"train": [], "dev": [], "test": []}
    for row in gq_df.itertuples(index=False):
        context = row.context
        chunks = builder.add_text(context, "germanquad")

        texts = list(row.answers["text"])
        starts = [int(s) for s in row.answers["answer_start"]]

        gold_indices = set()
        for t, s in zip(texts, starts):
            gold_indices.update(find_span_chunks(chunks, s, s + len(t)))

        coverage_counter["total"] += 1
        split = split_assignment[question_key(row.question)]

        if not gold_indices:
            coverage_counter["uncovered"] += 1
            continue  # loại khỏi eval theo 2.3.c, KHÔNG âm thầm — đếm ở coverage report

        sorted_idx = sorted(gold_indices, key=lambda i: chunks[i].char_start)
        gold_passage_ids = [passage_id(chunks[i].text) for i in sorted_idx]
        primary = chunks[sorted_idx[0]]

        answers_out = []
        for t, s in zip(texts, starts):
            e = s + len(t)
            if primary.char_start <= s and e <= primary.char_end:
                answers_out.append({"text": t, "answer_start": s - primary.char_start})

        records[split].append(
            {
                "query_id": f"gq-{split}-{int(row.id)}",
                "question": row.question,
                "gold_passage_ids": gold_passage_ids,
                "answers": answers_out,
                "source_dataset": "germanquad",
                "source_split": split,
            }
        )
    return records


def build_retrieval_training_records(gdpr_df, split_assignment, builder, prefix):
    records = {"train": [], "dev": [], "test": []}
    for i, row in enumerate(gdpr_df.itertuples(index=False)):
        split = split_assignment[question_key(row.question)]

        pos_ids = []
        for t in row.positive_ctxs["text"]:
            chunks = builder.add_text(t, "germandpr_positive")
            pos_ids.extend(passage_id(c.text) for c in chunks)

        hn_ids = []
        for t in row.hard_negative_ctxs["text"]:
            chunks = builder.add_text(t, "germandpr_hard_negative")
            hn_ids.extend(passage_id(c.text) for c in chunks)

        records[split].append(
            {
                "query_id": f"gdpr-{prefix}-{i}",
                "question": row.question,
                "positive_passage_ids": pos_ids,
                "hard_negative_passage_ids": hn_ids,
                "source_split": split,
            }
        )
    return records


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    print("Loading datasets...")
    gq_train, gq_test = load_germanquad()
    gdpr_train, gdpr_test = load_germandpr()

    print("Assigning unified train/dev/test splits (2.3.e)...")
    split_assignment, split_report = assign_splits(
        gq_train["question"].tolist(),
        gq_test["question"].tolist(),
        gdpr_train["question"].tolist(),
        gdpr_test["question"].tolist(),
    )
    print(json.dumps(split_report, ensure_ascii=False, indent=2))

    builder = CorpusBuilder()
    coverage_counter = {"total": 0, "uncovered": 0}

    print("Building QA records (GermanQuAD) + corpus chunks...")
    qa_records = build_qa_records(gq_train, split_assignment, builder, coverage_counter)
    qa_records_test = build_qa_records(gq_test, split_assignment, builder, coverage_counter)
    for k in qa_records:
        qa_records[k].extend(qa_records_test[k])

    print("Building retrieval training records (GermanDPR) + corpus chunks...")
    rt_train = build_retrieval_training_records(gdpr_train, split_assignment, builder, "train")
    rt_test = build_retrieval_training_records(gdpr_test, split_assignment, builder, "test")
    rt_records = {k: rt_train[k] + rt_test[k] for k in rt_train}

    # --- write corpus_base ---
    corpus_records = builder.records()
    write_jsonl(OUT_DIR / "corpus_base.jsonl", corpus_records)

    # --- write qa / retrieval-training records ---
    for split in ("train", "dev", "test"):
        write_jsonl(OUT_DIR / f"qa_germanquad_{split}.jsonl", qa_records[split])
        write_jsonl(OUT_DIR / f"retrieval_germandpr_{split}.jsonl", rt_records[split])

    # --- chunk coverage report ---
    coverage_report = {
        "total_germanquad_rows": coverage_counter["total"],
        "uncovered_rows": coverage_counter["uncovered"],
        "coverage_rate": round(
            1 - coverage_counter["uncovered"] / coverage_counter["total"], 4
        ),
    }
    with open(OUT_DIR / "chunk_coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, ensure_ascii=False, indent=2)
    print("Chunk coverage:", coverage_report)

    # --- split report (persist) ---
    with open(OUT_DIR / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(split_report, f, ensure_ascii=False, indent=2)

    # --- corpus manifest ---
    source_breakdown = {}
    for r in corpus_records:
        source_breakdown[r["source"]] = source_breakdown.get(r["source"], 0) + 1
    manifest = {
        "corpus_tier": "base",
        "target_size": None,
        "actual_size": len(corpus_records),
        "source_breakdown": source_breakdown,
        "chunk_window_words": 100,
        "chunk_overlap_words": 20,
        "seed": split_report["seed"],
        "qa_record_counts": {k: len(v) for k, v in qa_records.items()},
        "retrieval_training_record_counts": {k: len(v) for k, v in rt_records.items()},
    }
    with open(OUT_DIR / "corpus_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Manifest:", json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
