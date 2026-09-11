"""Data audit for GermanQuAD + GermanDPR.

Trước khi build corpus/split/chunking thật (module 2 trong
docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md),
script này kiểm tra dữ liệu thật để xác nhận các giả định trong spec:
  1. Field nào tồn tại, positive/hard-negative lưu theo cấu trúc gì.
  2. Số unique passage sau dedup (theo đúng normalize+hash rule đã chốt).
  3. Câu hỏi trùng giữa dataset/split (input cho quy trình split hợp nhất 2.3.e).
  4. Phân phối độ dài context theo từ và theo tokenizer gbert-base (input cho quyết định chunking 2.3.c).
  5. Answer offset có khớp text không, và answer text có xuất hiện nhiều lần trong context không
     (không mặc định lấy occurrence đầu khi remap offset).

Chỉ đọc dữ liệu và in báo cáo — KHÔNG ghi corpus/split files (đó là việc của module 2).
"""
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import BertWordPieceTokenizer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_split(repo: str, split: str) -> pd.DataFrame:
    """Load qua parquet auto-convert branch — repo gốc chỉ có loading script cũ,
    không còn được `datasets` mới hỗ trợ (xem log audit run đầu tiên)."""
    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision="refs/convert/parquet",
        filename=f"plain_text/{split}/0000.parquet",
    )
    return pd.read_parquet(path)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def passage_id(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def question_key(q: str) -> str:
    return normalize_text(q)


def to_native(obj):
    """Ép numpy/pandas scalar về kiểu Python thuần để json.dump không lỗi."""
    if isinstance(obj, dict):
        return {str(to_native(k)): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_native(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def percentile_stats(arr) -> dict:
    s = sorted(arr)
    n = len(s)

    def pct(p):
        return s[int(p * (n - 1))]

    return {
        "n": n,
        "min": s[0],
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": s[-1],
        "mean": round(sum(s) / n, 1),
    }


def main():
    report = {}

    print("Loading datasets (GermanQuAD, GermanDPR)...")
    gq_train = load_split("deepset/germanquad", "train")
    gq_test = load_split("deepset/germanquad", "test")
    gdpr_train = load_split("deepset/germandpr", "train")
    gdpr_test = load_split("deepset/germandpr", "test")
    gq_all = pd.concat([gq_train, gq_test], ignore_index=True)

    print(f"GermanQuAD train={len(gq_train)} test={len(gq_test)}")
    print(f"GermanDPR   train={len(gdpr_train)} test={len(gdpr_test)}")

    # --- 1. Field structure ---
    report["fields"] = {
        "germanquad_columns": list(gq_train.columns),
        "germandpr_columns": list(gdpr_train.columns),
    }

    def count_dist(df, col):
        counts = df[col].apply(lambda d: len(d["text"]))
        return {str(k): int(v) for k, v in counts.value_counts().items()}

    report["germandpr_train_positive_count_dist"] = count_dist(gdpr_train, "positive_ctxs")
    report["germandpr_train_hard_negative_count_dist"] = count_dist(gdpr_train, "hard_negative_ctxs")
    report["germandpr_train_negative_count_dist"] = count_dist(gdpr_train, "negative_ctxs")
    report["germandpr_test_positive_count_dist"] = count_dist(gdpr_test, "positive_ctxs")
    report["germandpr_test_hard_negative_count_dist"] = count_dist(gdpr_test, "hard_negative_ctxs")

    n_no_answer_gq = int((gq_all["answers"].apply(lambda a: len(a["text"])) == 0).sum())
    report["germanquad_rows_with_zero_answers"] = n_no_answer_gq

    # --- 2. Unique passages sau dedup ---
    all_passages = {}  # passage_id -> set(sources)

    def add_passage(text, source):
        pid = passage_id(text)
        all_passages.setdefault(pid, set()).add(source)

    for context in gq_all["context"]:
        add_passage(context, "germanquad")
    for df in (gdpr_train, gdpr_test):
        for row in df.itertuples(index=False):
            for t in row.positive_ctxs["text"]:
                add_passage(t, "germandpr_positive")
            for t in row.hard_negative_ctxs["text"]:
                add_passage(t, "germandpr_hard_negative")

    report["unique_passage_count"] = len(all_passages)
    source_counter = Counter()
    for sources in all_passages.values():
        for s in sources:
            source_counter[s] += 1
    report["passage_source_breakdown"] = dict(source_counter)
    report["passages_shared_across_sources"] = sum(1 for s in all_passages.values() if len(s) > 1)

    # --- 3. Câu hỏi trùng giữa dataset/split ---
    def qkeys(df):
        return set(df["question"].map(question_key))

    keys = {
        "germanquad_train": qkeys(gq_train),
        "germanquad_test": qkeys(gq_test),
        "germandpr_train": qkeys(gdpr_train),
        "germandpr_test": qkeys(gdpr_test),
    }
    report["question_counts_raw"] = {k: len(v) for k, v in keys.items()}

    names = list(keys.keys())
    overlap = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap[f"{a} ∩ {b}"] = len(keys[a] & keys[b])
    report["question_overlap"] = overlap

    leak_a = keys["germandpr_train"] & keys["germanquad_test"]
    leak_b = keys["germanquad_train"] & keys["germandpr_test"]
    report["leakage_germandpr_train_in_germanquad_test"] = len(leak_a)
    report["leakage_germanquad_train_in_germandpr_test"] = len(leak_b)
    report["leakage_examples"] = list(leak_a | leak_b)[:5]

    # --- 4. Phân phối độ dài context ---
    # AutoTokenizer.from_pretrained thất bại trên transformers 5.x vì repo
    # deepset/gbert-base chỉ có vocab.txt (không có tokenizer.json fast-tokenizer
    # đã convert sẵn) và slow-tokenizer fallback path bị lỗi trong môi trường này.
    # Load thẳng vocab.txt qua `tokenizers.BertWordPieceTokenizer` (cased, đúng gbert-base).
    print("Loading tokenizer deepset/gbert-base (vocab.txt trực tiếp)...")
    vocab_path = hf_hub_download(repo_id="deepset/gbert-base", filename="vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=False)

    all_contexts = gq_all["context"].tolist()
    word_lens = [len(c.split()) for c in all_contexts]

    random.seed(42)
    sample = random.sample(all_contexts, min(2000, len(all_contexts)))
    tok_lens = [len(tokenizer.encode(c).ids) - 2 for c in sample]  # trừ [CLS]/[SEP]

    report["context_length_words"] = percentile_stats(word_lens)
    report["context_length_gbert_tokens_sample_n"] = len(sample)
    report["context_length_gbert_tokens"] = percentile_stats(tok_lens)
    report["pct_context_over_300_gbert_tokens_sampled"] = round(
        sum(1 for t in tok_lens if t > 300) / len(tok_lens), 4
    )

    # --- 5. Answer offset validation ---
    mismatches = []
    multi_occurrence_count = 0
    not_first_occurrence_count = 0
    checked = 0

    for row in gq_all.itertuples(index=False):
        context = row.context
        texts = row.answers["text"]
        starts = row.answers["answer_start"]
        for t, s in zip(texts, starts):
            checked += 1
            s = int(s)
            actual = context[s : s + len(t)]
            if actual != t:
                mismatches.append(
                    {"id": int(row.id), "expected": t, "actual": actual, "answer_start": s}
                )
            occurrences = [m.start() for m in re.finditer(re.escape(t), context)]
            if len(occurrences) > 1:
                multi_occurrence_count += 1
                first = occurrences[0]
                if s != first:
                    not_first_occurrence_count += 1

    report["answer_offset_checked"] = checked
    report["answer_offset_mismatches"] = len(mismatches)
    report["answer_offset_mismatch_examples"] = mismatches[:5]
    report["answers_with_multiple_occurrences_in_context"] = multi_occurrence_count
    report["answers_with_multiple_occurrences_pct"] = round(multi_occurrence_count / checked, 4)
    report["answer_start_not_first_occurrence_count"] = not_first_occurrence_count

    # --- save + print ---
    report = to_native(report)
    out_path = RESULTS_DIR / "data_audit_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 60)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
