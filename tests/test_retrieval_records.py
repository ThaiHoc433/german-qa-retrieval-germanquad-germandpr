"""Fixture T5.5: DPR positive union và hard-negative groups. Expected hash/ID
tính từ src.corpus.passage_id trực tiếp (đối chiếu doi-chieu-T5.5.md §3), không
gọi Hub hoặc ghi artifact."""
from copy import deepcopy
import json

import pytest

from src.corpus import CorpusBuilder, chunk_text, passage_id
from src.retrieval_records import build_retrieval_training_records
from src.schemas import validate_retrieval_training_record
from src.splits import assign_splits


TEXT = " ".join(f"w{i:03d}" for i in range(201))
CHUNK_IDS = ["c3cbe252c56e270c", "8451b74d508323dc", "78eaa148763a161a"]


def ctx_group(*, texts, titles=None, pids=None):
    n = len(texts)
    return {
        "title": [None] * n if titles is None else list(titles),
        "text": list(texts),
        "passage_id": [""] * n if pids is None else list(pids),
    }


def raw(question="Frage A", answers=None, positive_texts=("Berlin",),
        hard_negative_texts=(), negative_texts=()):
    return {
        "question": question,
        "answers": ["Berlin"] if answers is None else list(answers),
        "positive_ctxs": ctx_group(texts=positive_texts),
        "hard_negative_ctxs": ctx_group(texts=hard_negative_texts),
        "negative_ctxs": ctx_group(texts=negative_texts),
    }


def corpus(*contexts):
    builder = CorpusBuilder()
    for context in contexts:
        builder.add_text(context, "germandpr_positive")
    return {r["passage_id"]: r for r in builder.records()}


def inputs(row=None, corpus_contexts=None):
    row = raw() if row is None else row
    decisions, _ = assign_splits([], [], [row], [], dev_ratio=1)
    contexts = row["positive_ctxs"]["text"] if corpus_contexts is None else corpus_contexts
    return [row], [], decisions, corpus(*contexts)


def group(source_text, chunk_texts):
    return {"source_context_id": passage_id(source_text), "texts": list(chunk_texts),
            "passage_ids": [passage_id(t) for t in chunk_texts]}


# ---------- §7 fixture: positive occurrence union ----------

def test_positive_union_selects_only_chunks_with_full_answer_span():
    result = build_retrieval_training_records(*inputs(raw(answers=["w010", "w185"], positive_texts=[TEXT])))
    rec = result["records"]["dev"][0]
    assert rec["positive_passage_ids"] == [CHUNK_IDS[0], CHUNK_IDS[2]]
    assert rec["positive_texts"] == [chunk_text(TEXT)[0].text, chunk_text(TEXT)[2].text]
    assert rec["hard_negative_groups"] == []
    assert rec["source_split"] == "dev"


def test_overlapping_answer_span_dedups_containing_chunks():
    result = build_retrieval_training_records(*inputs(raw(answers=["w085"], positive_texts=[TEXT])))
    rec = result["records"]["dev"][0]
    assert rec["positive_passage_ids"] == [CHUNK_IDS[0], CHUNK_IDS[1]]


def test_every_occurrence_of_answer_is_found_not_just_first():
    words = TEXT.split(" ")
    words[10] = words[185] = "Berlin"
    text = " ".join(words)
    result = build_retrieval_training_records(*inputs(raw(answers=["Berlin"], positive_texts=[text]),
                                                       corpus_contexts=[text]))
    rec = result["records"]["dev"][0]
    assert rec["positive_passage_ids"] == ["5a94dae7398cce00", "d5f147a2841f1e2f"]


def test_self_overlapping_answer_adds_positive_at_last_chunk_boundary():
    # 110 tokens, answer 30 tokens: chunk cuối [160,219) chỉ phủ occurrence
    # bắt đầu 160. Tìm không chồng nhau chỉ thử 0,60,120 và bỏ sót chunk cuối.
    text = " ".join(["a"] * 110)
    answer = " ".join(["a"] * 30)
    result = build_retrieval_training_records(*inputs(raw(answers=[answer], positive_texts=[text])))
    rec = result["records"]["dev"][0]
    assert rec["positive_texts"] == [" ".join(["a"] * 100), answer]
    assert rec["positive_passage_ids"] == ["79361a890669c12e", "6442083dd0e3359b"]


def test_positive_dedup_keeps_first_raw_text_for_same_id_with_different_case():
    # Hai chunk 100 từ cùng hash: chunk 0 toàn Berlin, chunk 1 có
    # 20 Berlin + 80 berlin. Ghi đè khi gặp lại ID sẽ làm mất text đầu.
    first = " ".join(["Berlin"] * 100)
    text = " ".join(["Berlin"] * 100 + ["berlin"] * 80)
    row = raw(answers=["berlin", "Berlin"], positive_texts=[text])
    result = build_retrieval_training_records(*inputs(row))
    rec = result["records"]["dev"][0]
    assert rec["positive_texts"] == [first]
    assert rec["positive_passage_ids"] == ["08269b9f6f3a1a29"]


def test_positive_chunks_follow_source_order_for_sparse_indices():
    # w110 chỉ ở chunk 1; w670 chỉ ở chunk 8. Trên CPython 3.13,
    # set {1,8} duyệt 8 trước 1; output vẫn phải theo vị trí nguồn.
    text = " ".join(f"w{i:03d}" for i in range(740))
    row = raw(answers=["w110", "w670"], positive_texts=[text])
    result = build_retrieval_training_records(*inputs(row))
    rec = result["records"]["dev"][0]
    assert rec["positive_texts"] == [
        " ".join(f"w{i:03d}" for i in range(80, 180)),
        " ".join(f"w{i:03d}" for i in range(640, 740)),
    ]
    assert rec["positive_passage_ids"] == ["8451b74d508323dc", "5f451ba682b6e014"]


def test_no_occurrence_drops_whole_record_as_unmappable():
    row = raw(answers=[TEXT[395:505]], positive_texts=[TEXT])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=[TEXT]))
    assert result["records"]["dev"] == []
    files = result["germandpr_chunk_mapping_report"]["files"]["retrieval_germandpr_dev.jsonl"]
    assert files["n_unmappable"] == 1
    assert files["unmappable_query_ids"] == ["gdpr-train-0"]


@pytest.mark.parametrize("row", [
    raw(answers=[], positive_texts=["Berlin"]),
    raw(answers=["Berlin"], positive_texts=[]),
])
def test_empty_answers_or_no_positive_contexts_are_unmappable(row):
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=["Berlin"]))
    assert result["records"]["dev"] == []
    files = result["germandpr_chunk_mapping_report"]["files"]["retrieval_germandpr_dev.jsonl"]
    assert files["n_unmappable"] == 1


def test_multiple_positive_contexts_keep_context_order_not_answer_order():
    row = raw(answers=["missing", "Ulm", "Berlin"], positive_texts=["Berlin", "Ulm"])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=["Berlin", "Ulm"]))
    rec = result["records"]["dev"][0]
    assert rec["positive_passage_ids"] == ["fb38a93fd89e1a5f", "37bfc8ba5ec56136"]
    assert rec["positive_texts"] == ["Berlin", "Ulm"]


def test_duplicate_positive_occurrences_dedup_by_id():
    row = raw(answers=["Berlin", "Berlin"], positive_texts=["Berlin", "Berlin"])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=["Berlin"]))
    rec = result["records"]["dev"][0]
    assert rec["positive_passage_ids"] == ["fb38a93fd89e1a5f"]
    assert rec["positive_texts"] == ["Berlin"]


def test_short_context_single_chunk_is_still_a_valid_positive():
    result = build_retrieval_training_records(
        *inputs(raw(answers=["Berlin"], positive_texts=["Berlin"]), corpus_contexts=["Berlin"]))
    assert result["records"]["dev"][0]["positive_texts"] == ["Berlin"]


def test_raw_positive_text_kept_even_when_corpus_representative_differs_in_case():
    row = raw(answers=["Berlin"], positive_texts=["Berlin"])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=["berlin"]))
    rec = result["records"]["dev"][0]
    assert rec["positive_texts"] == ["Berlin"]
    assert rec["positive_passage_ids"] == [passage_id("berlin")] == [passage_id("Berlin")]


# ---------- §7 fixture: hard-negative groups and mask ----------

def test_hard_negative_group_masks_gold_overlap_chunks():
    row = raw(answers=["w010", "w185"], positive_texts=[TEXT], hard_negative_texts=[TEXT, "Paris"])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=[TEXT]))
    rec = result["records"]["dev"][0]
    assert rec["hard_negative_groups"] == [group(TEXT, [chunk_text(TEXT)[1].text]), group("Paris", ["Paris"])]
    filtering = result["germandpr_chunk_mapping_report"]["hard_negative_filtering"]
    assert filtering == [{
        "query_id": "gdpr-train-0", "source_context_index": 0, "source_context_id": passage_id(TEXT),
        "removed_passage_ids": [CHUNK_IDS[0], CHUNK_IDS[2]], "group_removed": False, "reason": "gold_overlap",
    }]
    files = result["germandpr_chunk_mapping_report"]["files"]["retrieval_germandpr_dev.jsonl"]
    assert files["n_masked_hard_negative_chunks"] == 2
    assert files["n_dropped_hard_negative_groups"] == 0
    assert files["n_hard_negative_groups"] == 2
    assert files["n_hard_negative_chunks"] == 2


def test_hard_negative_group_fully_masked_is_dropped_but_others_kept():
    row = raw(answers=["Berlin"], positive_texts=["Berlin"], hard_negative_texts=["Berlin", "Paris"])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=["Berlin"]))
    rec = result["records"]["dev"][0]
    assert rec["hard_negative_groups"] == [group("Paris", ["Paris"])]
    filtering = result["germandpr_chunk_mapping_report"]["hard_negative_filtering"]
    assert filtering == [{
        "query_id": "gdpr-train-0", "source_context_index": 0, "source_context_id": passage_id("Berlin"),
        "removed_passage_ids": [passage_id("Berlin")], "group_removed": True, "reason": "gold_overlap",
    }]
    files = result["germandpr_chunk_mapping_report"]["files"]["retrieval_germandpr_dev.jsonl"]
    assert files["n_dropped_hard_negative_groups"] == 1
    assert files["n_masked_hard_negative_chunks"] == 1


def test_empty_hard_negative_context_produces_no_group_and_empty_context_reason():
    row = raw(answers=["Berlin"], positive_texts=["Berlin"], hard_negative_texts=["   "])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=["Berlin"]))
    assert result["records"]["dev"][0]["hard_negative_groups"] == []
    filtering = result["germandpr_chunk_mapping_report"]["hard_negative_filtering"]
    assert filtering == [{
        "query_id": "gdpr-train-0", "source_context_index": 0, "source_context_id": passage_id("   "),
        "removed_passage_ids": [], "group_removed": True, "reason": "empty_context",
    }]
    files = result["germandpr_chunk_mapping_report"]["files"]["retrieval_germandpr_dev.jsonl"]
    assert files["n_dropped_hard_negative_groups"] == 1
    assert files["n_masked_hard_negative_chunks"] == 0


def test_same_source_context_id_hard_negative_occurrences_kept_as_separate_groups():
    row = raw(answers=["Berlin"], positive_texts=["Berlin"], hard_negative_texts=["Paris", "Paris"])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=["Berlin"]))
    rec = result["records"]["dev"][0]
    assert rec["hard_negative_groups"] == [group("Paris", ["Paris"]), group("Paris", ["Paris"])]


def test_duplicate_chunk_ids_within_one_hard_negative_group_are_not_deduped():
    x_text = " ".join(["x"] * 201)
    row = raw(answers=["Berlin"], positive_texts=["Berlin"], hard_negative_texts=[x_text])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=["Berlin"]))
    rec = result["records"]["dev"][0]
    chunks = chunk_text(x_text)
    assert len(chunks) == 3
    assert rec["hard_negative_groups"] == [group(x_text, [c.text for c in chunks])]
    assert rec["hard_negative_groups"][0]["passage_ids"] == [
        "efa6349ac9ff5f40", "efa6349ac9ff5f40", "66bc7fc85822c579"]


def test_hard_negative_mask_is_scoped_per_query_not_shared():
    row_a = raw(question="A", answers=["Berlin"], positive_texts=["Berlin"], hard_negative_texts=["Ulm"])
    row_b = raw(question="B", answers=["Ulm"], positive_texts=["Ulm"])
    decisions, _ = assign_splits([], [], [row_a, row_b], [], dev_ratio=1)
    result = build_retrieval_training_records([row_a, row_b], [], decisions, corpus("Berlin", "Ulm"))
    rec_a = next(r for r in result["records"]["dev"] if r["query_id"] == "gdpr-train-0")
    assert rec_a["hard_negative_groups"] == [group("Ulm", ["Ulm"])]


# ---------- D1 / split identity ----------

def test_original_test_rows_only_get_test_split():
    test_row = raw(answers=["Berlin"], positive_texts=["Berlin"])
    decisions, _ = assign_splits([], [], [], [test_row])
    result = build_retrieval_training_records([], [test_row], decisions, corpus("Berlin"))
    assert result["records"]["test"][0]["query_id"] == "gdpr-test-0"
    assert result["records"]["train"] == result["records"]["dev"] == []


def test_leakage_excluded_rows_are_not_looked_up_in_corpus():
    rows, test, decisions, passages = inputs()
    decisions[0]["source_split"] = None
    result = build_retrieval_training_records(rows, test, decisions, {})
    assert all(not r for r in result["records"].values())
    report = result["germandpr_chunk_mapping_report"]
    assert report["excluded_leakage_rows"] == 1
    assert report["excluded_leakage_query_ids"] == ["gdpr-train-0"]


def test_invalid_excluded_row_is_not_silently_skipped():
    rows, test, decisions, passages = inputs()
    decisions[0]["source_split"] = None
    rows[0]["positive_ctxs"] = {"title": [None], "text": ["Berlin"], "passage_id": []}
    with pytest.raises(ValueError, match="positive_ctxs"):
        build_retrieval_training_records(rows, test, decisions, passages)


@pytest.mark.parametrize("assigned", [None, "train", "dev"])
def test_original_test_cannot_be_removed_or_reassigned(assigned):
    test_row = raw(answers=["Berlin"], positive_texts=["Berlin"])
    decisions, _ = assign_splits([], [], [], [test_row])
    decisions[0]["source_split"] = assigned
    with pytest.raises(ValueError, match="source_split"):
        build_retrieval_training_records([], [test_row], decisions, corpus("Berlin"))


# ---------- report conservation and ordering ----------

def test_report_conserves_counts_and_orders_filtering_train_then_test():
    train_rows = [
        raw(question="A", answers=["Berlin"], positive_texts=["Berlin"], hard_negative_texts=["Berlin"]),
        raw(question="B", answers=["missing"], positive_texts=["Berlin"]),
    ]
    test_rows = [raw(question="C", answers=["Ulm"], positive_texts=["Ulm"], hard_negative_texts=["   ", "Ulm"])]
    decisions, _ = assign_splits([], [], train_rows, test_rows, dev_ratio=0)
    result = build_retrieval_training_records(train_rows, test_rows, decisions, corpus("Berlin", "Ulm"))
    report = result["germandpr_chunk_mapping_report"]
    assert report["input_rows"] == 3
    assert report["excluded_leakage_rows"] == 0
    train_file = report["files"]["retrieval_germandpr_train.jsonl"]
    test_file = report["files"]["retrieval_germandpr_test.jsonl"]
    assert train_file["n_input_after_d1"] == 2
    assert train_file["n_records"] == 1 and train_file["n_unmappable"] == 1
    assert train_file["unmappable_query_ids"] == ["gdpr-train-1"]
    assert test_file["n_records"] == 1 and test_file["n_unmappable"] == 0
    assert [f["query_id"] for f in report["hard_negative_filtering"]] == [
        "gdpr-train-0", "gdpr-test-0", "gdpr-test-0"]
    assert [f["source_context_index"] for f in report["hard_negative_filtering"]] == [0, 0, 1]


def test_empty_input_has_no_division_by_zero():
    result = build_retrieval_training_records([], [], [], {})
    assert result["records"] == {"train": [], "dev": [], "test": []}
    report = result["germandpr_chunk_mapping_report"]
    assert report["input_rows"] == report["excluded_leakage_rows"] == 0
    for split in ("train", "dev", "test"):
        f = report["files"][f"retrieval_germandpr_{split}.jsonl"]
        assert f["n_records"] == f["n_unmappable"] == 0
        assert f["positive_chunks_per_record"] == {}
        assert f["hard_negative_groups_per_record"] == {}
        assert f["all_positive_texts_are_raw_chunks"] is True
        assert f["no_null_positive_ids"] is True
    assert report["hard_negative_filtering"] == []


def test_inputs_are_not_mutated_results_validate_and_repeat():
    row = raw(answers=["w010", "w185"], positive_texts=[TEXT], hard_negative_texts=[TEXT, "Paris"])
    args = inputs(row, corpus_contexts=[TEXT])
    before = deepcopy(args)
    result = build_retrieval_training_records(*args)
    assert args == before
    assert build_retrieval_training_records(*args) == result
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result
    for records in result["records"].values():
        assert all(validate_retrieval_training_record(r) == [] for r in records)


# ---------- invalid raw shape ----------

@pytest.mark.parametrize("field,value,match", [
    ("question", " ", "question"), ("question", None, "question"),
    ("answers", "Berlin", "answers"), ("answers", [None], "answer"), ("answers", [" "], "answer"),
])
def test_invalid_raw_row_fields_fail_before_mapping(field, value, match):
    rows, test, decisions, passages = inputs()
    rows[0][field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        build_retrieval_training_records(rows, test, decisions, passages)


@pytest.mark.parametrize("field,value,match", [
    ("positive_ctxs", "not a mapping", "positive_ctxs"),
    ("positive_ctxs", {"title": [None], "text": ["Berlin"], "passage_id": []}, "positive_ctxs"),
    ("positive_ctxs", {"title": [None], "text": [123], "passage_id": [""]}, "text"),
    ("positive_ctxs", {"title": [1], "text": ["Berlin"], "passage_id": [""]}, "title"),
    ("positive_ctxs", {"title": [None], "text": ["Berlin"], "passage_id": [None]}, "passage_id"),
    ("hard_negative_ctxs", "not a mapping", "hard_negative_ctxs"),
    ("hard_negative_ctxs", {"title": [], "text": ["a"], "passage_id": []}, "hard_negative_ctxs"),
    ("negative_ctxs", "not a mapping", "negative_ctxs"),
])
def test_invalid_ctx_group_shape_fails(field, value, match):
    rows, test, decisions, passages = inputs()
    rows[0][field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        build_retrieval_training_records(rows, test, decisions, passages)


# ---------- decision identity / one-to-one ----------

@pytest.mark.parametrize("field,value,match", [
    ("source_dataset", "unknown", "source_dataset"), ("source_dataset", "germanquad", "decision"),
    ("query_id", "gdpr-train-99", "decision"), ("query_id", [], "query_id"),
    ("original_split", "test", "original_split"), ("source_row_index", 1, "source_row_index"),
    ("source_row_index", False, "source_row_index"), ("source_row_index", 0.0, "source_row_index"),
    ("question_key", "FRAGE A", "question_key"), ("source_split", "test", "source_split"),
    ("source_split", "other", "source_split"), ("source_split", [], "source_split"),
])
def test_split_identity_must_match_raw(field, value, match):
    rows, test, decisions, passages = inputs()
    decisions[0][field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        build_retrieval_training_records(rows, test, decisions, passages)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "missing_field", "non_mapping"])
def test_decisions_are_one_to_one(change):
    rows, test, decisions, passages = inputs()
    if change == "missing":
        decisions.clear()
    elif change == "duplicate":
        decisions.append(deepcopy(decisions[0]))
    elif change == "extra":
        decisions.append({**decisions[0], "query_id": "gdpr-train-99"})
    elif change == "missing_field":
        del decisions[0]["source_split"]
    else:
        decisions[0] = "not a decision"
    with pytest.raises((TypeError, ValueError), match="decision"):
        build_retrieval_training_records(rows, test, decisions, passages)


@pytest.mark.parametrize("index,value", [(0, "old inputs"), (1, {}), (2, {}), (3, []), (0, ["not a row"])])
def test_invalid_input_containers(index, value):
    args = list(inputs())
    args[index] = value
    with pytest.raises((TypeError, ValueError)):
        build_retrieval_training_records(*args)


# ---------- positive corpus membership ----------

@pytest.mark.parametrize("change", ["missing", "wrong_key", "missing_field", "non_mapping"])
def test_missing_or_corrupt_positive_passage_fails(change):
    rows, test, decisions, passages = inputs()
    pid = passage_id("Berlin")
    if change == "missing":
        passages.clear()
    elif change == "wrong_key":
        passages[pid] = next(iter(corpus("Ulm").values()))
    elif change == "missing_field":
        del passages[pid]["title"]
    else:
        passages[pid] = None
    with pytest.raises((TypeError, ValueError), match="passage|Passage"):
        build_retrieval_training_records(rows, test, decisions, passages)


def test_hard_negative_ids_absent_from_corpus_are_valid():
    row = raw(answers=["Berlin"], positive_texts=["Berlin"], hard_negative_texts=["Paris"])
    result = build_retrieval_training_records(*inputs(row, corpus_contexts=["Berlin"]))
    rec = result["records"]["dev"][0]
    assert rec["hard_negative_groups"] == [group("Paris", ["Paris"])]
