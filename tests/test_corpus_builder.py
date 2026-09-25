import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import chunk_text, passage_id
from src.corpus_builder import build_base_corpus, normalize_title


BERLIN_ID = "fb38a93fd89e1a5f"
ULM_ID = "37bfc8ba5ec56136"
PARIS_ID = "1670f2e42fefa504"


def gq_row(context, **extra):
    """Ví dụ minh họa: build_base_corpus chỉ đọc `context` của GQ."""
    return {"context": context, "id": 1, "question": "noise", **extra}


def dpr_row(texts, titles, passage_ids=None, hard_negative_texts=(), **extra):
    """Ví dụ minh họa: build_base_corpus chỉ đọc `positive_ctxs`; các field
    khác (answers/negative_ctxs/hard_negative_ctxs) chỉ để kiểm bị bỏ qua."""
    if passage_ids is None:
        passage_ids = [""] * len(texts)
    return {
        "question": "noise", "answers": ["noise"],
        "positive_ctxs": {"title": list(titles), "text": list(texts), "passage_id": list(passage_ids)},
        "hard_negative_ctxs": {"title": [None] * len(hard_negative_texts),
                                "text": list(hard_negative_texts),
                                "passage_id": [""] * len(hard_negative_texts)},
        "negative_ctxs": {"title": [], "text": [], "passage_id": []},
        **extra,
    }


def inputs(gq_train=(), gq_test=(), gdpr_train=(), gdpr_test=()):
    return (list(gq_train), list(gq_test), list(gdpr_train), list(gdpr_test))


def rep(result, pid):
    return result["passages"][pid]


# ---------- normalize_title (P4) ----------

@pytest.mark.parametrize("raw,expected", [
    ("Ä_B", "ä b"), (" ä b ", "ä b"), ("A__B", "a b"),
    (None, None), ("___", None),
    ("Ä_B", "ä b"), ("Straße", "straße"), ("STRASSE", "strasse"),
])
def test_normalize_title_nfc_underscore_lower_collapse_strip(raw, expected):
    assert normalize_title(raw) == expected


def test_normalize_title_does_not_casefold_distinct_forms():
    assert normalize_title("Straße") != normalize_title("STRASSE")


def test_normalize_title_rejects_wrong_type():
    with pytest.raises((TypeError, ValueError)):
        normalize_title(123)
    with pytest.raises((TypeError, ValueError)):
        normalize_title(True)


# ---------- D6 source and per-occurrence P4 collection ----------

def test_base_source_is_gq_context_and_dpr_positive_only_ignores_hard_negative():
    result = build_base_corpus(*inputs(
        gq_train=[gq_row("Berlin")],
        gdpr_train=[dpr_row(["Ulm"], [None], hard_negative_texts=["Berlin", "Paris"])],
    ))
    assert set(result["passages"]) == {BERLIN_ID, ULM_ID}
    assert PARIS_ID not in result["passages"]
    assert result["passages"][BERLIN_ID]["source"] == "germanquad"
    assert result["passages"][ULM_ID]["source"] == "germandpr_positive"


def test_no_d1_or_unmappable_filtering_context_always_enters_base():
    # Câu hỏi GQ train trùng DPR test; DPR answers không map được — build_base_corpus
    # không nhận decisions/answers nên không có khái niệm loại theo D1/unmappable.
    result = build_base_corpus(*inputs(
        gq_train=[gq_row("Berlin", question="Wie viele?")],
        gdpr_test=[dpr_row(["Ulm"], [None], question="Wie viele?", answers=["nicht tìm thấy"])],
    ))
    assert set(result["passages"]) == {BERLIN_ID, ULM_ID}


def test_every_chunk_of_a_long_context_enters_base_even_without_any_answer():
    text = " ".join(f"w{i:03d}" for i in range(201))
    result = build_base_corpus(*inputs(gdpr_train=[dpr_row([text], [None], hard_negative_texts=[])]))
    assert set(result["passages"]) == {"c3cbe252c56e270c", "8451b74d508323dc", "78eaa148763a161a"}


# ---------- title collection before dedup (P4) ----------

def test_title_collected_before_dedup_even_when_representative_wins_by_priority():
    result = build_base_corpus(*inputs(
        gq_train=[gq_row("Ä_B\nBerlin")],
        gdpr_train=[dpr_row(["Ä_B\nBerlin"], ["Zweite_Quelle"])],
    ))
    assert set(result["passages"]) == {"65aabd0ccb0e2bbf"}
    rec = rep(result, "65aabd0ccb0e2bbf")
    assert rec["source"] == "germanquad"
    assert rec["title"] == "Ä_B"
    assert result["base_title_exclusions"] == sorted(["zweite quelle", "ä b"])


def test_gq_missing_title_first_line_empty_does_not_invent_title():
    result = build_base_corpus(*inputs(gq_train=[gq_row("\nBerlin")]))
    rec = rep(result, BERLIN_ID)
    assert rec["title"] is None
    assert (rec["chunk_start_char"], rec["chunk_end_char"]) == (1, 7)
    files = result["source_report"]["input_sources"]["germanquad"]["train"]
    assert files["n_missing_titles"] == 1


def test_dpr_missing_title_context_still_enters_base():
    result = build_base_corpus(*inputs(gdpr_train=[dpr_row(["Berlin"], [None])]))
    assert rep(result, BERLIN_ID)["title"] is None
    files = result["source_report"]["input_sources"]["germandpr_positive"]["train"]
    assert files["n_missing_titles"] == 1


def test_empty_context_with_a_title_contributes_no_passage_but_keeps_title():
    result = build_base_corpus(*inputs(gdpr_train=[dpr_row(["   "], ["Known_Title"])]))
    assert result["passages"] == {}
    assert result["base_title_exclusions"] == ["known title"]
    files = result["source_report"]["input_sources"]["germandpr_positive"]["train"]
    assert files["n_empty_contexts"] == 1


def test_representative_priority_keeps_gq_but_both_titles_are_collected():
    result = build_base_corpus(*inputs(
        gq_train=[gq_row("Berlin")],
        gdpr_train=[dpr_row(["berlin"], ["Andere_Quelle"])],
    ))
    assert set(result["passages"]) == {BERLIN_ID}
    rec = rep(result, BERLIN_ID)
    assert rec["source"] == "germanquad" and rec["text"] == "Berlin"
    assert result["base_title_exclusions"] == sorted(["andere quelle", "berlin"])


def test_dpr_title_equal_to_multiline_text_is_not_truncated_to_first_line():
    result = build_base_corpus(*inputs(gdpr_train=[dpr_row(["Gott\nBerlin"], ["Gott\nBerlin"])]))
    assert result["base_title_exclusions"] == ["gott berlin"]


def test_two_occurrences_of_same_hash_and_source_keep_first_occurrences_metadata_together():
    # "  Berlin  " (chunk char_start=2,end=8) và "BERLIN" (char_start=0,end=6) khác
    # provenance dù cùng passage_id — bắt lỗi trộn text/title với provenance sai occurrence.
    forward = build_base_corpus(*inputs(gdpr_train=[dpr_row(
        ["  Berlin  ", "BERLIN"], ["Erste_Quelle", "Zweite_Quelle"])]))
    rec = rep(forward, BERLIN_ID)
    assert rec["text"] == "Berlin" and rec["title"] == "Erste_Quelle"
    assert (rec["chunk_start_char"], rec["chunk_end_char"]) == (2, 8)
    assert rec["parent_document_id"] == BERLIN_ID  # cùng hash ở cả hai occurrence, không phân biệt ca này
    assert forward["base_title_exclusions"] == sorted(["erste quelle", "zweite quelle"])

    backward = build_base_corpus(*inputs(gdpr_train=[dpr_row(
        ["BERLIN", "  Berlin  "], ["Zweite_Quelle", "Erste_Quelle"])]))
    rec2 = rep(backward, BERLIN_ID)
    assert rec2["text"] == "BERLIN" and rec2["title"] == "Zweite_Quelle"
    assert (rec2["chunk_start_char"], rec2["chunk_end_char"]) == (0, 6)
    assert backward["base_title_exclusions"] == forward["base_title_exclusions"]


def test_title_exclusions_are_not_summed_naively_across_groups():
    # Cùng key P4 "ä b" xuất hiện ở hai nhóm nguồn/split khác nhau (GQ train, DPR test),
    # cộng thêm một key riêng ở DPR train để buộc "đếm theo nhóm" khác "đếm lũy kế toàn cục"
    # (nếu implementation lỡ dùng biến đếm toàn cục thay vì tập riêng của từng nhóm,
    # DPR train/test sẽ báo sai 2 thay vì đúng 1).
    result = build_base_corpus(*inputs(
        gq_train=[gq_row("Ä_B\nX")],
        gdpr_train=[dpr_row(["Z"], ["Other_Title"])],
        gdpr_test=[dpr_row(["Y"], ["ä b"])],
    ))
    gq_files = result["source_report"]["input_sources"]["germanquad"]["train"]
    dpr_train_files = result["source_report"]["input_sources"]["germandpr_positive"]["train"]
    dpr_test_files = result["source_report"]["input_sources"]["germandpr_positive"]["test"]
    assert gq_files["n_unique_title_keys"] == 1
    assert dpr_train_files["n_unique_title_keys"] == 1
    assert dpr_test_files["n_unique_title_keys"] == 1
    assert result["source_report"]["n_title_exclusions"] == 2
    assert result["base_title_exclusions"] == sorted(["ä b", "other title"])


def test_two_identical_occurrences_count_as_two_contexts_but_one_passage_and_title():
    result = build_base_corpus(*inputs(gdpr_train=[dpr_row(["Berlin", "Berlin"], ["X", "X"])]))
    assert set(result["passages"]) == {BERLIN_ID}
    assert result["base_title_exclusions"] == ["x"]
    files = result["source_report"]["input_sources"]["germandpr_positive"]["train"]
    assert files["n_contexts"] == 2


# ---------- schema/input validation ----------

@pytest.mark.parametrize("bad_context", [123, None, True, 4.5])
def test_gq_context_wrong_type_fails(bad_context):
    with pytest.raises((TypeError, ValueError)):
        build_base_corpus(*inputs(gq_train=[gq_row(bad_context)]))


@pytest.mark.parametrize("bad_title", [123, True, 4.5])
def test_dpr_title_wrong_type_fails(bad_title):
    with pytest.raises((TypeError, ValueError)):
        build_base_corpus(*inputs(gdpr_train=[dpr_row(["Berlin"], [bad_title])]))


def test_dpr_ctx_group_mismatched_list_lengths_fails():
    row = dpr_row(["Berlin"], ["T"])
    row["positive_ctxs"]["passage_id"] = []  # length mismatch: 1/1/0
    with pytest.raises((TypeError, ValueError)):
        build_base_corpus(*inputs(gdpr_train=[row]))


def test_dpr_raw_passage_id_wrong_type_fails_even_though_unused_for_hashing():
    row = dpr_row(["Berlin"], ["T"], passage_ids=[123])
    with pytest.raises((TypeError, ValueError)):
        build_base_corpus(*inputs(gdpr_train=[row]))


@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_rows_container_rejects_string_instead_of_sequence(slot):
    args = list(inputs())
    args[slot] = "not a sequence of rows"
    with pytest.raises((TypeError, ValueError)):
        build_base_corpus(*args)


@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_empty_string_rows_container_is_rejected_not_silently_zero(slot):
    # Chuỗi rỗng "" duyệt qua `for row in rows` sẽ không lặp lần nào — nếu bỏ kiểm tra
    # kiểu container, hàm sẽ âm thầm trả 0 context thay vì báo lỗi input sai kiểu.
    args = list(inputs())
    args[slot] = ""
    with pytest.raises((TypeError, ValueError)):
        build_base_corpus(*args)


def test_passages_are_returned_sorted_by_passage_id_regardless_of_input_order():
    result = build_base_corpus(*inputs(
        gq_train=[gq_row("Ulm"), gq_row("Berlin")],
        gdpr_train=[dpr_row(["Paris"], [None])],
    ))
    assert list(result["passages"].keys()) == sorted([ULM_ID, BERLIN_ID, PARIS_ID])


# ---------- purity, invariance, aggregate milestones ----------

def test_all_four_empty_inputs_produce_empty_result_with_zero_counts():
    result = build_base_corpus(*inputs())
    assert result["passages"] == {} and result["base_title_exclusions"] == []
    rep_report = result["source_report"]
    assert rep_report["n_contexts_total"] == 0 and rep_report["n_passages"] == 0
    for group in ("germanquad", "germandpr_positive"):
        for split in ("train", "test"):
            files = rep_report["input_sources"][group][split]
            assert files == {"n_contexts": 0, "n_chunks_before_dedup": 0,
                              "n_empty_contexts": 0, "n_missing_titles": 0, "n_unique_title_keys": 0}


def test_inputs_are_not_mutated_and_result_is_a_fresh_copy():
    gq_rows = [gq_row("Berlin")]
    before = deepcopy(gq_rows)
    result = build_base_corpus(*inputs(gq_train=gq_rows))
    assert gq_rows == before
    result["passages"][BERLIN_ID]["text"] = "changed externally"
    result["base_title_exclusions"].append("changed externally")
    again = build_base_corpus(*inputs(gq_train=gq_rows))
    assert again["passages"][BERLIN_ID]["text"] == "Berlin"
    assert "changed externally" not in again["base_title_exclusions"]


def test_representative_records_pass_the_passage_validator_with_parent_text():
    from src.schemas import validate_passage_record
    parent = "Berlin\n Eine Stadt."
    result = build_base_corpus(*inputs(gq_train=[gq_row(parent)]))
    for pid, rec in result["passages"].items():
        assert validate_passage_record(rec, parent_text=parent if rec["parent_document_id"] == passage_id(parent) else None) == [] \
            or validate_passage_record(rec) == []
