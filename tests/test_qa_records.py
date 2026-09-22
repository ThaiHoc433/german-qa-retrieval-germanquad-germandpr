"""Fixture T5.3/T5.4: expected tính tay, không gọi Hub hoặc ghi artifact."""
from copy import deepcopy
import json

import pytest

from src.corpus import CorpusBuilder, chunk_text, passage_id
from src.qa_records import build_qa_records
from src.schemas import validate_qa_record
from src.splits import assign_splits


TEXT = " ".join(f"w{i:03d}" for i in range(201))
CHUNK_IDS = ["c3cbe252c56e270c", "8451b74d508323dc", "78eaa148763a161a"]


def raw(context="Berlin", texts=None, starts=None, *, raw_id=1, question="Frage A"):
    return {"id": raw_id, "question": question, "context": context,
            "answers": {"text": ["Berlin"] if texts is None else texts,
                        "answer_start": [0] if starts is None else starts}}


def corpus(*contexts):
    builder = CorpusBuilder()
    for context in contexts:
        builder.add_text(context, "germanquad")
    return {r["passage_id"]: r for r in builder.records()}


def inputs(row=None, representative=None):
    row = raw() if row is None else row
    decisions, _ = assign_splits([row], [], [], [], dev_ratio=1)
    return [row], [], decisions, corpus(row["context"] if representative is None else representative)


def evidence(pid, text, start):
    return {"passage_id": pid, "answer_text": text,
            "answer_start": start, "answer_end": start + len(text)}


def test_overlap_emits_all_local_evidences():
    result = build_qa_records(*inputs(raw(TEXT, ["w085"], [425])))
    assert result["records"]["dev"] == [{
        "query_id": "gq-train-1", "question": "Frage A", "source_dataset": "germanquad",
        "source_split": "dev", "reference_answers": ["w085"],
        "gold_evidences": [evidence(CHUNK_IDS[0], "w085", 425), evidence(CHUNK_IDS[1], "w085", 25)],
    }]
    assert result["records"]["train"] == result["records"]["test"] == []


def test_multiple_references_map_to_all_containing_chunks():
    result = build_qa_records(*inputs(raw(TEXT, ["w085", "w185"], [425, 925])))
    rec = result["records"]["dev"][0]
    assert rec["reference_answers"] == ["w085", "w185"]
    assert rec["gold_evidences"] == [evidence(CHUNK_IDS[0], "w085", 425),
        evidence(CHUNK_IDS[1], "w085", 25), evidence(CHUNK_IDS[2], "w185", 125)]


def test_all_references_survive_unmapped_answers():
    result = build_qa_records(*inputs(raw(TEXT, ["w085", TEXT[395:505]], [425, 395])))
    rec = result["records"]["dev"][0]
    assert rec["reference_answers"] == ["w085", TEXT[395:505]]
    assert rec["gold_evidences"] == [evidence(CHUNK_IDS[0], "w085", 425), evidence(CHUNK_IDS[1], "w085", 25)]


@pytest.mark.parametrize("row", [raw(TEXT, [TEXT[395:505]], [395]), raw("", [], []), raw("  ", [], [])])
def test_unmapped_and_empty_answers_keep_qa_record(row):
    result = build_qa_records(*inputs(row))
    rec = result["records"]["dev"][0]
    assert rec["reference_answers"] == row["answers"]["text"]
    assert rec["gold_evidences"] == []
    assert result["coverage_report"]["files"]["qa_germanquad_dev.jsonl"] == {
        "n_records": 1, "n_empty_gold": 1, "coverage_rate": 0.0, "empty_gold_query_ids": ["gq-train-1"]}


def test_annotated_occurrence_is_used_without_search():
    result = build_qa_records(*inputs(raw("alpha Berlin omega Berlin end", ["Berlin"], [19])))
    assert result["records"]["dev"][0]["gold_evidences"] == [evidence("09f06d8cf4a66bda", "Berlin", 19)]
    assert result["ambiguous_offset_remap_report"] == []


def test_duplicate_references_dedup_only_exact_evidence():
    row = raw("alpha Berlin omega Berlin end", ["Berlin", "Berlin", "Berlin"], [6, 19, 19])
    result = build_qa_records(*inputs(row))
    rec = result["records"]["dev"][0]
    assert rec["reference_answers"] == ["Berlin", "Berlin", "Berlin"]
    assert rec["gold_evidences"] == [evidence("09f06d8cf4a66bda", "Berlin", 6), evidence("09f06d8cf4a66bda", "Berlin", 19)]


def test_reference_unicode_casing_and_whitespace_are_preserved():
    row = raw("Ort:  Berlin und Äpfel", [" Berlin", "Äpfel"], [5, 17], question="  WOHER?  ")
    rec = build_qa_records(*inputs(row))["records"]["dev"][0]
    assert rec["reference_answers"] == [" Berlin", "Äpfel"]
    assert rec["question"] == "  WOHER?  "
    assert [ev["answer_start"] for ev in rec["gold_evidences"]] == [5, 17]


def test_source_chunk_offset_is_used_with_another_representative_parent():
    result = build_qa_records(*inputs(raw(), "  Berlin  "))
    assert result["records"]["dev"][0]["gold_evidences"] == [evidence("fb38a93fd89e1a5f", "Berlin", 0)]


@pytest.mark.parametrize("source,representative,pid", [
    ("alpha Berlin omega  end", "alpha Berlin omega end", "7dbc214416c0931c"),
    ("alpha Berlin omega  Berlin end", "alpha Berlin omega Berlin end", "09f06d8cf4a66bda"),
])
def test_collision_with_valid_direct_slice_does_not_require_remap(source, representative, pid):
    result = build_qa_records(*inputs(raw(source, ["Berlin"], [6]), representative))
    assert result["records"]["dev"][0]["gold_evidences"] == [evidence(pid, "Berlin", 6)]
    assert result["ambiguous_offset_remap_report"] == []


@pytest.mark.parametrize("source,representative,answer,start,candidates,selected", [
    ("alpha Berlin omega", "alpha berlin omega", "Berlin", 6, [], None),
    ("alpha  Berlin omega", "alpha Berlin omega", "Berlin", 7, [6], 6),
    ("alpha  Berlin omega Berlin end", "alpha Berlin omega Berlin end", "Berlin", 20, [6, 19], 19),
    ("A  a     ", "a a", "a", 3, [0, 2], 0),
    ("a\u0308ää", "äää", "ää", 2, [0, 1], 1),
])
def test_collision_remap_zero_one_many_tie_and_overlapping(source, representative, answer, start, candidates, selected):
    args = inputs(raw(source, [answer], [start]), representative)
    result = build_qa_records(*args)
    pid = passage_id(representative)
    assert passage_id(source) == pid  # Fixture là dedup collision thật.
    rec = result["records"]["dev"][0]
    assert rec["reference_answers"] == [answer]
    assert rec["gold_evidences"] == ([] if selected is None else [evidence(pid, answer, selected)])
    report = result["ambiguous_offset_remap_report"]
    if len(candidates) == 1:
        assert report == []
    else:
        assert report == [{"query_id": "gq-train-1", "passage_id": pid, "answer_text": answer,
            "answer_index": 0, "source_answer_start": start, "source_context_char_len": len(source),
            "source_chunk_start_char": 0, "source_chunk_end_char": len(source.rstrip()),
            "representative_char_len": len(representative), "candidate_offsets": candidates,
            "selected_offset": selected}]


def test_zero_occurrences_drop_only_affected_evidence():
    result = build_qa_records(*inputs(raw("alpha Berlin omega", ["Berlin", "omega"], [6, 13]), "alpha berlin omega"))
    rec = result["records"]["dev"][0]
    assert rec["reference_answers"] == ["Berlin", "omega"]
    assert rec["gold_evidences"] == [evidence("1aeb9dce1a833b3a", "omega", 13)]
    assert result["coverage_report"]["files"]["qa_germanquad_dev.jsonl"]["coverage_rate"] == 1.0
    assert result["ambiguous_offset_remap_report"][0]["selected_offset"] is None


def test_remap_uses_relative_position_in_source_document_not_chunk():
    words = [f"w{i:03d}" for i in range(120)]
    words[85] = words[105] = "Berlin"
    context = " ".join(words)
    chunks = chunk_text(context)
    representative = chunks[1].text.replace("Berlin", " Berlin", 1)
    passages = corpus(chunks[0].text, representative)
    rows, test, decisions, _ = inputs(raw(context, ["Berlin"], [425]))
    result = build_qa_records(rows, test, decisions, passages)
    # Document: start=425, len=603. Chunk 1: start=400, len=203; rep len=204.
    # 128/204 closer to 425/603 than 26/204; local source 25/203 would select 26.
    assert len(context) == 603 and len(representative) == 204
    assert result["records"]["dev"][0]["gold_evidences"] == [
        evidence(passage_id(chunks[0].text), "Berlin", 425), evidence(passage_id(representative), "Berlin", 128)]
    assert result["ambiguous_offset_remap_report"] == [{
        "query_id": "gq-train-1", "passage_id": passage_id(representative), "answer_text": "Berlin",
        "answer_index": 0, "source_answer_start": 425, "source_context_char_len": 603,
        "source_chunk_start_char": 400, "source_chunk_end_char": 603,
        "representative_char_len": 204, "candidate_offsets": [26, 128], "selected_offset": 128}]


def test_duplicate_annotations_keep_separate_remap_log_entries():
    row = raw("alpha  Berlin omega Berlin end", ["Berlin", "Berlin"], [20, 20])
    result = build_qa_records(*inputs(row, "alpha Berlin omega Berlin end"))
    assert [e["answer_index"] for e in result["ambiguous_offset_remap_report"]] == [0, 1]
    assert len(result["records"]["dev"][0]["gold_evidences"]) == 1


def test_d1_decisions_preserve_order_ids_exclusions_and_original_test():
    train = [raw(raw_id=4, question="A"), raw(raw_id=3, question="B"), raw(raw_id=2, question="C")]
    test = [raw(raw_id=4, question="D")]
    decisions, _ = assign_splits(train, test, [{"question": "B"}], [{"question": "A"}], dev_ratio=1)
    result = build_qa_records(train, test, list(reversed(decisions)), corpus("Berlin"))
    assert [r["query_id"] for r in result["records"]["dev"]] == ["gq-train-3", "gq-train-2"]
    assert [r["query_id"] for r in result["records"]["test"]] == ["gq-test-4"]
    assert result["records"]["train"] == []
    assert result["coverage_report"]["input_rows"] == 4
    assert result["coverage_report"]["excluded_leakage_rows"] == 1


def test_leakage_exclusion_does_not_require_passage_or_remove_corpus():
    rows, test, decisions, passages = inputs()
    decisions[0]["source_split"] = None
    result = build_qa_records(rows, test, decisions, {})
    assert all(not records for records in result["records"].values())
    assert result["coverage_report"]["excluded_leakage_rows"] == 1
    assert passages == corpus("Berlin")


def test_coverage_per_file_uses_all_retained_rows_and_sorted_empty_ids():
    train = [raw("", [], [], raw_id=i, question=f"Q{i}") for i in (30, 2)] + [raw(raw_id=1)]
    test = [raw(raw_id=8, question="Test"), raw("", [], [], raw_id=9, question="Empty")]
    decisions, _ = assign_splits(train, test, [], [], dev_ratio=0)
    result = build_qa_records(train, test, decisions, corpus("Berlin"))
    assert result["coverage_report"] == {"input_rows": 5, "excluded_leakage_rows": 0, "files": {
        "qa_germanquad_train.jsonl": {"n_records": 3, "n_empty_gold": 2, "coverage_rate": 1/3,
                                     "empty_gold_query_ids": ["gq-train-2", "gq-train-30"]},
        "qa_germanquad_dev.jsonl": {"n_records": 0, "n_empty_gold": 0, "coverage_rate": None, "empty_gold_query_ids": []},
        "qa_germanquad_test.jsonl": {"n_records": 2, "n_empty_gold": 1, "coverage_rate": 0.5, "empty_gold_query_ids": ["gq-test-9"]}}}


def test_empty_input_has_no_division_by_zero():
    result = build_qa_records([], [], [], {})
    assert result["records"] == {"train": [], "dev": [], "test": []}
    assert result["coverage_report"]["input_rows"] == result["coverage_report"]["excluded_leakage_rows"] == 0
    assert all(v["coverage_rate"] is None for v in result["coverage_report"]["files"].values())
    assert result["ambiguous_offset_remap_report"] == []


@pytest.mark.parametrize("field,value,match", [
    ("id", True, "id"), ("id", -1, "id"), ("id", "1", "id"), ("id", None, "id"),
    ("question", " ", "question"), ("question", None, "question"),
    ("context", None, "context"), ("context", 12, "context"),
    ("answers", [], "answers"), ("answers", {}, "answers"),
    ("answers", {"text": ("Berlin",), "answer_start": [0]}, "answers"),
    ("answers", {"text": ["Berlin"], "answer_start": (0,)}, "answers"),
    ("answers", {"text": ["Berlin"], "answer_start": []}, "answers"),
    ("answers", {"text": [], "answer_start": [0]}, "answers"),
    ("answers", {"text": [""], "answer_start": [0]}, "answer"),
    ("answers", {"text": [" "], "answer_start": [0]}, "answer"),
    ("answers", {"text": [None], "answer_start": [0]}, "answer"),
    ("answers", {"text": ["Berlin"], "answer_start": [True]}, "answer_start"),
    ("answers", {"text": ["Berlin"], "answer_start": [-1]}, "answer_start"),
    ("answers", {"text": ["Berlin"], "answer_start": [0.0]}, "answer_start"),
    ("answers", {"text": ["Berlin"], "answer_start": [1]}, "slice"),
    ("answers", {"text": ["berlin"], "answer_start": [0]}, "slice"),
])
def test_invalid_raw_records_fail_before_mapping(field, value, match):
    rows, test, decisions, passages = inputs()
    rows[0][field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        build_qa_records(rows, test, decisions, passages)


def test_invalid_excluded_row_is_not_silently_skipped():
    rows, test, decisions, passages = inputs()
    decisions[0]["source_split"] = None
    rows[0]["answers"]["answer_start"] = [1]
    with pytest.raises(ValueError, match="slice"):
        build_qa_records(rows, test, decisions, passages)


@pytest.mark.parametrize("field,value,match", [
    ("source_dataset", "unknown", "source_dataset"), ("source_dataset", "germandpr", "decision"),
    ("query_id", "gq-train-99", "decision"), ("query_id", [], "query_id"),
    ("original_split", "test", "original_split"), ("source_row_index", 1, "source_row_index"),
    ("source_row_index", False, "source_row_index"), ("source_row_index", 0.0, "source_row_index"),
    ("question_key", "FRAGE A", "question_key"), ("source_split", "test", "source_split"),
    ("source_split", "other", "source_split"), ("source_split", [], "source_split"),
])
def test_split_identity_must_match_raw(field, value, match):
    rows, test, decisions, passages = inputs()
    decisions[0][field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        build_qa_records(rows, test, decisions, passages)


@pytest.mark.parametrize("assigned", [None, "train", "dev"])
def test_original_test_cannot_be_removed_or_reassigned(assigned):
    test = [raw()]
    decisions, _ = assign_splits([], test, [], [])
    decisions[0]["source_split"] = assigned
    with pytest.raises(ValueError, match="source_split"):
        build_qa_records([], test, decisions, corpus("Berlin"))


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "missing_field", "non_mapping"])
def test_decisions_are_one_to_one(change):
    rows, test, decisions, passages = inputs()
    if change == "missing":
        decisions.clear()
    elif change == "duplicate":
        decisions.append(deepcopy(decisions[0]))
    elif change == "extra":
        decisions.append({**decisions[0], "query_id": "gq-train-99"})
    elif change == "missing_field":
        del decisions[0]["source_split"]
    else:
        decisions[0] = "not a decision"
    with pytest.raises((TypeError, ValueError), match="decision"):
        build_qa_records(rows, test, decisions, passages)


def test_duplicate_raw_id_is_rejected():
    rows, test, decisions, passages = inputs()
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="query_id"):
        build_qa_records(rows, test, decisions, passages)


@pytest.mark.parametrize("index,value", [(0, "old inputs"), (1, {}), (2, {}), (3, []), (0, ["not a row"])])
def test_invalid_input_containers(index, value):
    args = list(inputs())
    args[index] = value
    with pytest.raises((TypeError, ValueError)):
        build_qa_records(*args)


@pytest.mark.parametrize("change", ["missing", "wrong_key", "wrong_text", "missing_field", "bad_offsets", "non_mapping"])
def test_missing_or_corrupt_representative_fails(change):
    rows, test, decisions, passages = inputs()
    pid = passage_id("Berlin")
    if change == "missing":
        passages.clear()
    elif change == "wrong_key":
        passages[pid] = next(iter(corpus("Ulm").values()))
    elif change == "wrong_text":
        passages[pid]["text"] = "Paris!"
    elif change == "missing_field":
        del passages[pid]["title"]
    elif change == "bad_offsets":
        passages[pid]["chunk_start_char"] = True
    else:
        passages[pid] = None
    with pytest.raises((TypeError, ValueError), match="passage|Passage"):
        build_qa_records(rows, test, decisions, passages)


def test_inputs_are_not_mutated_results_validate_and_repeat():
    args = inputs(raw("alpha  Berlin omega Berlin end", ["Berlin"], [20]), "alpha Berlin omega Berlin end")
    before = deepcopy(args)
    result = build_qa_records(*args)
    assert args == before
    assert build_qa_records(*args) == result
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result
    texts = {pid: p["text"] for pid, p in args[3].items()}
    for records in result["records"].values():
        assert all(validate_qa_record(r, passage_texts=texts) == [] for r in records)
    result["records"]["dev"][0]["reference_answers"].append("new")
    assert args == before
