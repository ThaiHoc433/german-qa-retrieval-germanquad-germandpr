import sys
from itertools import permutations
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import CorpusBuilder, chunk_text, find_span_chunks, normalize_text, passage_id
from src.schemas import validate_passage_record


BERLIN_ID = "fb38a93fd89e1a5f"
PASSAGE_FIELDS = {
    "passage_id", "text", "normalized_text", "source", "title", "char_len",
    "parent_document_id", "chunk_start_char", "chunk_end_char",
}


# Fixture minh họa; expected char-offset được tính trước khi sửa CorpusBuilder.
def words(n):
    return " ".join(f"w{i:03d}" for i in range(n))


def test_normalization_preserves_german_letters_and_raw_input():
    raw = "  A\u0308\tÖ\nÜ   ß  "
    assert normalize_text(raw) == "ä ö ü ß"
    assert raw == "  A\u0308\tÖ\nÜ   ß  "
    assert passage_id("  BERLIN  ") == BERLIN_ID
    assert passage_id("Ulm") == "37bfc8ba5ec56136"


@pytest.mark.parametrize("text,count", [
    ("", 0), (" \t\n ", 0), (words(1), 1), (words(100), 1),
    (words(101), 2), (words(201), 3),
])
def test_chunk_boundaries_do_not_create_extra_tail(text, count):
    chunks = chunk_text(text)
    assert len(chunks) == count
    assert all(c.text == text[c.char_start:c.char_end] for c in chunks)


def test_201_word_offsets_and_answer_in_overlap():
    text = words(201)
    assert len(text) == 1004
    chunks = chunk_text(text)
    assert [(c.char_start, c.char_end) for c in chunks] == [(0, 499), (400, 899), (800, 1004)]
    assert [len(c.text) for c in chunks] == [499, 499, 204]
    assert text[425:429] == "w085"
    indices = find_span_chunks(chunks, 425, 429)
    assert indices == [0, 1]
    assert [(425 - chunks[i].char_start, 429 - chunks[i].char_start) for i in indices] == [(425, 429), (25, 29)]
    assert all(c.text == text[c.char_start:c.char_end] for c in chunks)
    # Span starts before chunk 2 and ends after chunk 1: never stretch a window.
    assert find_span_chunks(chunks, 395, 505) == []


def test_chunk_keeps_internal_whitespace_and_original_offsets():
    chunks = chunk_text("  Berlin  ")
    assert [(c.text, c.char_start, c.char_end) for c in chunks] == [("Berlin", 2, 8)]
    assert chunk_text("  Berlin\t ist\n schön! ")[0].text == "Berlin\t ist\n schön!"


def test_builder_emits_nine_fields_with_first_line_title_and_parent_provenance():
    builder = CorpusBuilder()
    parent = "Berlin\n Eine Stadt."
    chunks = builder.add_text(parent, "germanquad")
    rec, = builder.records()
    assert set(rec) == PASSAGE_FIELDS
    assert rec["title"] == "Berlin"
    assert rec["text"] == parent
    assert rec["source"] == "germanquad"
    assert rec["normalized_text"] == "berlin eine stadt."
    assert rec["char_len"] == len(parent)
    assert (rec["chunk_start_char"], rec["chunk_end_char"]) == (0, len(parent))
    assert chunks == chunk_text(parent)
    assert validate_passage_record(rec, parent_text=parent) == []


def test_each_chunk_keeps_full_parent_identity():
    parent = words(201)
    builder = CorpusBuilder()
    builder.add_text(parent, "germandpr_positive")
    records = builder.records()
    assert len(records) == 3
    assert {r["parent_document_id"] for r in records} == {"2708af420817cfc6"}
    assert all(r["passage_id"] != r["parent_document_id"] for r in records)
    assert [(r["chunk_start_char"], r["chunk_end_char"]) for r in records] == [(0, 499), (400, 899), (800, 1004)]
    assert all(validate_passage_record(r, parent_text=parent) == [] for r in records)


@pytest.mark.parametrize("source", ["germandpr_positive", "wikipedia_distractor"])
def test_source_title_is_retained_and_not_inferred_when_missing(source):
    builder = CorpusBuilder()
    builder.add_text("Berlin", source, title="Berlin_Artikel")
    builder.add_text("Ulm", source)
    records = {r["text"]: r for r in builder.records()}
    assert records["Berlin"]["title"] == "Berlin_Artikel"
    assert records["Ulm"]["title"] is None


@pytest.mark.parametrize("title", ["", " \t "])
def test_empty_source_title_is_none(title):
    builder = CorpusBuilder()
    builder.add_text("Berlin", "wikipedia_distractor", title=title)
    assert builder.records()[0]["title"] is None


def test_germanquad_empty_first_line_does_not_invent_title_from_second_line():
    builder = CorpusBuilder()
    builder.add_text("\nBerlin", "germanquad")
    rec, = builder.records()
    assert rec["title"] is None
    assert (rec["chunk_start_char"], rec["chunk_end_char"]) == (1, 7)


@pytest.mark.parametrize("source", ["germandpr_hard_negative", "unknown"])
@pytest.mark.parametrize("text", ["Ulm", ""])
def test_builder_rejects_disallowed_sources_before_mutating_corpus(source, text):
    builder = CorpusBuilder()
    builder.add_text("Berlin", "germanquad")
    before = builder.records()
    with pytest.raises(ValueError, match="source"):
        builder.add_text(text, source)
    assert builder.records() == before


def test_allowed_empty_context_has_no_passage():
    builder = CorpusBuilder()
    assert builder.add_text(" \t\n ", "germanquad") == []
    assert builder.records() == []


def test_higher_priority_replaces_all_metadata_together():
    builder = CorpusBuilder()
    builder.add_text("  BERLIN  ", "germandpr_positive", title="DPR title")
    builder.add_text("Berlin", "germanquad", title="GQ title")
    rec, = builder.records()
    assert rec == {
        "passage_id": BERLIN_ID, "text": "Berlin", "normalized_text": "berlin",
        "source": "germanquad", "title": "GQ title", "char_len": 6,
        "parent_document_id": BERLIN_ID, "chunk_start_char": 0, "chunk_end_char": 6,
    }


def test_dedup_replacement_updates_char_length_when_raw_whitespace_differs():
    builder = CorpusBuilder()
    builder.add_text("  BERLIN  ist gut  ", "germandpr_positive", title="DPR")
    builder.add_text("Berlin ist gut", "germanquad", title="GQ")
    rec, = builder.records()
    assert rec["text"] == "Berlin ist gut"
    assert rec["char_len"] == 14
    assert (rec["chunk_start_char"], rec["chunk_end_char"]) == (0, 14)
    assert rec["title"] == "GQ"
    assert validate_passage_record(rec, parent_text="Berlin ist gut") == []


def test_replacement_changes_parent_identity_not_just_chunk_identity():
    builder = CorpusBuilder()
    long_parent = words(101)
    builder.add_text(long_parent, "germandpr_positive", title="DPR")
    first_chunk = words(100)
    builder.add_text(first_chunk, "germanquad", title="GQ")
    records = builder.records()
    rec = next(r for r in records if r["text"] == first_chunk)
    assert rec["parent_document_id"] == rec["passage_id"]
    assert rec["title"] == "GQ"
    assert validate_passage_record(rec, parent_text=first_chunk) == []
    other = next(r for r in records if r["text"] != first_chunk)
    assert other["title"] == "DPR"
    assert validate_passage_record(other, parent_text=long_parent) == []


def test_equal_priority_preserves_first_occurrence_and_all_its_metadata():
    builder = CorpusBuilder()
    builder.add_text("  BERLIN  ", "germanquad", title="First")
    before = builder.records()
    builder.add_text("Berlin", "germanquad", title="Second")
    assert builder.records() == before
    rec, = builder.records()
    assert rec["text"] == "BERLIN"
    assert rec["title"] == "First"
    assert (rec["chunk_start_char"], rec["chunk_end_char"]) == (2, 8)


@pytest.mark.parametrize("order", list(permutations([
    "germanquad", "germandpr_positive", "wikipedia_distractor",
])))
def test_source_priority_is_independent_of_order_between_sources(order):
    builder = CorpusBuilder()
    for source in order:
        builder.add_text("Berlin", source, title=source)
    rec, = builder.records()
    assert rec["source"] == "germanquad"
    assert rec["title"] == "germanquad"


def test_dpr_positive_replaces_wikipedia_but_not_the_other_way_around():
    builder = CorpusBuilder()
    builder.add_text("BERLIN", "wikipedia_distractor", title="Wiki")
    builder.add_text("Berlin", "germandpr_positive", title="DPR")
    before = builder.records()
    builder.add_text("  berlin  ", "wikipedia_distractor", title="New wiki")
    assert builder.records() == before
    assert before[0]["source"] == "germandpr_positive"
    assert before[0]["text"] == "Berlin"
    assert before[0]["title"] == "DPR"


def test_records_are_serializable_copies_without_internal_source_sets():
    builder = CorpusBuilder()
    builder.add_text("Berlin", "germanquad")
    rec, = builder.records()
    assert set(rec) == PASSAGE_FIELDS
    rec["text"] = "changed externally"
    assert builder.records()[0]["text"] == "Berlin"
