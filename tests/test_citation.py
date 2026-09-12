import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.citation import PassageInput, find_quote_in_passage, find_short_answer_offsets, is_short_answer_supported, verify_citation
from src.answer_parser import ParsedAnswer


def test_exact_substring_match():
    text = "Berlin ist die Hauptstadt Deutschlands. Es liegt an der Spree."
    match = find_quote_in_passage(text, "Es liegt an der Spree.")
    assert match is not None
    assert match.quote_match_method == "exact"
    assert text[match.quote_start:match.quote_end] == "Es liegt an der Spree."
    assert match.quote_occurrence_count == 1


def test_no_match_returns_none():
    assert find_quote_in_passage("Berlin ist die Hauptstadt.", "Paris ist die Hauptstadt.") is None


def test_multiple_occurrences_counts_and_takes_first():
    text = "Die Katze schlaeft. Die Katze spielt. Die Katze schlaeft wieder."
    match = find_quote_in_passage(text, "Die Katze schlaeft")
    assert match is not None
    assert match.quote_occurrence_count == 2
    assert match.quote_start == 0


def test_normalized_match_handles_extra_whitespace():
    text = "Der Fluss   Spree fliesst durch Berlin."
    match = find_quote_in_passage(text, "Der Fluss Spree fliesst durch Berlin.")
    assert match is not None
    assert match.quote_match_method == "normalized"
    assert match.quote_start == 0
    assert match.quote_end == len(text)


def test_normalized_match_does_not_fuzzy_match_different_words():
    text = "Der Fluss Spree fliesst durch Berlin."
    assert find_quote_in_passage(text, "Der Fluss Spree fliesst durch Hamburg.") is None


def test_short_answer_supported_exact_substring():
    quote = "Albert Einstein entwickelte die Relativitaetstheorie."
    assert is_short_answer_supported(quote, "Albert Einstein") is True
    offsets = find_short_answer_offsets(quote, "Albert Einstein")
    assert offsets is not None
    assert quote[offsets.quote_start:offsets.quote_end] == "Albert Einstein"


def test_short_answer_not_in_quote():
    quote = "Marie Curie erhielt den Nobelpreis."
    assert is_short_answer_supported(quote, "Albert Einstein") is False
    assert find_short_answer_offsets(quote, "Albert Einstein") is None


def test_short_answer_supported_uses_em_f1_style_normalization():
    # normalize_answer (NFC + lowercase + xoa dau cau) -> "1300." va "1300" duoc coi la
    # ho tro, khac voi cach chuan hoa NFC+khoang-trang dung khi tim quote trong passage.
    quote = "Es gibt etwa 1300 ethnische Gruppen."
    assert is_short_answer_supported(quote, "1300.") is True


P1 = PassageInput("p1", "Titel A", "Albert Einstein wurde 1879 geboren. Er entwickelte die Relativitaetstheorie.")
P2 = PassageInput("p2", "Titel B", "Marie Curie erhielt zwei Nobelpreise.")


def _parsed(**overrides):
    base = dict(
        format_violation=False, answered=True, short_answer="Albert Einstein",
        answer="Albert Einstein entwickelte die Relativitaetstheorie.",
        # NOTE: brief's original fixture had quote="Er entwickelte die Relativitaetstheorie."
        # (only the 2nd sentence, referring to Einstein via pronoun "Er"), which never
        # contains the literal string "Albert Einstein" -- inconsistent with this test's
        # own assertion that answer_supported is True, since is_short_answer_supported is
        # pure literal-substring matching (no coreference resolution, by design). Fixed to
        # the full P1 text so the fixture is self-consistent; see task-4-report.md.
        quelle_refs=("P1",), quote=P1.text,
    )
    base.update(overrides)
    return ParsedAnswer(**base)


def test_verified_and_supported():
    outcome = verify_citation([P1, P2], _parsed())
    assert outcome.citation_status == "verified"
    assert outcome.answer_supported is True
    assert len(outcome.citations) == 1
    assert outcome.citations[0].passage_id == "p1"


def test_verified_but_not_supported():
    outcome = verify_citation([P1, P2], _parsed(short_answer="Marie Curie"))
    assert outcome.citation_status == "verified"
    assert outcome.answer_supported is False


def test_wrong_passage():
    # quote thật ra nằm ở P2, nhưng model ghi QUELLE: P1
    outcome = verify_citation([P1, P2], _parsed(quote="Marie Curie erhielt zwei Nobelpreise."))
    assert outcome.citation_status == "wrong_passage"
    assert outcome.answer_supported is None


def test_quote_not_found_anywhere():
    outcome = verify_citation([P1, P2], _parsed(quote="Das steht nirgendwo im Text."))
    assert outcome.citation_status == "quote_not_found"


def test_out_of_range_quelle():
    outcome = verify_citation([P1, P2], _parsed(quelle_refs=("P9",)))
    assert outcome.citation_status == "out_of_range"


def test_missing_citation():
    outcome = verify_citation([P1, P2], _parsed(quelle_refs=(), quote=None))
    assert outcome.citation_status == "missing"
    assert outcome.answer_supported is None


def test_multiple_sources_keeps_raw_refs_without_choosing():
    outcome = verify_citation([P1, P2], _parsed(quelle_refs=("P1", "P2")))
    assert outcome.citation_status == "multiple_sources"
    assert outcome.citations == []
    assert outcome.raw_citation_refs == ("P1", "P2")
