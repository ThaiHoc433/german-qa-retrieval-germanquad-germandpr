import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.citation import find_quote_in_passage, find_short_answer_offsets, is_short_answer_supported


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
