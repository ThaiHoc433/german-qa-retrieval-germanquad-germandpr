import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.citation import find_quote_in_passage


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
