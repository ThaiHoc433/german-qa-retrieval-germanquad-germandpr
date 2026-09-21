import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.splits import assign_splits, question_key


def six_rows():
    """Fixture minh họa từ thẻ T5.2, không phải câu hỏi dataset thật."""
    return (
        [{"id": 51870, "question": "Frage A"},
         {"id": 101, "question": "Frage B"},
         {"id": 102, "question": "Frage C"}],
        [{"id": 40369, "question": "Frage D"}],
        [{"question": "Frage B"}],
        [{"question": "Frage A"}],
    )


def test_cross_dataset_test_key_excludes_train_row_only():
    rows = six_rows()
    before = deepcopy(rows)
    decisions, _ = assign_splits(*rows)
    assert decisions == [
        {"query_id": "gq-train-51870", "source_dataset": "germanquad", "original_split": "train",
         "source_row_index": 0, "question_key": "frage a", "source_split": None},
        {"query_id": "gq-train-101", "source_dataset": "germanquad", "original_split": "train",
         "source_row_index": 1, "question_key": "frage b", "source_split": "train"},
        {"query_id": "gq-train-102", "source_dataset": "germanquad", "original_split": "train",
         "source_row_index": 2, "question_key": "frage c", "source_split": "train"},
        {"query_id": "gq-test-40369", "source_dataset": "germanquad", "original_split": "test",
         "source_row_index": 0, "question_key": "frage d", "source_split": "test"},
        {"query_id": "gdpr-train-0", "source_dataset": "germandpr", "original_split": "train",
         "source_row_index": 0, "question_key": "frage b", "source_split": "train"},
        {"query_id": "gdpr-test-0", "source_dataset": "germandpr", "original_split": "test",
         "source_row_index": 0, "question_key": "frage a", "source_split": "test"},
    ]
    assert rows == before


def test_report_counts_keys_and_rows_per_dataset():
    _, report = assign_splits(*six_rows())
    assert report == {
        "seed": 42, "dev_ratio": 0.10,
        "key_counts": {"test_union": 2, "train_dev_pool": 2, "train": 2, "dev": 0},
        "datasets": {
            "germanquad": {
                "input": {"train": {"rows": 3, "keys": 3}, "test": {"rows": 1, "keys": 1}},
                "assigned": {"train": {"rows": 2, "keys": 2}, "dev": {"rows": 0, "keys": 0},
                             "test": {"rows": 1, "keys": 1}},
                "removed": {"rows": 1, "keys": 1, "query_ids": ["gq-train-51870"]},
            },
            "germandpr": {
                "input": {"train": {"rows": 1, "keys": 1}, "test": {"rows": 1, "keys": 1}},
                "assigned": {"train": {"rows": 1, "keys": 1}, "dev": {"rows": 0, "keys": 0},
                             "test": {"rows": 1, "keys": 1}},
                "removed": {"rows": 0, "keys": 0, "query_ids": []},
            },
        },
    }


def test_dev_assignment_preserves_original_query_ids():
    rows = six_rows()
    rows[2].insert(0, {"question": "Frage A"})  # Excluded before the surviving DPR row.
    decisions, report = assign_splits(*rows, dev_ratio=1)
    assert [(d["query_id"], d["source_split"]) for d in decisions] == [
        ("gq-train-51870", None), ("gq-train-101", "dev"), ("gq-train-102", "dev"),
        ("gq-test-40369", "test"), ("gdpr-train-0", None), ("gdpr-train-1", "dev"),
        ("gdpr-test-0", "test"),
    ]
    assert report["key_counts"] == {"test_union": 2, "train_dev_pool": 2, "train": 0, "dev": 2}


def test_same_dataset_test_overlap_excludes_dpr_train():
    rows = six_rows()
    rows[2].append({"question": "Frage A"})
    decisions, report = assign_splits(*rows)
    by_id = {d["query_id"]: d for d in decisions}
    assert by_id["gdpr-train-1"]["source_split"] is None
    assert by_id["gdpr-test-0"]["source_split"] == "test"
    assert report["datasets"]["germandpr"]["removed"] == {
        "rows": 1, "keys": 1, "query_ids": ["gdpr-train-1"],
    }


def test_gq_test_also_protects_both_datasets_and_keeps_duplicate_test_rows():
    decisions, report = assign_splits(
        [{"id": 1, "question": "A"}],
        [{"id": 1, "question": "A"}, {"id": 2, "question": " A "}],
        [{"question": "a"}], [],
    )
    assert [(d["query_id"], d["source_split"]) for d in decisions] == [
        ("gq-train-1", None), ("gq-test-1", "test"), ("gq-test-2", "test"), ("gdpr-train-0", None),
    ]
    assert report["datasets"]["germanquad"]["assigned"]["test"] == {"rows": 2, "keys": 1}
    assert report["key_counts"] == {"test_union": 1, "train_dev_pool": 0, "train": 0, "dev": 0}


def test_duplicate_questions_keep_rows_and_share_split():
    rows = six_rows()
    rows[0].append({"id": 103, "question": "  FRAGE B "})
    decisions, report = assign_splits(*rows)
    shared = [d for d in decisions if d["question_key"] == "frage b"]
    assert [d["query_id"] for d in shared] == ["gq-train-101", "gq-train-103", "gdpr-train-0"]
    assert [d["source_split"] for d in shared] == ["train", "train", "train"]
    assert report["datasets"]["germanquad"]["assigned"]["train"] == {"rows": 3, "keys": 2}
    assert report["key_counts"]["train_dev_pool"] == 2


def test_question_key_preserves_internal_whitespace():
    assert question_key("Frage  A") == "frage  a"
    assert question_key("Frage A") == "frage a"
    assert question_key(" Frage\tA ") == "frage\ta"


def test_question_key_normalizes_unicode_case_and_edges():
    assert question_key(" A\u0308  B \t") == "ä  b"
    assert question_key(" Ä Ö Ü ß ") == "ä ö ü ß"


def test_preserved_whitespace_changes_leakage_decision():
    decisions, _ = assign_splits(
        [{"id": 1, "question": "Frage  A"}], [], [], [{"question": "Frage A"}],
    )
    assert [d["source_split"] for d in decisions] == ["train", "test"]


def test_sorted_pool_sampling_is_repeatable_without_changing_global_rng():
    rows = [{"id": i, "question": f"Frage {i:02d}"} for i in range(10)]
    rng_before = random.getstate()
    first = assign_splits(rows, [], [], [])
    assert first == assign_splits(rows, [], [], [])
    assert random.getstate() == rng_before
    decisions, report = first
    assert [d["query_id"] for d in decisions if d["source_split"] == "dev"] == ["gq-train-1"]
    assert report["key_counts"] == {"test_union": 0, "train_dev_pool": 10, "train": 9, "dev": 1}
    reversed_decisions, reversed_report = assign_splits(list(reversed(rows)), [], [], [])
    assert {d["question_key"]: d["source_split"] for d in reversed_decisions} == {
        d["question_key"]: d["source_split"] for d in decisions
    }
    assert reversed_report == report
    changed_seed, _ = assign_splits(rows, [], [], [], seed=0)
    assert [d["query_id"] for d in changed_seed if d["source_split"] == "dev"] == ["gq-train-6"]


def test_dev_size_rounds_down_and_samples_keys_not_rows():
    rows = [{"id": i, "question": f"Frage {i:02d}"} for i in range(19)]
    rows += [{"id": 100 + i, "question": "Frage 03"} for i in range(30)]
    decisions, report = assign_splits(rows, [], [], [])
    assert report["key_counts"] == {"test_union": 0, "train_dev_pool": 19, "train": 18, "dev": 1}
    assert {d["question_key"] for d in decisions if d["source_split"] == "dev"} == {"frage 03"}
    assert report["datasets"]["germanquad"]["assigned"]["dev"] == {"rows": 31, "keys": 1}


def test_report_conserves_rows_and_lists_all_excluded_ids():
    # More than ten excluded IDs, with duplicate keys and an unrelated row with no answers.
    train = [{"id": i, "question": "A", "answers": {"text": [], "answer_start": []}} for i in range(12)]
    train += [{"id": 99, "question": "B", "answers": {"text": [], "answer_start": []}}]
    decisions, report = assign_splits(train, [], [], [{"question": "A"}])
    assert len(decisions) == 14
    assert decisions[12]["query_id"] == "gq-train-99"
    assert decisions[12]["source_split"] == "train"
    assert report["datasets"]["germanquad"]["removed"] == {
        "rows": 12, "keys": 1,
        "query_ids": ["gq-train-0", "gq-train-1", "gq-train-10", "gq-train-11", "gq-train-2",
                      "gq-train-3", "gq-train-4", "gq-train-5", "gq-train-6", "gq-train-7",
                      "gq-train-8", "gq-train-9"],
    }
    for ds in report["datasets"].values():
        assert sum(s["rows"] for s in ds["assigned"].values()) + ds["removed"]["rows"] == sum(
            s["rows"] for s in ds["input"].values()
        )
    assert json.loads(json.dumps([decisions, report])) == [decisions, report]


def test_empty_inputs_and_no_train_pool():
    decisions, report = assign_splits([], [], [], [])
    assert decisions == []
    assert report["key_counts"] == {"test_union": 0, "train_dev_pool": 0, "train": 0, "dev": 0}
    for dataset in ("germanquad", "germandpr"):
        assert report["datasets"][dataset] == {
            "input": {"train": {"rows": 0, "keys": 0}, "test": {"rows": 0, "keys": 0}},
            "assigned": {"train": {"rows": 0, "keys": 0}, "dev": {"rows": 0, "keys": 0},
                         "test": {"rows": 0, "keys": 0}},
            "removed": {"rows": 0, "keys": 0, "query_ids": []},
        }
    only_test, _ = assign_splits([], [], [], [{"question": "A"}, {"question": "A"}])
    assert [d["source_split"] for d in only_test] == ["test", "test"]


@pytest.mark.parametrize("ratio", [0, 0.25, 1])
def test_valid_ratio_is_used_and_input_tuples_are_supported(ratio):
    rows = tuple({"id": i, "question": str(i)} for i in range(4))
    _, report = assign_splits(rows, (), (), (), dev_ratio=ratio)
    assert report["dev_ratio"] == ratio
    assert report["key_counts"]["dev"] == {0: 0, 0.25: 1, 1: 4}[ratio]


@pytest.mark.parametrize("ratio", [True, False, -0.1, 1.1, float("nan"), float("inf"), "0.1", None])
def test_invalid_dev_ratio_is_rejected(ratio):
    with pytest.raises((TypeError, ValueError), match="dev_ratio"):
        assign_splits(*six_rows(), dev_ratio=ratio)


@pytest.mark.parametrize("seed", [True, 1.5, "42", None])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises((TypeError, ValueError), match="seed"):
        assign_splits(*six_rows(), seed=seed)


@pytest.mark.parametrize("slot,rows,field", [
    (0, ["Frage A"], "row"), (1, [None], "row"),
    (2, [{}], "question"), (3, [{"question": " \t "}], "question"),
    (0, [{"id": 1, "question": 123}], "question"),
    (0, [{"question": "A"}], "id"), (1, [{"id": True, "question": "A"}], "id"),
    (0, [{"id": -1, "question": "A"}], "id"), (0, [{"id": "1", "question": "A"}], "id"),
    (0, [{"id": 1.0, "question": "A"}], "id"),
    (0, [{"id": 1, "question": "A"}, {"id": 1, "question": "B"}], "query_id"),
    (0, None, "rows"), (2, "A", "rows"), (3, {}, "rows"),
])
def test_invalid_input_fails_without_mutating_rows(slot, rows, field):
    inputs = list(six_rows())
    inputs[slot] = rows
    before = deepcopy(inputs)
    with pytest.raises((TypeError, ValueError), match=field):
        assign_splits(*inputs)
    assert inputs == before


def test_result_mutation_does_not_change_input_or_later_calls():
    inputs = six_rows()
    expected = assign_splits(*inputs)
    decisions, report = assign_splits(*inputs)
    decisions[0]["source_split"] = "train"
    report["datasets"]["germanquad"]["removed"]["query_ids"].clear()
    assert assign_splits(*inputs) == expected
