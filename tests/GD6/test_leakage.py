from germanqa.evaluation.leakage import normalize_question, seen_question_flags, summarize, unseen_ids

QS = [{"question_id": "a", "question": "Wer  war Adenauer?"}, {"question_id": "b", "question": "Wo liegt Bonn?"},
      {"question_id": "c", "question": "Was ist Gott?"}]
SEEN = {"dpr_train": {"wer war adenauer?"}, "dpr_test": {"wo liegt bonn?"}}


def test_chuan_hoa_khoang_trang_va_hoa_thuong():
    assert normalize_question("  Wer\n war   ADENAUER? ") == "wer war adenauer?"


def test_co_va_tong_hop():
    f = seen_question_flags(QS, SEEN)
    assert f["a"] == {"dpr_train": True, "dpr_test": False} and f["c"] == {"dpr_train": False, "dpr_test": False}
    assert summarize(f) == {"questions": 3, "in_dpr_test": 1, "in_dpr_train": 1}


def test_unseen_chi_loai_tap_huan_luyen():
    f = seen_question_flags(QS, SEEN)
    assert unseen_ids(f) == ["b", "c"]                       # câu b chỉ nằm trong DPR test: encoder chưa học
    assert unseen_ids(f, ("dpr_train", "dpr_test")) == ["c"]
