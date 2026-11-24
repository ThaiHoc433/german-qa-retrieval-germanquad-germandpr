from germanqa.evaluation.run_cache import per_question, to_run, validate_rows
from germanqa.evaluation.select_set import build_select_set

CORPUS = {"p1", "p2", "p3"}


def row(qid, hits):
    return {"question_id": qid, "hits": hits}


def test_hop_le_thi_khong_loi():
    assert validate_rows([row("q1", [["p1", 3.0], ["p2", 2.0], ["p3", 2.0]])], 3, CORPUS, {"q1"}) == []


def test_bat_cac_loi_chinh():
    assert any("hai lần" in p for p in validate_rows([row("q1", []), row("q1", [])], 3, CORPUS))
    assert any("không thuộc tập" in p for p in validate_rows([row("qT", [])], 3, CORPUS, {"q1"}))
    assert any("> k" in p for p in validate_rows([row("q1", [["p1", 3], ["p2", 2]])], 1, CORPUS))
    assert any("trùng" in p for p in validate_rows([row("q1", [["p1", 3], ["p1", 2]])], 3, CORPUS))
    assert any("không có trong corpus" in p for p in validate_rows([row("q1", [["zz", 3]])], 3, CORPUS))
    assert any("sai thứ tự" in p for p in validate_rows([row("q1", [["p1", 1], ["p2", 2]])], 3, CORPUS))


def test_per_question_giu_thu_tu_qids():
    run = to_run([row("a", [["p1", 2], ["p2", 1]]), row("b", [["p2", 2], ["p1", 1]])])
    pq = per_question(run, {"a": ["p1"], "b": ["p1"]}, ["b", "a"], ks=(1,))
    assert pq["hit@1"] == [0.0, 1.0]


def test_tap_chon_loai_cau_da_thay():
    qa = {"train": [{"question_id": "t1", "question": "Was ist  das?"}, {"question_id": "t2", "question": "Neu?"}],
          "dev": [{"question_id": "d1", "question": "neu?"}]}
    out = build_select_set(qa, ["was ist das?"])
    assert out["train"] == ["t2"] and out["dev"] == ["d1"] and out["counts"]["total"] == 2
