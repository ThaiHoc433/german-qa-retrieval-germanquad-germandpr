from germanqa.data.split import (UnionFind, make_groups, remove_conflicts_with_test,
                            split_train_dev)


def test_unionfind_bac_cau():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("a") is not None
    assert uf.find("x") != uf.find("a")


def _rows(n_groups, per_group):
    """Mỗi nhóm dùng chung một đoạn: n_groups nhóm, mỗi nhóm per_group câu hỏi."""
    rows, qrels, doc_of = [], {}, {}
    for g in range(n_groups):
        pid = f"p{g}"
        doc_of[pid] = f"d{g}"
        for i in range(per_group):
            qid = f"q{g}_{i}"
            rows.append({"question_id": qid, "passage_id": pid})
            qrels[qid] = {pid}
    return rows, qrels, doc_of


def test_chia_theo_nhom_khong_roi_ri():
    rows, qrels, doc_of = _rows(50, 4)
    groups = make_groups(rows, qrels, doc_of)
    train, dev = split_train_dev(rows, groups, 0.1, 42)
    assert len(train) + len(dev) == len(rows)
    assert {r["passage_id"] for r in train}.isdisjoint({r["passage_id"] for r in dev})
    assert abs(len(dev) - 20) <= 4                         # xấp xỉ 10% của 200


def test_chia_tai_lap_duoc():
    rows, qrels, doc_of = _rows(30, 3)
    groups = make_groups(rows, qrels, doc_of)
    a = split_train_dev(rows, groups, 0.1, 42)
    b = split_train_dev(list(reversed(rows)), groups, 0.1, 42)
    assert sorted(r["question_id"] for r in a[1]) == sorted(r["question_id"] for r in b[1])


def test_chung_bai_viet_cung_nhom():
    rows = [{"question_id": "q1", "passage_id": "p1"}, {"question_id": "q2", "passage_id": "p2"}]
    qrels = {"q1": {"p1"}, "q2": {"p2"}}
    doc_of = {"p1": "dX", "p2": "dX"}                      # hai đoạn cùng một bài
    groups = make_groups(rows, qrels, doc_of)
    assert groups["q1"] == groups["q2"]


def test_remove_conflicts_with_test():
    train = [{"question_id": "q1", "passage_id": "p1"},
             {"question_id": "q2", "passage_id": "p9"},
             {"question_id": "q3", "passage_id": "p3"}]
    test = [{"question_id": "q1", "passage_id": "p5"}, {"question_id": "q8", "passage_id": "p9"}]
    test_qrels = {"q1": {"p5"}, "q8": {"p9"}}
    kept, removed = remove_conflicts_with_test(train, test, test_qrels)
    assert [r["question_id"] for r in kept] == ["q3"]
    assert {(r["question_id"], r["reason"]) for r in removed} == {
        ("q1", "QUESTION_IN_TEST"), ("q2", "PASSAGE_IN_TEST")}