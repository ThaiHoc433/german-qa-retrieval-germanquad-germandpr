import pytest

from germanqa.retrieval.rrf import rrf


def test_tay_ID_mot_ben_hai_ben_va_dong_diem():
    # c=1: điểm = 1/(1+hạng). a: bm25 hạng 1, dense hạng 3 ; b: bm25 hạng 2 ; c: dense hạng 1 ; d: bm25 hạng 3
    out = rrf({"bm25": ["a", "b", "d"], "dense": ["c", "x", "a"]}, c=1)
    sc = {r["passage_id"]: r["score"] for r in out}
    assert sc["a"] == pytest.approx(1 / 2 + 1 / 4)          # hai bên
    assert sc["b"] == pytest.approx(1 / 3)                   # chỉ bm25
    assert sc["c"] == pytest.approx(1 / 2)                   # chỉ dense hạng 1
    assert sc["d"] == pytest.approx(1 / 4)
    assert sc["x"] == pytest.approx(1 / 3)                   # đồng điểm với b
    ids = [r["passage_id"] for r in out]
    assert ids == ["a", "c", "b", "x", "d"]                  # b và x đồng điểm: hạng tốt nhất b=2 < x=2? cả hai 2 -> theo passage_id
    assert [r["rank"] for r in out] == [1, 2, 3, 4, 5]


def test_khong_trung_ID_va_nguon():
    out = rrf({"bm25": ["a", "b"], "dense": ["b", "a"]}, c=60)
    assert [r["passage_id"] for r in out] == ["a", "b"] or [r["passage_id"] for r in out] == ["b", "a"]
    assert len({r["passage_id"] for r in out}) == 2
    assert out[0]["sources"] == {"bm25": 1, "dense": 2} or out[0]["sources"] == {"bm25": 2, "dense": 1}


def test_dong_diem_on_dinh():
    # a (bm25 hạng 1) và c (dense hạng 1) cùng điểm; cùng hạng tốt nhất 1 -> theo passage_id
    assert [r["passage_id"] for r in rrf({"bm25": ["z"], "dense": ["a"]}, c=10)] == ["a", "z"]
    assert [r["passage_id"] for r in rrf({"dense": ["a"], "bm25": ["z"]}, c=10)] == ["a", "z"]


def test_top_k_c_khong_phai_top_k_va_loi():
    out = rrf({"bm25": ["a", "b", "c"]}, c=60, top_k=2)
    assert [r["passage_id"] for r in out] == ["a", "b"]
    assert rrf({"bm25": ["a"]}, top_k=0) == []
    with pytest.raises(ValueError):
        rrf({"bm25": ["a"]}, c=-1)


def test_id_lap_trong_mot_danh_sach_chi_tinh_lan_dau():
    out = rrf({"bm25": ["a", "a", "b"]}, c=1)
    assert out[0]["score"] == pytest.approx(1 / 2)
    assert {r["passage_id"]: r["sources"]["bm25"] for r in out} == {"a": 1, "b": 3}


def test_danh_sach_rong():
    assert rrf({"bm25": [], "dense": []}) == []
    assert [r["passage_id"] for r in rrf({"bm25": [], "dense": ["q"]})] == ["q"]
