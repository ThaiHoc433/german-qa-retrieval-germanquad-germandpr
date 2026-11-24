import pytest

from germanqa.retrieval.hybrid import HybridRetriever
from germanqa.retrieval.rrf import rrf


# ---------- RRF có trọng số ----------
def test_trong_so_tinh_tay():
    # c=1, w_dense=2: a: bm25 hạng 1 (1/2) + dense hạng 2 (2/3) ; b: dense hạng 1 (2/2=1)
    out = rrf({"bm25": ["a"], "dense": ["b", "a"]}, c=1, weights={"dense": 2})
    sc = {r["passage_id"]: r["score"] for r in out}
    assert sc["a"] == pytest.approx(1 / 2 + 2 / 3)
    assert sc["b"] == pytest.approx(1.0)
    assert [r["passage_id"] for r in out] == ["a", "b"]


def test_khong_trong_so_giong_cu_va_trong_so_1():
    lists = {"bm25": ["a", "b", "c"], "dense": ["c", "a", "d"]}
    assert rrf(lists, c=7) == rrf(lists, c=7, weights=None) == rrf(lists, c=7, weights={"dense": 1.0, "bm25": 1})


def test_trong_so_doi_thu_tu():
    lists = {"bm25": ["x", "y"], "dense": ["y", "x"]}
    assert rrf(lists, c=1, weights={"dense": 3})[0]["passage_id"] == "y"
    assert rrf(lists, c=1, weights={"dense": 0.2})[0]["passage_id"] == "x"
    assert [r["passage_id"] for r in rrf({"bm25": ["x"], "dense": ["y"]}, c=1, weights={"dense": 0})] == ["x", "y"]


def test_trong_so_loi():
    with pytest.raises(ValueError):
        rrf({"bm25": ["a"]}, weights={"bm25": -1})
    with pytest.raises(ValueError):
        rrf({"bm25": ["a"]}, weights={"dense": 1})


# ---------- HybridRetriever với nhánh giả ----------
class Fake:
    def __init__(self, rows):
        self.rows, self.asked = rows, []

    def retrieve(self, question, top_k=20):
        self.asked.append(top_k)
        return [{"passage_id": p, "score": s, "rank": i} for i, (p, s) in enumerate(self.rows[:top_k], start=1)]


def make(b, d, **kw):
    return HybridRetriever(Fake(b), Fake(d), **kw)


def test_hop_dung_nguon_va_diem_goc():
    h = make([("a", 9.0), ("b", 5.0)], [("b", 70.0), ("c", 60.0)], rrf_c=1, fusion_top_k=10)
    out = h.retrieve("q")
    by = {r["passage_id"]: r for r in out}
    assert [r["passage_id"] for r in out] == ["b", "a", "c"]
    assert by["b"]["sources"] == {"bm25": 2, "dense": 1} and by["b"]["bm25_score"] == 5.0 and by["b"]["dense_score"] == 70.0
    assert by["a"]["dense_score"] is None and by["a"]["sources"] == {"bm25": 1}      # nhánh vắng -> None, không phải 0
    assert by["c"]["bm25_score"] is None
    assert [r["rank"] for r in out] == [1, 2, 3]


def test_do_sau_va_top_k():
    h = make([("a", 3.0), ("b", 2.0), ("c", 1.0)], [("d", 3.0), ("e", 2.0), ("f", 1.0)],
             bm25_top_k=2, dense_top_k=3, fusion_top_k=4)
    assert h.bm25.asked == [] and len(h.retrieve("q")) == 4
    assert h.bm25.asked == [2] and h.dense.asked == [3]             # mỗi nhánh được hỏi đúng độ sâu của nó
    assert len(h.retrieve("q", top_k=100)) == 5                     # top_k quá lớn: chỉ có 2 + 3 đoạn
    assert len(h.retrieve("q", top_k=1)) == 1


def test_bm25_rong_va_ca_hai_rong():
    assert [r["passage_id"] for r in make([], [("d", 1.0)]).retrieve("q")] == ["d"]
    assert make([], []).retrieve("q") == []


def test_tat_dinh_va_trong_so_tu_cau_hinh():
    h = HybridRetriever.from_config(Fake([("a", 1.0)]), Fake([("b", 1.0)]),
                                    {"bm25_top_k": 5, "dense_top_k": 6, "fusion_top_k": 7, "rrf_c": 9, "w_dense": 2.0})
    assert h.config() == {"bm25_top_k": 5, "dense_top_k": 6, "fusion_top_k": 7, "rrf_c": 9, "w_dense": 2.0}
    assert h.retrieve("q") == h.retrieve("q")
    assert h.retrieve("q")[0]["passage_id"] == "b"                  # dense nặng gấp đôi, cùng hạng 1


def test_tham_so_sai():
    with pytest.raises(ValueError):
        make([], [], fusion_top_k=0)
    with pytest.raises(ValueError):
        make([], [], rrf_c=-1)
