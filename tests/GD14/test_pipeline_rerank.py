import pytest

from germanqa.pipeline import QAPipeline


class FakeTok:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offs, i = [], 0
        for w in text.split(" "):
            offs.append((i, i + len(w)))
            i += len(w) + 1
        return {"offset_mapping": offs, "input_ids": list(range(len(offs)))}


class FakeReader:
    """Mỗi đoạn trả về span là từ đầu tiên; điểm = số trong tên đoạn, để biết đoạn nào thắng."""
    tokenizer = FakeTok()

    def __init__(self):
        self.seen = []

    def read(self, question, passages):
        self.seen.append([p["passage_id"] for p in passages])
        return [{"text": p["text"].split(" ")[0], "start": 0, "end": len(p["text"].split(" ")[0]),
                 "score": float(p["passage_id"][1:])} for p in passages]


class FakeRetriever:
    def retrieve(self, question, top_k):
        return [{"passage_id": f"p{i}", "score": 1.0 / (i + 1), "rank": i} for i in range(1, 9)][:top_k]   # p1..p8


PASSAGES = {f"p{i}": f"wort{i} text" for i in range(1, 9)}


class ReverseReranker:
    last_stats = {"pairs": 8, "windows": 8, "seconds": 0.0}

    def rerank(self, query, cands, top_n=None):
        out = []
        for new, c in enumerate(reversed(cands), 1):
            r = dict(c)
            r.update(rank_before=c["rank"], rank=new, rerank_score=float(new))
            out.append(r)
        return out


def make(reranker=None):
    reader = FakeReader()
    return QAPipeline(FakeRetriever(), reader, PASSAGES, "m", "c", "e", retrieval_top_k=8, reader_top_k=3, reranker=reranker), reader


def test_khong_co_reranker_giu_dung_hanh_vi_cu():
    pipe, reader = make()
    r = pipe.answer("Wer ist hier?", trace=True)
    assert reader.seen == [["p1", "p2", "p3"]] and r.passage_id == "p3"
    assert r.rerank_time == 0.0 and r.rerank_score is None
    assert "reranked" not in r.trace and "rank_before_rerank" not in r.trace["candidates"][0]
    assert [h["passage_id"] for h in r.trace["retrieved"]] == [f"p{i}" for i in range(1, 9)]


def test_reranker_doi_tap_dua_vao_reader_va_giu_diem_rrf():
    pipe, reader = make(ReverseReranker())
    r = pipe.answer("Wer ist hier?", trace=True)
    assert reader.seen == [["p8", "p7", "p6"]]                              # reader nhận 3 đoạn đầu SAU khi xếp lại
    assert r.passage_id == "p8" and r.rerank_score == 1.0
    assert r.retrieval_score == pytest.approx(1.0 / 9)                      # điểm truy hồi vẫn là điểm RRF gốc, không bị ghi đè
    assert [h["passage_id"] for h in r.trace["retrieved"]] == [f"p{i}" for i in range(1, 9)]       # vết truy hồi giữ nguyên
    assert [x["passage_id"] for x in r.trace["reranked"]][:3] == ["p8", "p7", "p6"]
    assert r.trace["reranked"][0]["rank_before"] == 8 and r.trace["candidates"][0]["rank_before_rerank"] == 8
    assert sorted(x["passage_id"] for x in r.trace["reranked"]) == sorted(PASSAGES)       # không thêm, không mất đoạn nào
    assert r.rerank_time >= 0.0 and r.trace["rerank_stats"]["pairs"] == 8


def test_cau_hoi_rong_va_khong_doan_khong_goi_reranker():
    class Boom:
        def rerank(self, *a, **k):
            raise AssertionError("không được gọi")
    pipe, _ = make(Boom())
    assert pipe.answer("   ").status == "empty_question"
