
"""Test pipeline GĐ5 bằng retriever và reader GIẢ: không cần GPU, không tải mô hình."""
import inspect

import pytest

import germanqa.pipeline as pipeline_module
from germanqa.pipeline import MAX_QUESTION_TOKENS, QAPipeline

PASSAGES = {"p1": "Der Hund bellt laut im Garten.", "p2": "Die Katze schläft leise auf dem Sofa.",
            "p3": "Ein Vogel singt am Morgen."}


class FakeTokenizer:
    """Mỗi từ là một token; offset là vị trí ký tự của từ."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offs, pos = [], 0
        for w in text.split(" "):
            offs.append((pos, pos + len(w)))
            pos += len(w) + 1
        return {"offset_mapping": offs}


class FakeRetriever:
    def __init__(self, ids):
        self.ids, self.calls = ids, []

    def retrieve(self, question, top_k):
        self.calls.append((question, top_k))
        return [{"passage_id": p, "score": 10.0 - i, "rank": i + 1} for i, p in enumerate(self.ids[:top_k])]


class FakeReader:
    """Trả span cố định theo từng đoạn: {passage_id: (start, end, score)}; thiếu thì không có span hợp lệ."""
    tokenizer = FakeTokenizer()

    def __init__(self, spans):
        self.spans, self.calls = spans, []

    def read(self, question, passages):
        self.calls.append((question, [p["passage_id"] for p in passages]))
        out = []
        for p in passages:
            if p["passage_id"] in self.spans:
                a, b, sc = self.spans[p["passage_id"]]
                out.append({"passage_id": p["passage_id"], "text": p["text"][a:b], "start": a, "end": b, "score": sc})
            else:
                out.append({"passage_id": p["passage_id"], "text": "", "start": -1, "end": -1, "score": float("-inf")})
        return out


def make(ids=("p1", "p2", "p3"), spans=None, **kw):
    spans = spans if spans is not None else {"p1": (4, 8, 3.0), "p2": (4, 9, 7.5), "p3": (4, 9, 1.0)}
    return QAPipeline(FakeRetriever(list(ids)), FakeReader(spans), PASSAGES, "m1", "c1", "e1", **kw)


def test_chon_span_diem_reader_cao_nhat_khong_phai_doan_hang_1():
    r = make().answer("Wer schläft?")
    assert r.passage_id == "p2" and r.answer == "Katze"
    assert r.retrieval_score == 9.0                      # điểm BM25 của p2 (hạng 2), chỉ để báo cáo, không tham gia chọn
    assert r.reader_score == 7.5


def test_dap_an_la_chuoi_con_dung_tai_offset():
    r = make().answer("Wer bellt?")
    assert r.passage_text[r.answer_start:r.answer_end] == r.answer


def test_chi_5_doan_dau_vao_reader_va_20_doan_di_truy_hoi():
    ids = [f"p{i}" for i in range(1, 4)]
    pipe = make(ids=ids, reader_top_k=2)
    pipe.answer("Frage")
    assert pipe.retriever.calls == [("Frage", 20)]
    assert pipe.reader.calls == [("Frage", ["p1", "p2"])]


def test_hoa_diem_chon_doan_hang_cao_hon():
    r = make(spans={"p1": (4, 8, 5.0), "p2": (4, 9, 5.0)}).answer("Frage")
    assert r.passage_id == "p1"


def test_cau_hoi_rong_tra_dap_an_rong_khong_goi_retriever():
    pipe = make()
    r = pipe.answer("   ")
    assert r.status == "empty_question" and r.answer == "" and pipe.retriever.calls == []


def test_retriever_khong_tra_doan_nao():
    r = make(ids=()).answer("Frage")
    assert r.status == "no_passages" and r.answer == ""


def test_khong_doan_nao_co_span_hop_le():
    r = make(spans={}).answer("Frage")
    assert r.status == "no_valid_span" and r.answer == ""


def test_top_k_lon_hon_kho_khong_sap():
    r = make().answer("Frage", retrieval_top_k=1000, reader_top_k=1000)
    assert r.status == "ok"


def test_tham_so_sai():
    with pytest.raises(TypeError):
        make().answer(None)
    with pytest.raises(ValueError):
        make().answer("Frage", reader_top_k=0)


def test_cau_hoi_qua_dai_bi_cat_va_bao_trang_thai():
    pipe = make()
    r = pipe.answer(" ".join(["Wort"] * (MAX_QUESTION_TOKENS + 30)))
    assert r.status == "question_truncated"
    assert len(pipe.reader.calls[0][0].split(" ")) == MAX_QUESTION_TOKENS


def test_thoi_gian_va_phien_ban():
    r = make().answer("Frage")
    assert r.total_time >= r.retrieval_time + r.reader_time >= 0
    assert (r.model_version, r.corpus_version, r.experiment_id) == ("m1", "c1", "e1")


def test_trace_bat_thi_co_top_20_va_ung_vien_tat_thi_none():
    on, off = make().answer("Frage", trace=True), make().answer("Frage")
    assert [h["passage_id"] for h in on.trace["retrieved"]] == ["p1", "p2", "p3"]
    assert [c["passage_id"] for c in on.trace["candidates"]] == ["p1", "p2", "p3"]
    assert off.trace is None


def test_loi_khi_dap_an_khong_khop_van_ban_goc():
    class BadReader(FakeReader):
        def read(self, question, passages):
            return [{"passage_id": "p1", "text": "FALSCH", "start": 0, "end": 3, "score": 1.0}]
    pipe = QAPipeline(FakeRetriever(["p1"]), BadReader({}), PASSAGES, "m", "c", "e")
    with pytest.raises(ValueError):
        pipe.answer("Frage")


def test_duong_chay_answer_khong_dung_qrels_hay_dap_an_vang():
    src = inspect.getsource(QAPipeline.answer) + inspect.getsource(QAPipeline._cut_question)
    for banned in ("qrels", "qa_dev", "qa_test", "answers", "gold"):
        assert banned not in src
    assert "load_examples" not in inspect.getsource(pipeline_module)

def test_bat_trace_van_co_du_khoa_khi_khong_co_doan_nao_hoac_cau_rong():
    """Lỗi gặp khi chạy test GĐ7: BM25 không trả đoạn nào và trace=True trả về None, script đọc trace['retrieved'] bị sập."""
    for pipe, q in ((make(ids=()), "Frage"), (make(), "   ")):
        r = pipe.answer(q, trace=True)
        assert r.trace == {"retrieved": [], "candidates": []}
    assert make(ids=()).answer("Frage").trace is None

def test_cau_hoi_duoc_chuan_hoa_nfc_va_bo_ky_tu_vo_hinh_truoc_retriever_va_reader():
    """Lỗi GĐ2/GĐ6: câu hỏi chỉ được strip, nên dạng NFD ('o' + dấu ¨ rời) hay ký tự zero-width làm BM25 tách
    'höchste' thành 'ho', 'chste'. Corpus đã NFC từ GĐ2, nên câu hỏi phải được chuẩn hóa y như vậy ở điểm vào."""
    import unicodedata
    nfc = "Welche Zeitung hat die höchste Auflage?"
    variants = (unicodedata.normalize("NFD", nfc), nfc.replace("Zeitung", "Zei​tung"),
                "﻿  " + nfc.replace(" hat", " hat") + " ‍")
    for q in variants:
        pipe = make()
        pipe.answer(q)
        assert pipe.retriever.calls[0][0] == nfc
        assert pipe.reader.calls[0][0] == nfc


def test_cau_hoi_chi_co_ky_tu_vo_hinh_la_cau_rong():
    pipe = make()
    r = pipe.answer("​‌ ﻿")
    assert r.status == "empty_question" and pipe.retriever.calls == []
