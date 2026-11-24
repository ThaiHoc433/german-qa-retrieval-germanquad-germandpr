"""/api/v1/read: đọc hiểu với đoạn văn tự nhập (không truy hồi). Reader GIẢ, QAPipeline thật: không GPU."""
import pytest
from fastapi.testclient import TestClient

from germanqa.api.app import create_app
from germanqa.pipeline import QAPipeline

from tests.GD8.test_api import FakeRetriever, FakeTokenizer, settings, wait_ready

CTX = "Die Berliner Mauer wurde am 13. August 1961 errichtet und trennte West-Berlin von Ost-Berlin."


class SpanReader:
    """Trả đúng chuỗi 'needle' nếu có trong đoạn; không có thì không có span hợp lệ."""
    tokenizer = FakeTokenizer()

    def __init__(self, needle="am 13. August 1961"):
        self.needle, self.calls = needle, []

    def read(self, question, passages):
        self.calls.append((question, passages))
        out = []
        for p in passages:
            i = p["text"].find(self.needle)
            out.append({"passage_id": p["passage_id"], "text": self.needle if i >= 0 else "", "start": i,
                        "end": i + len(self.needle) if i >= 0 else -1, "score": 5.0 if i >= 0 else float("-inf")})
        return out


@pytest.fixture
def reader():
    return SpanReader()


@pytest.fixture
def client(reader):
    pipe = QAPipeline(FakeRetriever(["p1"]), reader, {"p1": CTX}, "model-x", "corpus-y", "exp", retrieval_top_k=1, reader_top_k=1)
    with TestClient(create_app(settings(max_context_chars=200), lambda s: ({"bm25": pipe}, {}))) as c:
        wait_ready(c)
        reader.calls.clear()                                   # bỏ lượt warmup
        yield c


def test_read_tra_dap_an_offset_va_em_f1(client, reader):
    r = client.post("/api/v1/read", json={"question": "Wann wurde die Berliner Mauer gebaut?", "context": CTX,
                                          "expected_answer": "Am 13. August 1961"})
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ok" and b["answer"] == "am 13. August 1961"
    assert CTX[b["answer_start"]:b["answer_end"]] == b["answer"] and b["answer_start_utf16"] == b["answer_start"]
    assert b["em"] == 1.0 and b["f1"] == 1.0 and b["reader_score"] == 5.0 and b["model"] == "model-x"
    assert reader.calls == [("Wann wurde die Berliner Mauer gebaut?", [{"passage_id": "user", "text": CTX}])]   # không truy hồi


def test_read_khong_co_expected_thi_khong_tinh_em(client):
    b = client.post("/api/v1/read", json={"question": "Wann?", "context": CTX}).json()
    assert b["em"] is None and b["f1"] is None and b["expected_answer"] is None


def test_read_sai_dap_an_thi_em_0_f1_mot_phan(client):
    b = client.post("/api/v1/read", json={"question": "Wann?", "context": CTX, "expected_answer": "August 1962"}).json()
    assert b["em"] == 0.0 and 0 < b["f1"] < 1


def test_read_khong_tim_duoc_span(client):
    b = client.post("/api/v1/read", json={"question": "Wann?", "context": "Ganz anderer Text.", "expected_answer": "1961"}).json()
    assert b["status"] == "no_valid_span" and b["answer"] == "" and b["answer_start"] == -1 and b["reader_score"] is None
    assert b["em"] == 0.0


def test_read_utf16_offset_khi_co_emoji(client):
    b = client.post("/api/v1/read", json={"question": "Wann?", "context": "😀 " + CTX}).json()
    assert b["answer_start_utf16"] == b["answer_start"] + 1


@pytest.mark.parametrize("body,code", [
    ({"question": "  ", "context": CTX}, "empty_question"),
    ({"question": "Wann?", "context": "   "}, "empty_context"),
    ({"question": "Wann?", "context": "x" * 201}, "context_too_long"),
    ({"question": "Wann?"}, "invalid_request"),
    ({"question": "Wann?", "context": CTX, "retriever": "bm25"}, "invalid_request"),
])
def test_read_loi_ro_rang(client, body, code):
    r = client.post("/api/v1/read", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == code
