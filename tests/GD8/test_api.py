"""Test API GĐ8 bằng pipeline GIẢ (retriever + reader giả, QAPipeline thật): không GPU, không tải mô hình."""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from germanqa.api.app import create_app
from germanqa.api.settings import ApiSettings, load_settings
from germanqa.pipeline import QAPipeline

# đoạn p_emoji có ký tự ngoài BMP (😀 = 1 ký tự Python, 2 đơn vị UTF-16) đứng TRƯỚC đáp án
PASSAGES = {"p1": "Der Hund bellt laut im Garten.",
            "p_emoji": "😀😀 Die Katze schläft leise.",
            "p3": "Ein Vogel singt am Morgen."}


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offs, pos = [], 0
        for w in text.split(" "):
            offs.append((pos, pos + len(w)))
            pos += len(w) + 1
        return {"offset_mapping": offs}


class FakeRetriever:
    def __init__(self, ids, delay=0.0):
        self.ids, self.delay = ids, delay

    def retrieve(self, question, top_k):
        time.sleep(self.delay)
        return [{"passage_id": p, "score": 10.0 - i, "rank": i + 1} for i, p in enumerate(self.ids[:top_k])]


class FakeReader:
    tokenizer = FakeTokenizer()

    def __init__(self, spans):
        self.spans = spans

    def read(self, question, passages):
        out = []
        for p in passages:
            if p["passage_id"] in self.spans:
                a, b, sc = self.spans[p["passage_id"]]
                out.append({"passage_id": p["passage_id"], "text": p["text"][a:b], "start": a, "end": b, "score": sc})
            else:
                out.append({"passage_id": p["passage_id"], "text": "", "start": -1, "end": -1, "score": float("-inf")})
        return out


def fake_pipe(name="bm25", delay=0.0):
    spans = {"p_emoji": (7, 12, 9.0), "p1": (4, 8, 3.0)}            # "Katze" nằm sau 2 emoji + dấu cách
    return QAPipeline(FakeRetriever(["p1", "p_emoji", "p3"], delay), FakeReader(spans), PASSAGES,
                      "model-x", "corpus-y", f"exp-{name}", retrieval_top_k=3, reader_top_k=3)


def settings(**kw):
    base = dict(retrievers=["bm25", "dense"], default_retriever="bm25", max_question_chars=100,
                max_retrieval_top_k=5, max_reader_top_k=3, request_timeout_s=5.0, max_inflight=4,
                cors_origins=["http://localhost:5173"])
    return ApiSettings(**{**base, **kw})


def loader_ok(s):
    return {"bm25": fake_pipe("bm25"), "dense": fake_pipe("dense")}, {}


def wait_ready(client, status=200, tries=100):
    for _ in range(tries):
        r = client.get("/health/ready")
        if r.status_code == status:
            return r
        time.sleep(0.02)
    raise AssertionError(f"ready không về {status}: {r.status_code} {r.text}")


@pytest.fixture
def client():
    with TestClient(create_app(settings(), loader_ok)) as c:
        wait_ready(c)
        yield c


def ask(c, question="Wer schläft?", **kw):
    return c.post("/api/v1/qa", json={"question": question, **kw})


# ---------- health / lifespan ----------
def test_ready_503_khi_dang_nap_roi_200():
    gate = threading.Event()

    def slow_loader(s):
        gate.wait(5)
        return loader_ok(s)

    with TestClient(create_app(settings(), slow_loader)) as c:
        assert c.get("/health/live").json() == {"status": "alive"}           # sống ngay cả khi chưa nạp xong
        r = c.get("/health/ready")
        assert r.status_code == 503 and r.json()["status"] == "loading"
        q = ask(c)
        assert q.status_code == 503 and q.json()["error"]["code"] == "not_ready"
        gate.set()
        r = wait_ready(c)
        assert r.json()["status"] == "ready" and r.json()["model"] == "model-x" and r.json()["corpus"] == "corpus-y"
        assert ask(c).status_code == 200


def test_ready_bao_retriever_khong_kha_dung_va_api_van_chay():
    with TestClient(create_app(settings(), lambda s: ({"bm25": fake_pipe()}, {"dense": "faiss_not_installed"}))) as c:
        body = wait_ready(c).json()
        assert body["retrievers"]["bm25"]["available"] is True
        assert body["retrievers"]["dense"] == {"available": False, "reason": "faiss_not_installed"}
        assert ask(c).status_code == 200
        r = ask(c, retriever="dense")
        assert r.status_code == 422 and r.json()["error"]["code"] == "retriever_unavailable"
        assert r.json()["error"]["details"][0]["reason"] == "faiss_not_installed"


def test_nap_that_bai_thi_ready_failed_va_qa_503():
    def boom(s):
        raise RuntimeError("D:\\bí mật\\đường dẫn")

    with TestClient(create_app(settings(), boom)) as c:
        r = wait_ready(c, 503)
        for _ in range(100):
            if r.json()["status"] == "failed":
                break
            time.sleep(0.02)
            r = c.get("/health/ready")
        assert r.json()["status"] == "failed" and "bí mật" not in r.text       # chỉ lộ tên lớp lỗi
        assert ask(c).json()["error"]["code"] == "load_failed"


def test_warmup_loi_cua_dense_chi_loai_dense():
    bad = fake_pipe("dense")
    bad.retriever.retrieve = lambda q, k: (_ for _ in ()).throw(RuntimeError("OOM"))
    with TestClient(create_app(settings(), lambda s: ({"bm25": fake_pipe(), "dense": bad}, {}))) as c:
        body = wait_ready(c).json()
        assert body["retrievers"]["dense"] == {"available": False, "reason": "warmup_failed"}


def test_retriever_mac_dinh_khong_kha_dung_thi_khong_ready():
    with TestClient(create_app(settings(default_retriever="dense"), lambda s: ({"bm25": fake_pipe()}, {"dense": "artifacts_missing"}))) as c:
        for _ in range(100):
            r = c.get("/health/ready")
            if r.json()["status"] == "failed":
                break
            time.sleep(0.02)
        assert r.status_code == 503 and r.json()["status"] == "failed"


# ---------- /qa ----------
def test_qa_tra_dap_an_va_offset_utf16_cho_javascript(client):
    r = ask(client)
    assert r.status_code == 200
    b = r.json()
    assert b["answer"] == "Katze" and b["passage_id"] == "p_emoji" and b["status"] == "ok"
    t = b["passage_text"]
    assert t[b["answer_start"]:b["answer_end"]] == "Katze"                     # offset Python
    u = t.encode("utf-16-le")
    assert u[2 * b["answer_start_utf16"]:2 * b["answer_end_utf16"]].decode("utf-16-le") == "Katze"   # như JS slice
    assert b["answer_start_utf16"] == b["answer_start"] + 2                    # 2 emoji ngoài BMP lệch +1 mỗi cái
    assert b["versions"] == {"model": "model-x", "corpus": "corpus-y", "experiment_id": "exp-bm25"}
    assert b["retriever"] == "bm25" and r.headers["x-request-id"] == b["request_id"]
    tm = b["timings"]
    assert set(tm) == {"queue_ms", "retrieval_ms", "reader_ms", "pipeline_ms", "total_ms", "rerank_ms"} and tm["rerank_ms"] == 0.0   # GĐ18: rerank_ms thêm vào, bằng 0 khi không rerank
    assert tm["total_ms"] >= tm["pipeline_ms"] >= 0


def test_qa_chon_retriever_dense_dung_pipeline_dense(client):
    assert ask(client, retriever="dense").json()["versions"]["experiment_id"] == "exp-dense"


def test_qa_trace_bat_thi_co_vet_va_json_hop_le_voi_diem_am_vo_cuc(client):
    b = ask(client, trace=True).json()
    cands = b["trace"]["candidates"]
    assert len(cands) == 3 and any(c["reader_score"] is None for c in cands)   # -inf -> null, không làm vỡ JSON
    assert ask(client).json()["trace"] is None


def test_qa_trang_thai_khong_co_span_van_200_va_offset_0():
    pipe = fake_pipe()
    with TestClient(create_app(settings(), lambda s: ({"bm25": pipe, "dense": fake_pipe()}, {}))) as c:
        wait_ready(c)
        pipe.reader.spans = {}                            # đổi SAU warmup (warmup mà không span thì pipeline bị loại)
        b = ask(c).json()
        assert b["status"] == "no_valid_span" and b["answer"] == "" and b["answer_start_utf16"] == 0


# ---------- validation ----------
@pytest.mark.parametrize("payload,code", [
    ({"question": "   "}, "empty_question"),
    ({"question": "x" * 101}, "question_too_long"),
    ({"question": "Wer?", "retrieval_top_k": 0}, "invalid_top_k"),
    ({"question": "Wer?", "retrieval_top_k": 6}, "invalid_top_k"),
    ({"question": "Wer?", "reader_top_k": 4}, "invalid_top_k"),
    ({"question": "Wer?", "retrieval_top_k": 2, "reader_top_k": 3}, "invalid_top_k"),
])
def test_validation_422_voi_ma_loi_on_dinh(client, payload, code):
    r = client.post("/api/v1/qa", json=payload)
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == code and e["request_id"] == r.headers["x-request-id"]


@pytest.mark.parametrize("payload", [{}, {"question": 5}, {"question": "Wer?", "retriever": "khong_co"},
                                     {"question": "Wer?", "mystery": 1}, {"question": "Wer?", "retrieval_top_k": "a"}])
def test_pydantic_loi_kieu_cung_ra_dang_loi_chung(client, payload):
    r = client.post("/api/v1/qa", json=payload)
    e = r.json()["error"]
    assert r.status_code == 422 and e["code"] == "invalid_request" and e["details"] and "field" in e["details"][0]


def test_404_405_cung_dang_loi_chung(client):
    assert client.get("/nope").json()["error"]["code"] == "not_found"
    assert client.get("/api/v1/qa").json()["error"]["code"] == "method_not_allowed"


def test_request_id_cua_client_duoc_giu_neu_hop_le_va_thay_neu_doc_hai(client):
    assert client.get("/health/live", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    bad = client.get("/health/live", headers={"X-Request-ID": "a b\n<x>"}).headers["x-request-id"]
    assert bad != "a b\n<x>" and len(bad) == 16


# ---------- /retrieve ----------
def test_retrieve_khong_goi_reader_va_tra_van_ban(client):
    r = client.post("/api/v1/retrieve", json={"question": "Wer?", "top_k": 2})
    b = r.json()
    assert r.status_code == 200 and b["top_k"] == 2
    assert [c["passage_id"] for c in b["candidates"]] == ["p1", "p_emoji"]
    assert b["candidates"][0]["text"] == PASSAGES["p1"] and b["candidates"][0]["rank"] == 1
    assert "reader_ms" not in b["timings"]
    assert client.post("/api/v1/retrieve", json={"question": "Wer?", "top_k": 99}).status_code == 422


# ---------- đồng thời / timeout / hàng đợi ----------
def test_timeout_tra_504_va_nguoi_sau_van_xep_hang_sau_luong_dang_chay():
    slow = fake_pipe(delay=0.0)
    with TestClient(create_app(settings(request_timeout_s=0.3), lambda s: ({"bm25": slow, "dense": fake_pipe()}, {}))) as c:
        wait_ready(c)
        slow.retriever.delay = 0.8                       # sau warmup mới làm chậm
        calls = []
        orig = slow.retriever.retrieve
        slow.retriever.retrieve = lambda q, k: (calls.append(q), orig(q, k))[1]
        t = time.perf_counter()
        r = ask(c, "erste")
        assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"
        assert time.perf_counter() - t < 0.7              # trả lời đúng hạn chót, không đợi model xong
        slow.retriever.delay = 0.0
        r2 = ask(c, "zweite")                             # luồng cũ còn giữ khóa -> request mới xếp hàng rồi hết hạn chót
        assert r2.status_code == 504
        time.sleep(0.8)
        assert ask(c, "dritte").status_code == 200        # luồng cũ xong thì chạy lại bình thường
        assert calls == ["erste", "dritte"]               # "zweite" đã nhận 504 nên model KHÔNG chạy cho nó, và không chồng lên "erste"


def test_hang_doi_day_tra_503_busy():
    slow = fake_pipe()
    with TestClient(create_app(settings(max_inflight=2, request_timeout_s=5), lambda s: ({"bm25": slow, "dense": fake_pipe()}, {}))) as c:
        wait_ready(c)
        slow.retriever.delay = 0.5
        out = []
        threads = [threading.Thread(target=lambda: out.append(ask(c).status_code)) for _ in range(5)]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join()
        assert sorted(out) == [200, 200, 503, 503, 503]


def test_health_van_tra_loi_khi_qa_dang_chay():
    slow = fake_pipe()
    with TestClient(create_app(settings(), lambda s: ({"bm25": slow, "dense": fake_pipe()}, {}))) as c:
        wait_ready(c)
        slow.retriever.delay = 0.8
        th = threading.Thread(target=lambda: ask(c))
        th.start()
        time.sleep(0.1)
        t = time.perf_counter()
        assert c.get("/health/live").status_code == 200
        assert time.perf_counter() - t < 0.3              # event loop không bị model chặn
        th.join()


def test_cac_request_model_chay_noi_tiep_khong_chong_nhau():
    active, peak = [0], [0]
    lock = threading.Lock()
    pipe = fake_pipe()
    orig = pipe.retriever.retrieve

    def watched(q, k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            time.sleep(0.05)
            return orig(q, k)
        finally:
            with lock:
                active[0] -= 1

    with TestClient(create_app(settings(), lambda s: ({"bm25": pipe, "dense": fake_pipe()}, {}))) as c:
        wait_ready(c)
        pipe.retriever.retrieve = watched
        ts = [threading.Thread(target=lambda: ask(c)) for _ in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert peak[0] == 1


# ---------- log & CORS ----------
def test_log_khong_chua_noi_dung_cau_hoi(client, caplog):
    caplog.set_level("INFO", logger="germanqa.api")
    ask(client, "Geheime Frage über Adenauer?")
    lines = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    req = [x for x in lines if x.get("event") == "request" and x["path"] == "/api/v1/qa"][0]
    assert req["question_chars"] == len("Geheime Frage über Adenauer?") and len(req["question_sha"]) == 8
    assert "Geheime" not in caplog.text and req["retriever"] == "bm25" and req["status"] == 200


def test_cors_cho_react_dev_server(client):
    r = client.options("/api/v1/qa", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                                              "Access-Control-Request-Headers": "content-type"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in client.options(
        "/api/v1/qa", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}).headers


def test_loi_500_khong_lo_chi_tiet(client):
    pipe = client.app.state.svc.pipelines["bm25"]
    pipe.retriever.retrieve = lambda q, k: (_ for _ in ()).throw(RuntimeError("D:\\secret\\path.txt"))
    r = ask(client)
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error" and "secret" not in r.text


# ---------- settings ----------
def test_load_settings_doc_api_yaml():
    s = load_settings()
    assert s.default_retriever in s.retrievers and s.max_reader_top_k <= s.max_retrieval_top_k
    with pytest.raises(ValueError):
        load_settings(default_retriever="khong_co")
