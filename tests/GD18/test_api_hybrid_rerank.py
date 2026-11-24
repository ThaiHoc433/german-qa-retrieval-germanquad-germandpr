"""GĐ18 backend: hybrid là retriever hợp lệ, reranker tùy chọn, lỗi rõ ràng, và KHÔNG làm vỡ client cũ.
Dùng retriever/reader/reranker GIẢ nên chạy không cần GPU; HybridRetriever là bản thật (RRF thật trên hai danh sách giả)."""
import time

import pytest
from fastapi.testclient import TestClient

from germanqa.api import loader as L
from germanqa.api.app import create_app
from germanqa.api.settings import ApiSettings, load_settings
from germanqa.pipeline import QAPipeline
from germanqa.retrieval.hybrid import HybridRetriever

PASSAGES = {"p1": "Der Hund bellt laut im Garten.", "p2": "Die Katze schläft leise.", "p3": "Ein Vogel singt am Morgen."}


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offs, pos = [], 0
        for w in text.split(" "):
            offs.append((pos, pos + len(w)))
            pos += len(w) + 1
        return {"offset_mapping": offs}


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids
        self.calls = 0

    def retrieve(self, question, top_k):
        self.calls += 1
        return [{"passage_id": p, "score": 10.0 - i, "rank": i + 1} for i, p in enumerate(self.ids[:top_k])]


class FakeReader:
    tokenizer = FakeTokenizer()

    def read(self, question, passages):
        # điểm reader = số cuối của passage_id; span = từ đầu tiên
        return [{"passage_id": p["passage_id"], "text": p["text"].split(" ")[0], "start": 0, "end": len(p["text"].split(" ")[0]),
                 "score": float(p["passage_id"][1:])} for p in passages]


class FakeReranker:
    last_stats = {"pairs": 3, "windows": 3, "seconds": 0.0}

    def rerank(self, query, candidates, top_n=None):
        time.sleep(0.01)                                   # để rerank_ms đo được khác 0 một cách chắc chắn
        out = []
        for new, c in enumerate(reversed(candidates), 1):
            out.append({**c, "rank_before": c["rank"], "rank": new, "rerank_score": float(new)})
        return out


def pipe(retriever, name, reranker=None):
    return QAPipeline(retriever, FakeReader(), PASSAGES, "model-x", "corpus-y", f"exp-{name}", retrieval_top_k=3, reader_top_k=3, reranker=reranker)


def settings(**kw):
    base = dict(retrievers=["hybrid", "bm25", "dense"], default_retriever="hybrid", max_question_chars=100, max_retrieval_top_k=5,
                max_reader_top_k=3, request_timeout_s=5.0, max_inflight=4, max_rerank_pool=3, reranker="mmarco",
                reranker_models={"mmarco": {"checkpoint": "x"}})
    return ApiSettings(**{**base, **kw})


def make_loader(with_rerank=True, only=None, rerank_for=("hybrid",)):
    """rerank_for mặc định giống loader thật (chỉ hybrid); test đường ống rerank ở tầng app có thể mở thêm."""
    def loader(s):
        bm, dn = FakeRetriever(["p1", "p2", "p3"]), FakeRetriever(["p3", "p2", "p1"])
        hy = HybridRetriever(bm, dn, bm25_top_k=3, dense_top_k=3, fusion_top_k=3, rrf_c=10)
        pipes = {"bm25": pipe(bm, "bm25"), "dense": pipe(dn, "dense"), "hybrid": pipe(hy, "hybrid")}
        if with_rerank:
            for n in rerank_for:
                pipes[n + "+rerank"] = pipe(pipes[n].retriever, n, FakeReranker())
        return {k: v for k, v in pipes.items() if only is None or k.split("+")[0] in only}, {}
    return loader


def wait_ready(c):
    for _ in range(100):
        r = c.get("/health/ready")
        if r.status_code == 200:
            return r
        time.sleep(0.02)
    raise AssertionError(r.text)


@pytest.fixture
def client():
    with TestClient(create_app(settings(), make_loader())) as c:
        wait_ready(c)
        yield c


# ---------------- settings ----------------
def test_api_yaml_mac_dinh_la_hybrid_va_khong_nap_reranker():
    s = load_settings()
    assert s.default_retriever == "hybrid" and set(s.retrievers) == {"hybrid", "bm25", "dense"}
    assert s.reranker is None and {"bge_m3", "mmarco"} <= set(s.reranker_models) and s.max_rerank_pool == 20


def test_reranker_khong_co_trong_cau_hinh_la_loi():
    with pytest.raises(ValueError, match="reranker_models"):
        load_settings(reranker="khong_co")
    with pytest.raises(ValueError, match="max_rerank_pool"):
        load_settings(max_rerank_pool=0)
    assert load_settings(reranker="mmarco").reranker == "mmarco"


# ---------------- /qa với hybrid và rerank ----------------
def test_hybrid_la_retriever_hop_le_va_tra_ve_dung_ten(client):
    r = client.post("/api/v1/qa", json={"question": "Wer bellt?", "retriever": "hybrid"})
    assert r.status_code == 200 and r.json()["retriever"] == "hybrid" and r.json()["reranked"] is False


def test_khong_neu_retriever_thi_dung_mac_dinh_hybrid(client):
    assert client.post("/api/v1/qa", json={"question": "Wer?"}).json()["retriever"] == "hybrid"


def test_rerank_doi_tap_dua_vao_reader_va_tra_diem_rerank():
    # Tầng app không tự gắn tên retriever: loader giả mở rerank cho bm25 để có thứ hạng tất định (RRF giả có điểm hòa).
    with TestClient(create_app(settings(), make_loader(rerank_for=("bm25",)))) as client:
        wait_ready(client)
        _check_rerank_bm25(client)


def test_rerank_voi_bm25_hoac_dense_bi_tu_choi_vi_chi_hybrid(client):
    for name in ("bm25", "dense"):
        r = client.post("/api/v1/qa", json={"question": "Wer?", "retriever": name, "rerank": True})
        e = r.json()["error"]
        assert r.status_code == 422 and e["code"] == "rerank_unavailable" and e["details"] == [{"retriever": name, "reason": "hybrid_only"}]
    r = client.post("/api/v1/qa", json={"question": "Wer?", "retriever": "hybrid", "rerank": True})
    assert r.status_code == 200 and r.json()["reranked"] is True and r.json()["retriever"] == "hybrid"


def _check_rerank_bm25(client):
    base = client.post("/api/v1/qa", json={"question": "Wer?", "retriever": "bm25"}).json()
    rr = client.post("/api/v1/qa", json={"question": "Wer?", "retriever": "bm25", "rerank": True, "trace": True})
    assert rr.status_code == 200
    b = rr.json()
    assert b["retriever"] == "bm25" and b["reranked"] is True                       # tên retriever không lẫn hậu tố nội bộ
    assert base["reranked"] is False and base["rerank_score"] is None and base["timings"]["rerank_ms"] == 0.0
    assert b["rerank_score"] is not None and b["timings"]["rerank_ms"] >= 5.0      # thời gian xếp lại được đo thật, không phải số 0 cố định
    assert b["retrieval_score"] == 8.0 and b["rerank_score"] == 1.0           # p3: điểm truy hồi gốc 8,0 giữ nguyên; điểm xếp lại là trường riêng
    assert [c["rank_before_rerank"] for c in b["trace"]["candidates"]] == [3, 2, 1]
    assert [x["passage_id"] for x in b["trace"]["reranked"]] == ["p3", "p2", "p1"]


def test_rerank_khi_may_chu_khong_co_reranker_la_loi_ro_rang():
    with TestClient(create_app(settings(reranker=None), make_loader(with_rerank=False))) as c:
        wait_ready(c)
        r = c.post("/api/v1/qa", json={"question": "Wer?", "rerank": True})
        e = r.json()["error"]
        assert r.status_code == 422 and e["code"] == "rerank_unavailable" and e["details"][0]["reason"] == "not_configured"
        assert c.post("/api/v1/qa", json={"question": "Wer?"}).status_code == 200            # không rerank vẫn chạy bình thường


def test_rerank_pool_vuot_gioi_han_la_loi(client):
    r = client.post("/api/v1/qa", json={"question": "Wer?", "rerank": True, "retrieval_top_k": 4})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_top_k" and "max_rerank_pool" in r.json()["error"]["message"]
    assert client.post("/api/v1/qa", json={"question": "Wer?", "retrieval_top_k": 4}).status_code == 200      # không rerank thì không bị giới hạn này


def test_client_cu_khong_gui_rerank_van_chay_va_moi_truong_cu_con_nguyen(client):
    b = client.post("/api/v1/qa", json={"question": "Wer?", "retriever": "bm25"}).json()
    old = {"request_id", "status", "question", "answer", "passage_id", "passage_text", "answer_start", "answer_end", "answer_start_utf16",
           "answer_end_utf16", "retrieval_score", "reader_score", "retriever", "timings", "versions", "trace"}
    assert old <= set(b) and {"queue_ms", "retrieval_ms", "reader_ms", "pipeline_ms", "total_ms"} <= set(b["timings"])


# ---------------- /health/ready ----------------
def test_ready_bao_reranker(client):
    body = wait_ready(client).json()
    assert body["reranker"] == {"configured": "mmarco", "available": True, "reason": None, "retrievers": ["hybrid"]}
    assert body["retrievers"]["hybrid"]["available"] is True


def test_ready_reranker_khong_nap_duoc_co_ma_ly_do():
    def loader(s):
        p, _ = make_loader(with_rerank=False)(s)
        return p, {"rerank": "artifacts_missing"}
    with TestClient(create_app(settings(), loader)) as c:
        body = wait_ready(c).json()
        assert body["reranker"] == {"configured": "mmarco", "available": False, "reason": "artifacts_missing", "retrievers": []}


# ---------------- /retrieve ----------------
def test_retrieve_hybrid_tra_hang_va_diem_tung_nhanh_con_bm25_thi_khong(client):
    h = client.post("/api/v1/retrieve", json={"question": "Wer?", "retriever": "hybrid"}).json()
    top = h["candidates"][0]
    assert h["retriever"] == "hybrid" and set(top["sources"]) == {"bm25", "dense"} and top["bm25_score"] is not None and top["dense_score"] is not None
    b = client.post("/api/v1/retrieve", json={"question": "Wer?", "retriever": "bm25"}).json()["candidates"][0]
    assert b["sources"] is None and b["bm25_score"] is None and b["dense_score"] is None


# ---------------- loader ----------------
@pytest.fixture
def fake_build(monkeypatch):
    built = {"pipeline": [], "retriever": []}

    def build_pipeline(checkpoint, data_dir, retriever="bm25", **kw):
        built["pipeline"].append(retriever)
        return pipe(FakeRetriever(["p1", "p2", "p3"]), retriever)

    def build_retriever(name, corpus, data_dir, *a, **kw):
        built["retriever"].append(name)
        return FakeRetriever(["p3", "p2", "p1"])
    monkeypatch.setattr(L, "build_pipeline", build_pipeline)
    monkeypatch.setattr(L, "build_retriever", build_retriever)
    monkeypatch.setattr(L, "read_jsonl", lambda p: [])
    return built


def test_hybrid_dung_chung_bm25_va_dense_da_nap_khong_dung_lai(fake_build):
    pipes, un = L.load_pipelines(settings(reranker=None))
    assert un == {} and sorted(pipes) == ["bm25", "dense", "hybrid"]
    assert fake_build["pipeline"] == ["bm25"] and fake_build["retriever"] == ["dense"]          # mỗi thành phần dựng ĐÚNG một lần
    assert pipes["hybrid"].retriever.bm25 is pipes["bm25"].retriever and pipes["hybrid"].retriever.dense is pipes["dense"].retriever
    assert pipes["hybrid"].reader is pipes["bm25"].reader                                       # một reader cho tất cả


def test_chi_liet_ke_hybrid_van_dung_hai_thanh_phan_nhung_khong_lo_ra_ngoai(fake_build):
    pipes, _ = L.load_pipelines(settings(retrievers=["hybrid"], default_retriever="hybrid", reranker=None))
    assert sorted(pipes) == ["hybrid"] and sorted(fake_build["pipeline"] + fake_build["retriever"]) == ["bm25", "dense"]


def test_thieu_mot_thanh_phan_thi_hybrid_khong_kha_dung(fake_build, monkeypatch):
    def broken(name, corpus, data_dir, *a, **kw):
        raise FileNotFoundError("không có artifacts")
    monkeypatch.setattr(L, "build_retriever", broken)
    pipes, un = L.load_pipelines(settings(reranker=None))
    assert sorted(pipes) == ["bm25"] and un == {"dense": "artifacts_missing", "hybrid": "dependency_unavailable"}


def test_reranker_duoc_nap_mot_lan_va_chi_gan_sau_hybrid(fake_build, monkeypatch):
    made = []
    monkeypatch.setattr("germanqa.rerank.factory.build_reranker", lambda cfg: made.append(cfg) or FakeReranker())
    pipes, un = L.load_pipelines(settings())
    assert len(made) == 1 and un == {}
    variants = {k: v for k, v in pipes.items() if k.endswith("+rerank")}
    assert sorted(variants) == ["hybrid+rerank"]                                                # H1: chỉ hybrid -> cross-encoder
    for n, v in variants.items():
        base = pipes[n[:-len("+rerank")]]
        assert v.retriever is base.retriever and v.reader is base.reader and v.passages is base.passages
        assert base.reranker is None                                                            # pipeline gốc KHÔNG bị đổi hành vi


def test_reranker_hong_khong_lam_mat_retriever(fake_build, monkeypatch):
    def boom(cfg):
        raise FileNotFoundError("thiếu model")
    monkeypatch.setattr("germanqa.rerank.factory.build_reranker", boom)
    pipes, un = L.load_pipelines(settings())
    assert sorted(pipes) == ["bm25", "dense", "hybrid"] and un == {"rerank": "artifacts_missing"}


def test_reranker_bge_m3_dung_khoi_rerank_cua_rerank_yaml(fake_build, monkeypatch):
    seen = []
    monkeypatch.setattr("germanqa.rerank.factory.build_reranker", lambda cfg: seen.append(cfg) or FakeReranker())
    L.load_pipelines(settings(reranker="bge_m3", reranker_models={"bge_m3": {"config": "rerank"}}))
    assert seen[0]["checkpoint"].startswith("models/rerank_bge_m3") and seen[0]["overflow"] == "windows"


def test_khong_co_hybrid_thi_khong_nap_reranker(fake_build, monkeypatch):
    made = []
    monkeypatch.setattr("germanqa.rerank.factory.build_reranker", lambda cfg: made.append(cfg) or FakeReranker())
    pipes, un = L.load_pipelines(settings(retrievers=["bm25", "dense"], default_retriever="bm25"))
    assert made == [] and sorted(pipes) == ["bm25", "dense"] and un == {"rerank": "hybrid_unavailable"}
