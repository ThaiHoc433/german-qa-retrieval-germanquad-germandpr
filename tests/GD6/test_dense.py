"""Test dense retriever GĐ6 bằng encoder GIẢ: không cần GPU, không tải mô hình."""
import numpy as np
import pytest

from germanqa.data.io_utils import file_sha1
from germanqa.retrieval.dense import (DenseRetriever, DPREncoder, build_index, check_embeddings, exact_topk,
                                      load_artifacts, save_artifacts)

IDS = [f"p{i}" for i in range(50)]


class FakeEncoder:
    """Câu hỏi 'q<i>' được mã hóa thành vector của đoạn i (nhân hệ số); câu lạ -> vector 0."""
    def __init__(self, vectors, scale=1.0):
        self.vectors, self.scale, self.calls = vectors, scale, []

    def encode_questions(self, questions, max_length=64, batch_size=64):
        self.calls.append((list(questions), max_length))
        return np.stack([self.vectors[int(q[1:])] * self.scale if q.startswith("q") else np.zeros(self.vectors.shape[1], np.float32)
                         for q in questions]).astype(np.float32)


@pytest.fixture
def vecs():
    return np.random.default_rng(0).normal(size=(50, 16)).astype(np.float32)


def test_check_embeddings_bat_loi_so_luong_trung_id_nan(vecs):
    check_embeddings(vecs, IDS)
    with pytest.raises(ValueError):
        check_embeddings(vecs[:-1], IDS)
    with pytest.raises(ValueError):
        check_embeddings(vecs, IDS[:-1] + ["p0"])
    bad = vecs.copy()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError):
        check_embeddings(bad, IDS)
    bad[3, 2] = np.inf
    with pytest.raises(ValueError):
        check_embeddings(bad, IDS)


def test_faiss_khop_voi_tinh_truc_tiep(vecs):
    q = np.random.default_rng(1).normal(size=(8, 16)).astype(np.float32)
    s_f, i_f = build_index(vecs).search(q, 10)
    s_e, i_e = exact_topk(q, vecs, 10)
    assert (i_f == i_e).all()
    assert np.allclose(s_f, s_e, atol=1e-4)


def test_retrieve_tra_dung_doan_thu_hang_bat_dau_tu_1(vecs):
    r = DenseRetriever(FakeEncoder(vecs, scale=3.0), vecs, IDS)
    hits = r.retrieve("q7", 5)
    assert hits[0]["passage_id"] == "p7" and [h["rank"] for h in hits] == [1, 2, 3, 4, 5]
    assert all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(4))


def test_diem_la_tich_vo_huong_khong_chuan_hoa(vecs):
    r = DenseRetriever(FakeEncoder(vecs, scale=2.0), vecs, IDS)
    top = r.retrieve("q3", 1)[0]
    assert top["score"] == pytest.approx(2.0 * float(vecs[3] @ vecs[3]), rel=1e-4)    # không phải cosine (sẽ là 1)


def test_thu_tu_mapping_khong_lech_khi_tron_id(vecs):
    perm = np.random.default_rng(2).permutation(50)
    ids2 = [IDS[i] for i in perm]
    r = DenseRetriever(FakeEncoder(vecs[perm]), vecs[perm], ids2)          # vector và ID cùng bị xáo
    assert r.retrieve("q0", 1)[0]["passage_id"] == ids2[0]


def test_top_k_lon_hon_kho_va_cau_hoi_la(vecs):
    r = DenseRetriever(FakeEncoder(vecs), vecs, IDS)
    assert len(r.retrieve("q1", 1000)) == 50
    assert len(r.retrieve("xyz", 3)) == 3                                  # vector 0: mọi đoạn đồng điểm vẫn trả đủ


def test_retrieve_many_khop_retrieve_tung_cau(vecs):
    r = DenseRetriever(FakeEncoder(vecs), vecs, IDS)
    many = r.retrieve_many(["q1", "q2", "q3"], 4)
    assert many == [r.retrieve(q, 4) for q in ("q1", "q2", "q3")]
    assert set(r.last_timing) == {"encode", "search"}


def test_cau_hoi_duoc_cat_theo_question_max_length(vecs):
    enc = FakeEncoder(vecs)
    DenseRetriever(enc, vecs, IDS, question_max_length=33).retrieve("q1", 1)
    assert enc.calls[0][1] == 33


def test_luu_doc_lai_va_tu_choi_corpus_da_doi(vecs, tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"passage_id": "p0"}\n', encoding="utf-8")
    save_artifacts(tmp_path / "art", vecs, IDS, {"corpus_sha1": file_sha1(corpus)})
    v2, ids2, man = load_artifacts(tmp_path / "art", corpus)
    assert (v2 == vecs).all() and ids2 == IDS and man["corpus_sha1"] == file_sha1(corpus)
    corpus.write_text('{"passage_id": "p1"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_artifacts(tmp_path / "art", corpus)


def test_passage_input_sai_bi_tu_choi_truoc_khi_nap_mo_hinh():
    enc = DPREncoder("a", "b", device="cpu")
    with pytest.raises(ValueError):
        enc.encode_passages(["t"], ["x"], passage_input="khac")


def test_dung_trong_qapipeline_khong_sua_pipeline(vecs):
    from germanqa.pipeline import QAPipeline

    class Reader:
        class tokenizer:
            def __call__(self, *a, **k): return {"offset_mapping": [(0, 1)]}
        tokenizer = tokenizer()
        def read(self, q, passages):
            return [{"passage_id": p["passage_id"], "text": p["text"][:1], "start": 0, "end": 1, "score": 1.0} for p in passages]

    texts = {pid: f"Text {pid}" for pid in IDS}
    pipe = QAPipeline(DenseRetriever(FakeEncoder(vecs), vecs, IDS), Reader(), texts, "m", "c", "e")
    r = pipe.answer("q5", trace=True)
    assert r.status == "ok" and r.trace["retrieved"][0]["passage_id"] == "p5"


def test_token_type_ids_luon_bang_0_khi_goi_encoder():
    """Encoder DPR của deepset học với segment id toàn 0; token_type_ids 0/1 làm retrieval kém đi rõ rệt."""
    import torch

    class Enc(dict):
        def to(self, device): return self

    class Tok:
        def __call__(self, *a, **k):
            return Enc(input_ids=torch.ones(len(a[0]), 4, dtype=torch.long), attention_mask=torch.ones(len(a[0]), 4, dtype=torch.long),
                       token_type_ids=torch.tensor([[0, 0, 1, 1]] * len(a[0])))
    seen = []

    class Model:
        class config: hidden_size = 3
        def __call__(self, **kw):
            seen.append(kw["token_type_ids"].clone())
            return type("O", (), {"pooler_output": torch.ones(kw["input_ids"].shape[0], 3)})()

    enc = DPREncoder("a", "b", device="cpu")
    enc._ctx = (Model(), Tok(), "rev")
    # độ dài thật được đếm bằng tok(...)["input_ids"]; Tok trả về hình dạng đủ dùng cho việc đó
    vecs, _ = enc.encode_passages(["T1", "T2"], ["x y", "z"], "title_text", 16, 2, 0)
    assert vecs.shape == (2, 3) and all(int(t.sum()) == 0 for t in seen) and seen
