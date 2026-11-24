import numpy as np
import pytest

from germanqa.data.io_utils import file_sha1, write_jsonl
from germanqa.retrieval import bm25_store as S
from germanqa.retrieval.bm25 import BM25, BM25Retriever
from germanqa.retrieval.tokenizer import GERMAN_STOPWORDS

CORPUS = [{"passage_id": f"p{i}", "retrieval_text": t, "original_text": t} for i, t in enumerate([
    "Der Rhein fließt durch Köln und Bonn", "Bonn war die Hauptstadt der Bundesrepublik Deutschland",
    "Köln hat einen Dom und viele Kirchen", "Die Hauptstadt von Österreich ist Wien", "Größe und Straße in Österreich",
    "Fußball ist in Deutschland sehr beliebt", "Der Dom steht in Köln am Rhein"])]
QUERIES = ["Hauptstadt Bonn", "Wo steht der Dom in Köln?", "Straße Österreich", "nichts davon xyzzy", ""]
P = dict(k1=0.7, b=0.2, use_stopwords=True)


@pytest.fixture
def cp(tmp_path):
    p = tmp_path / "corpus.jsonl"
    write_jsonl(p, CORPUS)
    return p


def fp(cp, **over):
    a = {**P, **over}
    return S.compute_fingerprint(file_sha1(cp), a["k1"], a["b"], a["use_stopwords"])


def test_nap_lai_cho_ket_qua_giong_het_ca_diem(cp, tmp_path):
    fresh = BM25Retriever(CORPUS, **P)
    S.save_index(fresh, fp(cp), tmp_path / "s")
    loaded = S.load_index(tmp_path / "s", fp(cp))
    for q in QUERIES:
        assert fresh.retrieve(q, 5) == loaded.retrieve(q, 5)          # passage_id, điểm (float), hạng


def test_dau_van_tay_doi_khi_doi_bat_ky_thanh_phan(cp, tmp_path, monkeypatch):
    base = fp(cp)["sha1"]
    assert fp(cp)["sha1"] == base                                       # xác định
    assert fp(cp, k1=0.9)["sha1"] != base
    assert fp(cp, b=0.5)["sha1"] != base
    assert fp(cp, use_stopwords=False)["sha1"] != base
    other = tmp_path / "c2.jsonl"
    write_jsonl(other, CORPUS[:-1])
    assert fp(other)["sha1"] != base                                    # corpus đổi


def test_doi_stopword_hoac_tokenizer_hoac_cong_thuc_deu_bat_duoc(cp, monkeypatch):
    base = fp(cp)["sha1"]
    with monkeypatch.context() as m:                                    # thêm một stopword KHÔNG xuất hiện trong bộ mẫu
        m.setattr(S, "GERMAN_STOPWORDS", GERMAN_STOPWORDS | {"zzqwort"})
        assert fp(cp)["sha1"] != base                                   # chỉ băm danh sách mới bắt được từ này
    real_tok = S.tokenize
    with monkeypatch.context() as m:                                    # sửa tokenizer: bỏ dấu ß
        m.setattr(S, "tokenize", lambda t, sw=None: real_tok(t.replace("ß", "ss"), sw))
        assert fp(cp)["sha1"] != base
    class Variant(BM25):                                               # sửa công thức: nhân điểm 1.1
        def fit(self, doc_ids, token_lists):
            super().fit(doc_ids, token_lists)
            self.weights = {t: (i, w * 1.1) for t, (i, w) in self.weights.items()}
            return self
    with monkeypatch.context() as m:
        m.setattr(S, "BM25", Variant)
        assert fp(cp)["sha1"] != base


def test_lech_dau_van_tay_strict_bao_loi_khong_strict_dung_lai(cp, tmp_path):
    d = tmp_path / "s"
    r, info = S.load_or_build(cp, CORPUS, **P, store_dir=d)
    assert info["source"] == "built" and info["reason"] == "no_store"
    r2, info2 = S.load_or_build(cp, CORPUS, **P, store_dir=d)
    assert info2["source"] == "loaded"
    with pytest.raises(S.FingerprintMismatch, match="k1"):
        S.load_or_build(cp, CORPUS, k1=0.9, b=0.2, use_stopwords=True, store_dir=d, strict=True)
    r3, info3 = S.load_or_build(cp, CORPUS, k1=0.9, b=0.2, use_stopwords=True, store_dir=d)
    assert info3["source"] == "built" and info3["reason"] == "fingerprint_mismatch"
    assert r3.index.k1 == 0.9
    assert S.load_or_build(cp, CORPUS, k1=0.9, b=0.2, use_stopwords=True, store_dir=d)[1]["source"] == "loaded"


def test_khong_luu_duoc_van_phuc_vu(cp, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise PermissionError("volume chỉ đọc")
    monkeypatch.setattr(S, "save_index", boom)
    r, info = S.load_or_build(cp, CORPUS, **P, store_dir=tmp_path / "ro")
    assert info["source"] == "built" and info["reason"].endswith("save_failed") and r.retrieve("Hauptstadt", 3)


def test_strict_thieu_chi_muc_la_loi(cp, tmp_path):
    with pytest.raises(FileNotFoundError):
        S.load_or_build(cp, CORPUS, **P, store_dir=tmp_path / "none", strict=True)


def test_tep_bi_sua_hoac_hong_thi_phat_hien(cp, tmp_path):
    d = tmp_path / "s"
    S.save_index(BM25Retriever(CORPUS, **P), fp(cp), d)
    raw = bytearray((d / "index.npz").read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (d / "index.npz").write_bytes(bytes(raw))
    with pytest.raises(S.FingerprintMismatch, match="SHA-1"):
        S.load_index(d, fp(cp))


def test_tat_luu_thi_khong_ghi_gi(cp, tmp_path):
    d = tmp_path / "s"
    r, info = S.load_or_build(cp, CORPUS, **P, store_dir=False)
    assert info["reason"] == "store_disabled" and not d.exists()
    assert r.retrieve("Hauptstadt", 3)


def test_khong_dung_pickle(cp, tmp_path):
    d = tmp_path / "s"
    S.save_index(BM25Retriever(CORPUS, **P), fp(cp), d)
    z = np.load(d / "index.npz", allow_pickle=False)                    # nạp được khi cấm pickle = chỉ có mảng số
    assert set(z.files) == {"terms", "offsets", "doc_idx", "weights"}
