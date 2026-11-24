import math

from germanqa.retrieval.bm25 import BM25, BM25Retriever
from germanqa.retrieval.tokenizer import GERMAN_STOPWORDS, tokenize

CORPUS = [
    {"passage_id": "d1", "retrieval_text": "der hund bellt"},
    {"passage_id": "d2", "retrieval_text": "die katze schläft"},
    {"passage_id": "d3", "retrieval_text": "der hund und die katze"},
    {"passage_id": "d4", "retrieval_text": "hund"},
    {"passage_id": "d5", "retrieval_text": "hund"},
]


def test_tokenize_giu_umlaut_va_eszett():
    assert tokenize("Schläft über Straße!") == ["schläft", "über", "straße"]


def test_tokenize_stopword():
    assert tokenize("Der Hund", GERMAN_STOPWORDS) == ["hund"]


def test_doan_ngan_hon_xep_truoc_khi_tf_bang_nhau():
    r = BM25Retriever(CORPUS, k1=1.5, b=0.75)
    ids = [h["passage_id"] for h in r.retrieve("hund", 10)]
    assert ids.index("d1") < ids.index("d3")           # cả hai chứa "hund" 1 lần, d1 ngắn hơn
    assert ids[0] in ("d4", "d5")                      # đoạn chỉ có đúng từ đó ngắn nhất


def test_dong_diem_on_dinh_theo_thu_tu_doan():
    r = BM25Retriever(CORPUS)
    ids = [h["passage_id"] for h in r.retrieve("hund", 10)]
    assert ids.index("d4") < ids.index("d5")


def test_chi_tra_doan_co_diem_duong_va_top_k_lon_hon_kho():
    r = BM25Retriever(CORPUS)
    hits = r.retrieve("katze", 100)
    assert [h["passage_id"] for h in hits] == ["d2", "d3"]
    assert all(h["score"] > 0 for h in hits)
    assert [h["rank"] for h in hits] == [1, 2]


def test_truy_van_rong_hoac_la():
    r = BM25Retriever(CORPUS)
    assert r.retrieve("", 5) == []
    assert r.retrieve("xyzxyz", 5) == []
    assert r.retrieve("hund", 0) == []


def test_diem_tinh_tay_mot_tu():
    # 2 đoạn: "a b" và "a". Truy vấn "b": df=1, N=2 -> idf = ln(1 + (2-1+0.5)/(1+0.5)) = ln 2
    idx = BM25(k1=1.0, b=0.0).fit(["x", "y"], [["a", "b"], ["a"]])
    expected = math.log(2) * 1 * (1.0 + 1) / (1 + 1.0)       # b=0: không chuẩn hóa độ dài, tf=1
    assert abs(idx.scores(["b"])[0] - expected) < 1e-12
    assert idx.scores(["b"])[1] == 0.0