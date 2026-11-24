"""GĐ11.2 (sửa D4): dấu vân tay BM25 và chỉ mục lưu ra tệp.

Vấn đề: BM25 dựng lại từ corpus mỗi lần khởi động (khoảng 16-24 s), và không có gì chứng minh chỉ mục đang dùng
ứng với corpus, tham số và tokenizer của lần đo B0. Sửa hai lớp:
  1. DẤU VÂN TAY = SHA-1 của mọi thứ làm đổi kết quả BM25: nội dung corpus, k1, b, use_stopwords, NỘI DUNG danh sách
     stopword, kết quả tách từ của bộ câu mẫu cố định, và điểm BM25 của một corpus nhỏ cố định (bắt được sửa công thức).
  2. Chỉ mục lưu ở artifacts/bm25/ (index.npz không dùng pickle + meta.json). Nạp thì so dấu vân tay và SHA-1 của
     tệp: lệch -> báo lỗi (strict) hoặc dựng lại và ghi đè (mặc định)."""
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from germanqa.config import ROOT
from germanqa.data.io_utils import file_sha1
from germanqa.retrieval.bm25 import BM25, BM25Retriever
from germanqa.retrieval.tokenizer import GERMAN_STOPWORDS, tokenize

log = logging.getLogger("germanqa.bm25_store")

FORMAT_VERSION = 1
DEFAULT_DIR = ROOT / "artifacts" / "bm25"

# Bộ câu mẫu cố định: có umlaut, ß, chữ hoa, số, dấu câu, từ dừng. Đổi chuỗi này = đổi dấu vân tay, nên đừng sửa.
PROBE_TEXTS = (
    "Wer war der erste Bundeskanzler der Bundesrepublik Deutschland?",
    "Größe, Straße & Fußball: 3,5 km² in Österreich!",
    "Die Änderungen der Verfassung (1990) betrafen §§ 23 und 146.",
    "ÜBER den Wolken muß die Freiheit wohl grenzenlos sein",
)
# Corpus nhỏ cố định để đo điểm: bắt được thay đổi công thức idf/tf/độ dài trong BM25.fit
PROBE_DOCS = (
    ("p1", "der Rhein fließt durch Köln und Bonn"),
    ("p2", "Bonn war die Hauptstadt der Bundesrepublik"),
    ("p3", "Köln hat einen Dom und viele Kirchen"),
    ("p4", "die Hauptstadt von Österreich ist Wien"),
)
PROBE_QUERY = "Hauptstadt Bonn Köln Dom"


class FingerprintMismatch(RuntimeError):
    pass


def _sha1_json(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def stopwords_sha1():
    return _sha1_json(sorted(GERMAN_STOPWORDS))


def tokenization_probe_sha1(use_stopwords):
    sw = GERMAN_STOPWORDS if use_stopwords else None
    return _sha1_json([tokenize(t, sw) for t in PROBE_TEXTS])


def scoring_probe_sha1(k1, b, use_stopwords):
    sw = GERMAN_STOPWORDS if use_stopwords else None
    bm = BM25(k1, b).fit([d for d, _ in PROBE_DOCS], [tokenize(t, sw) for _, t in PROBE_DOCS])
    return _sha1_json([round(x, 12) for x in bm.scores(tokenize(PROBE_QUERY, sw)).tolist()])


def compute_fingerprint(corpus_sha1, k1, b, use_stopwords):
    parts = {"format": FORMAT_VERSION, "corpus_sha1": corpus_sha1, "k1": float(k1), "b": float(b),
             "use_stopwords": bool(use_stopwords), "stopwords_sha1": stopwords_sha1(),
             "tokenization_probe_sha1": tokenization_probe_sha1(use_stopwords),
             "scoring_probe_sha1": scoring_probe_sha1(k1, b, use_stopwords)}
    return {"sha1": _sha1_json(parts), "parts": parts}


def save_index(retriever, fingerprint, store_dir):
    """Ghi index.npz (mảng số, không pickle) và meta.json. Trả về đường dẫn thư mục."""
    d = Path(store_dir)
    d.mkdir(parents=True, exist_ok=True)
    idx = retriever.index
    terms = list(idx.weights)
    if any("\n" in t for t in terms):
        raise ValueError("từ chứa ký tự xuống dòng: không lưu được theo định dạng này")
    lens = np.fromiter((len(idx.weights[t][0]) for t in terms), dtype=np.int64, count=len(terms))
    offsets = np.zeros(len(terms) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    doc_idx = np.concatenate([idx.weights[t][0] for t in terms]) if terms else np.zeros(0, dtype=np.int64)
    wts = np.concatenate([idx.weights[t][1] for t in terms]) if terms else np.zeros(0, dtype=np.float64)
    blob = np.frombuffer("\n".join(terms).encode("utf-8"), dtype=np.uint8)
    np.savez(d / "index.npz", terms=blob, offsets=offsets, doc_idx=doc_idx.astype(np.int64), weights=wts.astype(np.float64))
    meta = {"format": FORMAT_VERSION, "fingerprint": fingerprint, "k1": idx.k1, "b": idx.b,
            "use_stopwords": retriever.stopwords is not None, "n_docs": idx.n, "n_terms": len(terms),
            "doc_ids": idx.doc_ids, "index_npz_sha1": file_sha1(d / "index.npz"),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(d / "meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False)
    return d


def load_index(store_dir, expected_fingerprint):
    """Nạp chỉ mục. Ném FingerprintMismatch nếu dấu vân tay hoặc SHA-1 tệp không khớp; FileNotFoundError nếu chưa có."""
    d = Path(store_dir)
    with open(d / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["fingerprint"]["sha1"] != expected_fingerprint["sha1"]:
        diff = [k for k in expected_fingerprint["parts"]
                if meta["fingerprint"]["parts"].get(k) != expected_fingerprint["parts"][k]]
        raise FingerprintMismatch("chỉ mục đã lưu không khớp dấu vân tay hiện tại; khác ở: " + ", ".join(diff or ["?"]))
    if file_sha1(d / "index.npz") != meta["index_npz_sha1"]:
        raise FingerprintMismatch("index.npz bị đổi hoặc hỏng so với meta.json (SHA-1 không khớp)")
    z = np.load(d / "index.npz", allow_pickle=False)
    terms = bytes(z["terms"]).decode("utf-8").split("\n") if len(z["terms"]) else []
    off, di, w = z["offsets"], z["doc_idx"], z["weights"]
    weights = {t: (di[off[i]:off[i + 1]], w[off[i]:off[i + 1]]) for i, t in enumerate(terms)}
    index = BM25.from_parts(meta["k1"], meta["b"], meta["doc_ids"], weights)
    return BM25Retriever.from_index(index, meta["use_stopwords"])


def load_or_build(corpus_path, corpus, k1, b, use_stopwords, store_dir=None, strict=False):
    """Trả về (retriever, info). info: {source: 'loaded'|'built', reason, seconds, fingerprint}.
    strict=True (dành cho dịch vụ/Docker, GĐ10 và GĐ18): lệch dấu vân tay thì ném lỗi thay vì âm thầm dựng lại.
    store_dir=False: không dùng tệp lưu."""
    t0 = time.perf_counter()
    fp = compute_fingerprint(file_sha1(corpus_path), k1, b, use_stopwords)
    if store_dir is False:
        r = BM25Retriever(corpus, k1=k1, b=b, use_stopwords=use_stopwords)
        return r, {"source": "built", "reason": "store_disabled", "seconds": time.perf_counter() - t0, "fingerprint": fp["sha1"]}
    d = Path(store_dir) if store_dir is not None else DEFAULT_DIR
    reason = "no_store"
    try:
        r = load_index(d, fp)
        return r, {"source": "loaded", "reason": "fingerprint_ok", "seconds": time.perf_counter() - t0, "fingerprint": fp["sha1"]}
    except FileNotFoundError:
        if strict:
            raise
    except FingerprintMismatch as exc:
        if strict:
            raise
        reason = "fingerprint_mismatch"
        log.warning("BM25: %s; dựng lại", exc)
    r = BM25Retriever(corpus, k1=k1, b=b, use_stopwords=use_stopwords)
    try:
        save_index(r, fp, d)
    except OSError as exc:                  # volume chỉ đọc (Docker) hoặc hết chỗ: vẫn phục vụ được, chỉ không lưu
        log.warning("BM25: không lưu được chỉ mục (%s)", type(exc).__name__)
        reason += "+save_failed"
    return r, {"source": "built", "reason": reason, "seconds": time.perf_counter() - t0, "fingerprint": fp["sha1"]}
