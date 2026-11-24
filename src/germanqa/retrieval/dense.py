"""Dense retriever (GĐ6): cặp DPR (context encoder + question encoder) và FAISS flat inner-product.

Offline: mọi đoạn -> context encoder -> vector 768 chiều -> lưu cùng danh sách passage_id và manifest.
Online : câu hỏi -> question encoder -> vector -> tích vô hướng với mọi vector đoạn -> top-k.
Giao diện retrieve(question, top_k) -> [{passage_id, score, rank}] giống BM25Retriever, nên QAPipeline không đổi."""
import time
from pathlib import Path

import numpy as np

from germanqa.data.io_utils import file_sha1, read_json, write_json
from germanqa.tokenization import load_bert_tokenizer

PASSAGE_INPUTS = ("title_text", "text")     # title_text: cặp (title, text); text: chỉ văn bản đoạn (đã chứa tiêu đề ở dòng đầu)


def snapshot_dir(repo_id):
    """Thư mục snapshot trong cache HuggingFace (tên thư mục chính là mã revision). Ưu tiên cache, không cần mạng."""
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(repo_id, "config.json", local_files_only=True)).parent
    except Exception:
        return Path(hf_hub_download(repo_id, "config.json")).parent


class DPREncoder:
    """Hai encoder DPR. Mỗi encoder chỉ nạp khi dùng lần đầu: mã hóa corpus không cần question encoder và ngược lại,
    nên GPU 4 GB không phải giữ cả hai."""

    def __init__(self, ctx_repo, question_repo, device=None, fp16=True):
        import torch
        self.torch = torch
        self.ctx_repo, self.question_repo = ctx_repo, question_repo
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self._ctx = self._question = None

    # ---- nạp mô hình ----
    def _load(self, cls_name, repo):
        import transformers
        d = snapshot_dir(repo)
        model = getattr(transformers, cls_name).from_pretrained(d).to(self.device).eval()
        return model, load_bert_tokenizer(d / "vocab.txt"), d.name

    @property
    def ctx(self):
        if self._ctx is None:
            self._ctx = self._load("DPRContextEncoder", self.ctx_repo)
        return self._ctx

    @property
    def question(self):
        if self._question is None:
            self._question = self._load("DPRQuestionEncoder", self.question_repo)
        return self._question

    def revisions(self):
        return {"ctx_encoder": {"repo": self.ctx_repo, "revision": snapshot_dir(self.ctx_repo).name},
                "question_encoder": {"repo": self.question_repo, "revision": snapshot_dir(self.question_repo).name}}

    # ---- mã hóa ----
    def _embed(self, bundle, first, second, max_length, batch_size, log_every=0):
        """Vector [CLS] (pooler_output của DPR) của từng đầu vào. Trả về (mảng float32 [n, d], mảng độ dài token)."""
        torch = self.torch
        model, tok, _ = bundle
        n = len(first)
        if n == 0:
            return np.zeros((0, model.config.hidden_size), dtype=np.float32), np.zeros(0, dtype=np.int64)
        # độ dài THẬT (chưa cắt) để biết bao nhiêu đầu vào bị cắt, và để xếp lô theo độ dài cho ít padding
        full = tok(first, second, add_special_tokens=True, truncation=False, verbose=False)["input_ids"] \
            if second is not None else tok(first, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        lengths = np.array([len(x) for x in full], dtype=np.int64)
        order = np.argsort(-lengths, kind="stable")
        out = np.zeros((n, model.config.hidden_size), dtype=np.float32)
        t0 = time.perf_counter()
        for b, s in enumerate(range(0, n, batch_size)):
            idx = order[s:s + batch_size]
            args = ([first[i] for i in idx], [second[i] for i in idx]) if second is not None else ([first[i] for i in idx],)
            enc = tok(*args, add_special_tokens=True, truncation="longest_first", max_length=max_length,
                      padding=True, return_tensors="pt").to(self.device)
            # Encoder này được huấn luyện với segment id TOÀN 0, kể cả khi đầu vào là cặp (title, text). Để token_type_ids
            # 0/1 của tokenizer làm đoạn đúng chỉ đứng hạng 1 ở ~46% thay vì ~76% (đo trên 300 cặp GermanDPR train).
            enc["token_type_ids"] = torch.zeros_like(enc["token_type_ids"])
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
                vec = model(**enc).pooler_output
            out[idx] = vec.float().cpu().numpy()
            if log_every and (b + 1) % log_every == 0:
                print(f"  {min(s + batch_size, n)}/{n}  {time.perf_counter() - t0:.0f}s", flush=True)
        return out, lengths

    def encode_questions(self, questions, max_length=64, batch_size=64):
        vecs, _ = self._embed(self.question, list(questions), None, max_length, batch_size)
        return vecs

    def encode_passages(self, titles, texts, passage_input="title_text", max_length=256, batch_size=32, log_every=100):
        """Trả về (vector [n, d], thống kê độ dài/cắt cụt)."""
        if passage_input not in PASSAGE_INPUTS:
            raise ValueError(f"passage_input phải là một trong {PASSAGE_INPUTS}")
        first, second = ([t or "" for t in titles], list(texts)) if passage_input == "title_text" else (list(texts), None)
        vecs, lengths = self._embed(self.ctx, first, second, max_length, batch_size, log_every)
        stats = {"max_length": max_length, "n_truncated": int((lengths > max_length).sum()),
                 "frac_truncated": float((lengths > max_length).mean()) if len(lengths) else 0.0,
                 "tokens_mean": float(lengths.mean()) if len(lengths) else 0.0,
                 "tokens_p95": float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
                 "tokens_max": int(lengths.max()) if len(lengths) else 0}
        return vecs, stats


def check_embeddings(vectors, passage_ids):
    """Kiểm tra bắt buộc trước khi dùng/lưu: số vector = số ID, ID không trùng, không NaN/Inf."""
    if vectors.ndim != 2 or vectors.shape[0] != len(passage_ids):
        raise ValueError(f"số vector {vectors.shape} không khớp số passage_id {len(passage_ids)}")
    if len(set(passage_ids)) != len(passage_ids):
        raise ValueError("passage_id bị trùng")
    if not np.isfinite(vectors).all():
        raise ValueError("vector có NaN hoặc Inf")


def save_artifacts(out_dir, vectors, passage_ids, manifest):
    """embeddings.npy + passage_ids.json + manifest.json. Thứ tự dòng của embeddings = thứ tự passage_ids."""
    import faiss
    check_embeddings(vectors, passage_ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", vectors)
    write_json(out_dir / "passage_ids.json", list(passage_ids))
    faiss.write_index(build_index(vectors), str(out_dir / "index.faiss"))
    write_json(out_dir / "manifest.json", manifest)


def load_artifacts(out_dir, corpus_path=None):
    """Đọc lại artifacts và kiểm tra. Có corpus_path thì so SHA-1 corpus: index cũ so với corpus mới sẽ bị từ chối."""
    out_dir = Path(out_dir)
    manifest = read_json(out_dir / "manifest.json")
    ids = read_json(out_dir / "passage_ids.json")
    vectors = np.load(out_dir / "embeddings.npy")
    check_embeddings(vectors, ids)
    if corpus_path is not None and file_sha1(corpus_path) != manifest["corpus_sha1"]:
        raise ValueError("corpus đã đổi so với lúc mã hóa: phải mã hóa lại (index cũ không còn đúng)")
    return vectors, ids, manifest


def check_manifest_compat(manifest, cfg, revisions=None):
    """GĐ12.1: chỉ mục được mã hóa bằng encoder/định dạng nào thì lúc chạy phải dùng đúng cái đó.
    So manifest với cấu hình hiện tại: repo + revision của HAI encoder, passage_field, passage_input, similarity.
    Lệch thì báo lỗi, không dùng chỉ mục một cách im lặng. (passage_max_length là thông tin lúc mã hóa, không so.)"""
    problems = []
    for key in ("passage_field", "passage_input", "similarity"):
        if manifest.get(key) != cfg.get(key):
            problems.append(f"{key}: chỉ mục {manifest.get(key)!r} ≠ cấu hình {cfg.get(key)!r}")
    for role in ("ctx_encoder", "question_encoder"):
        m = manifest.get(role) or {}
        if m.get("repo") != cfg.get(role):
            problems.append(f"{role}: chỉ mục {m.get('repo')!r} ≠ cấu hình {cfg.get(role)!r}")
        if revisions is not None and (revisions.get(role) or {}).get("revision") != m.get("revision"):
            problems.append(f"{role} revision: chỉ mục {m.get('revision')!r} ≠ đang dùng {(revisions.get(role) or {}).get('revision')!r}")
    if problems:
        raise ValueError("chỉ mục dense không tương thích với cấu hình hiện tại, phải mã hóa lại: " + "; ".join(problems))


def build_index(vectors):
    """Flat inner-product: tìm CHÍNH XÁC, không xấp xỉ. DPR được huấn luyện bằng tích vô hướng nên không chuẩn hóa."""
    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def exact_topk(query_vectors, vectors, k):
    """Mốc đối chiếu: tính thẳng Q @ P.T bằng numpy, không qua FAISS. Trả về (điểm, chỉ số) giảm dần."""
    scores = np.asarray(query_vectors, dtype=np.float32) @ np.asarray(vectors, dtype=np.float32).T
    k = min(k, scores.shape[1])
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(scores, idx, axis=1), idx


class DenseRetriever:
    def __init__(self, encoder, vectors, passage_ids, question_max_length=64):
        check_embeddings(vectors, passage_ids)
        self.encoder, self.passage_ids = encoder, list(passage_ids)
        self.index = build_index(vectors)
        self.question_max_length = question_max_length
        self.last_timing = {"encode": 0.0, "search": 0.0}      # giây, của lần retrieve/retrieve_many gần nhất

    @classmethod
    def from_dir(cls, artifact_dir, encoder, corpus_path=None, question_max_length=64, expected_cfg=None):
        vectors, ids, manifest = load_artifacts(artifact_dir, corpus_path)
        if expected_cfg is not None:          # GĐ12.1: so manifest với cấu hình hiện tại
            check_manifest_compat(manifest, expected_cfg, encoder.revisions() if hasattr(encoder, "revisions") else None)
        return cls(encoder, vectors, ids, question_max_length)

    def _rows(self, scores, idx):
        return [{"passage_id": self.passage_ids[i], "score": float(s), "rank": r}
                for r, (s, i) in enumerate(zip(scores, idx), start=1) if i >= 0]

    def retrieve_many(self, questions, top_k=20, batch_size=64):
        t = time.perf_counter()
        q = self.encoder.encode_questions(questions, self.question_max_length, batch_size)
        t_enc = time.perf_counter() - t
        t = time.perf_counter()
        scores, idx = self.index.search(q, min(top_k, self.index.ntotal))
        self.last_timing = {"encode": t_enc, "search": time.perf_counter() - t}
        return [self._rows(s, i) for s, i in zip(scores, idx)]

    def retrieve(self, question, top_k=20):
        return self.retrieve_many([question], top_k)[0]
