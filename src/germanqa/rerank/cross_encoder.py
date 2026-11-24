"""Cross-encoder reranker (GĐ14): chấm lại từng cặp (câu hỏi, đoạn) rồi xếp lại danh sách ứng viên.

Quy ước (ghi sẵn để không hiểu sai điểm):
  * Điểm = LOGIT liên quan của model, KHÔNG phải xác suất và KHÔNG so được với điểm BM25/dense/RRF.
      - model một đầu ra (num_labels = 1): điểm = logit đó.
      - model nhiều nhãn: điểm = logit(nhãn liên quan) - logsumexp(các nhãn còn lại)  (log-odds). Hai nhãn thì
        bằng logit[liên quan] - logit[không]. Nhãn liên quan do người gọi chỉ định (relevant_label), không đoán.
  * Sắp xếp giảm dần theo điểm; hòa điểm thì giữ thứ tự cũ (hạng RRF nhỏ hơn đứng trước). Không trộn với điểm khác.
  * Đoạn dài hơn max_length: overflow = 'truncate' (cắt cuối đoạn) hoặc 'windows' (chia cửa sổ chồng lấn, điểm của
    đoạn = điểm lớn nhất trong các cửa sổ). Số cửa sổ được trả về để tính chi phí.
Reranker không thêm đoạn nào ngoài danh sách đưa vào, nên không thể cứu đoạn đúng vắng khỏi tập ứng viên."""
import hashlib
import json
import time
from pathlib import Path

MAX_QUERY_TOKENS = 64        # khớp MAX_QUESTION_TOKENS của pipeline: câu hỏi dài hơn thì cắt, chừa chỗ cho đoạn
CODE_VERSION = "rerank-1"    # đổi khi đổi cách tính điểm, để dấu vân tay (và bộ nhớ đệm điểm) không dùng nhầm


def combine_logits(logits, relevant_label):
    """logits: tensor [n, num_labels] -> điểm [n] theo quy ước ở đầu file."""
    import torch
    if logits.shape[1] == 1:
        return logits[:, 0]
    if relevant_label is None:
        raise ValueError("model có nhiều nhãn: phải chỉ định relevant_label (chỉ số nhãn 'liên quan')")
    if not 0 <= relevant_label < logits.shape[1]:
        raise ValueError("relevant_label ngoài phạm vi nhãn")
    others = torch.cat([logits[:, :relevant_label], logits[:, relevant_label + 1:]], dim=1)
    return logits[:, relevant_label] - torch.logsumexp(others, dim=1)


def stable_rank(rows, scores):
    """rows: danh sách ứng viên theo thứ tự cũ; scores: điểm tương ứng. Trả về danh sách mới sắp giảm dần theo điểm,
    hòa điểm giữ thứ tự cũ. Mỗi phần tử có thêm rank_before, rank (hạng mới, từ 1), rerank_score."""
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], i))
    out = []
    for new_rank, i in enumerate(order, 1):
        r = dict(rows[i])
        r.update(rank_before=i + 1, rank=new_rank, rerank_score=float(scores[i]))
        out.append(r)
    return out


class CrossEncoderReranker:
    def __init__(self, checkpoint, max_length=512, batch_size=16, fp16=True, relevant_label=None,
                 overflow="truncate", stride=128, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if overflow not in ("truncate", "windows"):
            raise ValueError("overflow phải là 'truncate' hoặc 'windows'")
        self.checkpoint = Path(checkpoint)
        self.max_length, self.batch_size, self.overflow, self.stride = max_length, batch_size, overflow, stride
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = bool(fp16) and self.device == "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("cần tokenizer nhanh (để ánh xạ cửa sổ về đoạn)")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint)
        self.model.eval().to(self.device)
        if self.fp16:
            self.model.half()
        self.num_labels = self.model.config.num_labels
        if self.num_labels > 1 and relevant_label is None:
            raise ValueError(f"model có {self.num_labels} nhãn: phải chỉ định relevant_label")
        self.relevant_label = relevant_label
        self.last_stats = {"pairs": 0, "windows": 0, "seconds": 0.0}

    # ---- chuẩn bị đầu vào ----
    def _cut_query(self, query):
        enc = self.tokenizer(query, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        return query if len(offs) <= MAX_QUERY_TOKENS else query[:offs[MAX_QUERY_TOKENS - 1][1]]

    def encode_pairs(self, query, texts):
        """Trả về (encoding, owner): owner[j] = chỉ số đoạn mà cửa sổ j thuộc về."""
        q = self._cut_query(query)
        kw = dict(truncation="only_second", max_length=self.max_length, padding=False)
        if self.overflow == "windows":
            enc = self.tokenizer([q] * len(texts), texts, return_overflowing_tokens=True, stride=self.stride, **kw)
            owner = list(enc["overflow_to_sample_mapping"])
        else:
            enc = self.tokenizer([q] * len(texts), texts, **kw)
            owner = list(range(len(texts)))
        return enc, owner

    def token_lengths(self, query, texts):
        """Độ dài (token) của cặp (câu hỏi đã cắt, đoạn) nếu KHÔNG cắt gì; dùng để kiểm cắt token (GĐ14.3)."""
        q = self._cut_query(query)
        enc = self.tokenizer([q] * len(texts), texts, add_special_tokens=True, truncation=False, padding=False)
        return [len(x) for x in enc["input_ids"]]

    # ---- chấm điểm ----
    def score_detail(self, query, texts):
        """Chấm từng đoạn. Trả về {'score': điểm theo overflow đã chọn, 'first': điểm cửa sổ ĐẦU (= điểm khi chỉ cắt cuối
        đoạn), 'windows': số cửa sổ của từng đoạn}. Chế độ 'truncate' thì score == first, windows == 1."""
        import torch
        t0 = time.perf_counter()
        if not texts:
            self.last_stats = {"pairs": 0, "windows": 0, "seconds": 0.0}
            return {"score": [], "first": [], "windows": []}
        enc, owner = self.encode_pairs(query, texts)
        ids = enc["input_ids"]
        n_win = len(ids)
        order = sorted(range(n_win), key=lambda j: len(ids[j]))         # gom cặp cùng độ dài: ít đệm hơn
        win_scores = [0.0] * n_win
        keys = ("input_ids", "attention_mask") + (("token_type_ids",) if "token_type_ids" in enc else ())
        with torch.inference_mode():
            for s in range(0, n_win, self.batch_size):
                idx = order[s:s + self.batch_size]
                batch = self.tokenizer.pad({k: [enc[k][j] for j in idx] for k in keys}, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits.float()
                for j, v in zip(idx, combine_logits(logits, self.relevant_label).cpu().tolist()):
                    win_scores[j] = v
        best = [float("-inf")] * len(texts)
        first = [None] * len(texts)
        count = [0] * len(texts)
        for j, o in enumerate(owner):                                       # các cửa sổ của một đoạn liền nhau, cửa sổ đầu trước
            best[o] = max(best[o], win_scores[j])
            count[o] += 1
            if first[o] is None:
                first[o] = win_scores[j]
        self.last_stats = {"pairs": len(texts), "windows": n_win, "seconds": time.perf_counter() - t0}
        return {"score": best, "first": first, "windows": count}

    def score_texts(self, query, texts):
        """Điểm (float) của từng đoạn theo thứ tự đầu vào. Cửa sổ của cùng một đoạn lấy điểm lớn nhất."""
        return self.score_detail(query, texts)["score"]

    def rerank(self, query, candidates, top_n=None):
        """candidates: [{'passage_id', 'text', ...}] theo thứ tự cũ. Trả về danh sách đã xếp lại (có rank_before,
        rank, rerank_score), cắt còn top_n nếu có. Không đổi/loại bớt ứng viên nào ngoài việc cắt top_n."""
        scores = self.score_texts(query, [c["text"] for c in candidates])
        ranked = stable_rank(candidates, scores)
        return ranked if top_n is None else ranked[:top_n]

    # ---- định danh ----
    def fingerprint(self):
        """SHA-1 của cấu hình chấm điểm + nội dung trọng số/tokenizer; đổi bất kỳ thứ gì thì điểm đệm không dùng lại."""
        from germanqa.data.io_utils import file_sha1
        files = {p.name: file_sha1(p) for p in sorted(self.checkpoint.iterdir()) if p.is_file()
                 and (p.suffix in (".safetensors", ".bin", ".model") or p.name in ("config.json", "tokenizer.json", "vocab.txt",
                                                                                   "tokenizer_config.json"))}
        cfg = {"files": files, "max_length": self.max_length, "overflow": self.overflow,
               "stride": self.stride if self.overflow == "windows" else None, "fp16": self.fp16,
               "relevant_label": self.relevant_label, "max_query_tokens": MAX_QUERY_TOKENS, "code": CODE_VERSION}
        return {"sha1": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest(), **cfg}
