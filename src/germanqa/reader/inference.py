
"""Reader dùng khi suy luận (GĐ5.1): nạp checkpoint MỘT lần, đọc nhiều (câu hỏi, đoạn văn).

GĐ4 viết sẵn từng bước (cửa sổ -> logit -> span). Lớp này chỉ ghép các bước đó lại, không đổi phép tính nào."""
import torch

from germanqa.reader.features import prepare_eval_features
from germanqa.reader.model_io import get_model, get_tokenizer
from germanqa.reader.postprocess import best_answers
from germanqa.reader.trainer import predict_logits


class Reader:
    def __init__(self, checkpoint, max_length=384, stride=128, n_best=20, max_answer_tokens=80,
                 score_mode="minus_cls", batch_size=32, fp16=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_tokenizer(checkpoint)
        self.model = get_model(checkpoint).to(self.device).eval()
        self.max_length, self.stride, self.n_best = max_length, stride, n_best
        self.max_answer_tokens, self.score_mode = max_answer_tokens, score_mode
        self.batch_size, self.fp16 = batch_size, fp16

    def logits(self, examples):
        """Cắt cửa sổ và chạy mô hình cho danh sách ví dụ {id, question, context, ...}.
        Trả về (feats, start_logits, end_logits). Tách riêng để GĐ4 chạy mô hình một lần rồi thử nhiều cách hậu xử lý."""
        feats = prepare_eval_features(examples, self.tokenizer, self.max_length, self.stride)
        s, e = predict_logits(self.model, feats, self.batch_size, self.device, self.fp16)
        return feats, s, e

    def decode(self, examples, feats, s, e, max_answer_tokens=None, score_mode=None):
        return best_answers(examples, feats, s, e, self.n_best,
                            self.max_answer_tokens if max_answer_tokens is None else max_answer_tokens,
                            score_mode or self.score_mode)

    def read(self, question, passages):
        """passages = [{"passage_id", "text"}, ...]. Trả về, theo đúng thứ tự đầu vào, span tốt nhất của TỪNG đoạn:
        [{"passage_id", "text", "start", "end", "score"}]; đoạn không có span hợp lệ có start = -1, score = -inf."""
        if not passages:
            return []
        examples = [{"id": p["passage_id"], "question": question, "context": p["text"], "answers": []}
                    for p in passages]
        feats, s, e = self.logits(examples)
        best = self.decode(examples, feats, s, e)
        return [{"passage_id": p["passage_id"], **best[p["passage_id"]]} for p in passages]
