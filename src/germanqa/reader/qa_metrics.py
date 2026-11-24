"""EM và F1 cho QA trích xuất (GĐ4). Nhiều đáp án tham chiếu: lấy điểm cao nhất."""
import re
import string
from collections import Counter

_PUNCT = set(string.punctuation) | set("„“”‚‘’»«–—…")


def normalize(text):
    """Chữ thường, bỏ dấu câu, gộp khoảng trắng. KHÔNG bỏ mạo từ (der, die, das là từ thật)."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred, ref):
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def f1_score(pred, ref):
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return 1.0 if p == r else 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def score_answers(predictions, examples):
    """Điểm TỪNG câu: danh sách [(em, f1)] theo đúng thứ tự examples, lấy cao nhất trên các tham chiếu.
    Bootstrap (GĐ7) cần điểm từng câu, không chỉ trung bình."""
    scores = []
    for ex in examples:
        pred = predictions.get(ex["id"], {"text": ""})["text"]
        refs = [a["text"] for a in ex["answers"]]
        scores.append((max(exact_match(pred, r) for r in refs), max(f1_score(pred, r) for r in refs)))
    return scores


def evaluate_answers(predictions, examples):
    """predictions = {id: {"text": ...}}; ví dụ có thể có nhiều đáp án tham chiếu."""
    scores = score_answers(predictions, examples)
    return {"queries": len(examples), "em": sum(s[0] for s in scores) / len(scores),
            "f1": sum(s[1] for s in scores) / len(scores)}

