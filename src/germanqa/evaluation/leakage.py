"""Kiểm tra rò rỉ giữa dữ liệu huấn luyện của encoder dense (GermanDPR train) và dev/test của dự án (GĐ6).

Encoder deepset/gbert-base-germandpr-* đã học trên GermanDPR train. Dev của dự án được tách từ GermanQuAD train,
mà GermanDPR train cũng sinh ra từ đó, nên nhiều câu hỏi dev là câu hỏi encoder đã thấy lúc huấn luyện.
Rò rỉ phía CÂU HỎI mới là vấn đề: corpus cố định, đoạn văn encoder đã thấy cũng là đoạn nằm trong index lúc triển khai."""
import re


def normalize_question(text):
    """Chuẩn hóa nhẹ để so khớp: bỏ khác biệt khoảng trắng và hoa/thường. Không bắt được câu diễn đạt lại."""
    return re.sub(r"\s+", " ", text).strip().lower()


def seen_question_flags(questions, seen_sets):
    """questions: [{'question_id','question'}]; seen_sets: {tên: tập câu hỏi đã chuẩn hóa}.
    Trả về {question_id: {tên: True/False}}."""
    out = {}
    for q in questions:
        n = normalize_question(q["question"])
        out[q["question_id"]] = {name: n in s for name, s in seen_sets.items()}
    return out


def summarize(flags):
    """Đếm số câu hỏi theo từng tập mà encoder đã thấy."""
    names = sorted({k for f in flags.values() for k in f})
    return {"questions": len(flags), **{f"in_{n}": sum(f[n] for f in flags.values()) for n in names}}


def unseen_ids(flags, names=("dpr_train",)):
    """question_id của các câu hỏi KHÔNG nằm trong bất kỳ tập huấn luyện nào trong `names`, theo thứ tự ổn định."""
    return sorted(qid for qid, f in flags.items() if not any(f[n] for n in names))
