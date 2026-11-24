"""Tập chọn (GĐ12.0): các câu hỏi train + dev mà encoder dense CHƯA thấy lúc huấn luyện."""
from germanqa.evaluation.leakage import normalize_question


def build_select_set(qa_by_split, dpr_train_questions):
    """qa_by_split: {'train': [...], 'dev': [...]} với phần tử có 'question_id' và 'question'.
    dpr_train_questions: các câu hỏi GermanDPR train (chưa chuẩn hóa). Trả về dict có ID đã sắp xếp."""
    seen = {normalize_question(q) for q in dpr_train_questions}
    out = {}
    for split in ("train", "dev"):
        out[split] = sorted(q["question_id"] for q in qa_by_split[split] if normalize_question(q["question"]) not in seen)
    out["counts"] = {"train": len(out["train"]), "dev": len(out["dev"]), "total": len(out["train"]) + len(out["dev"])}
    out["note"] = ("Chỉ số truy hồi: dùng train+dev. EM/F1: chỉ dev. Không bao giờ đưa các ID này vào huấn luyện (GĐ16). "
                   "Khớp theo chuẩn hóa chữ nên đây là cận trên của số câu chưa thấy.")
    return out
