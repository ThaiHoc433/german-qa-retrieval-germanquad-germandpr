"""Phân loại câu trả lời sai (GĐ7): sai Ở ĐÂU trong pipeline, và sai THẾ NÀO so với đáp án đúng.

Chỉ phân loại câu có EM = 0 (khớp chữ). "Sai" ở đây không có nghĩa đáp án vô lý: nhiều câu F1 cao chỉ lệch ranh giới."""
from germanqa.reader.qa_metrics import f1_score, normalize

STAGES = ("retrieval_failure", "selection_failure", "wrong_passage", "wrong_span_gold_passage")
RELATIONS = ("empty", "inside_gold", "contains_gold", "partial_overlap", "disjoint")

# Ba loại lỗi của plan GĐ7 ("Có ở top-5, đáp án sai -> Reader/extraction hoặc answer selection failure").
# Code tách loại thứ ba làm hai để biết reader sai vì chọn nhầm đoạn hay vì sai ranh giới span.
COARSE_STAGE = {"retrieval_failure": "retrieval", "selection_failure": "selection",
                "wrong_passage": "reader", "wrong_span_gold_passage": "reader"}


def coarse_stage(stage_name):
    """4 stage chi tiết -> 3 loại của plan: retrieval / selection / reader."""
    return COARSE_STAGE[stage_name]


def stage(retrieved_ids, selected_pid, gold_pids, reader_top_k=5, retrieval_top_k=20):
    """Lỗi xảy ra ở tầng nào, theo plan GĐ7:
    đoạn đúng vắng ở top-20 -> retrieval_failure; có ở top-20 nhưng vắng ở top-5 -> selection_failure;
    có ở top-5: reader chọn đoạn khác -> wrong_passage, chọn đúng đoạn nhưng sai span -> wrong_span_gold_passage."""
    gold = set(gold_pids)
    if not gold & set(retrieved_ids[:retrieval_top_k]):
        return "retrieval_failure"
    if not gold & set(retrieved_ids[:reader_top_k]):
        return "selection_failure"
    return "wrong_span_gold_passage" if selected_pid in gold else "wrong_passage"


def _contains(big, small):
    """small là dãy từ liên tiếp của big (đã chuẩn hóa), không tính khớp giữa chừng một từ."""
    return small != "" and f" {small} " in f" {big} "


def relation(pred, refs):
    """Quan hệ giữa đáp án dự đoán và tham chiếu GẦN NHẤT (F1 cao nhất)."""
    p = normalize(pred)
    if not p:
        return "empty"
    best = max(refs, key=lambda r: f1_score(pred, r))
    r = normalize(best)
    if _contains(r, p):
        return "inside_gold"            # dự đoán nằm trọn trong đáp án đúng (thiếu chữ)
    if _contains(p, r):
        return "contains_gold"          # dự đoán chứa đáp án đúng (thừa chữ)
    return "partial_overlap" if f1_score(pred, best) > 0 else "disjoint"


ARTICLES = ("der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "eines")


def article_only(pred, refs):
    """Dự đoán và một tham chiếu chỉ khác nhau ở mạo từ đứng đầu (ví dụ 'der Radsport' và 'Radsport').
    EM của dự án KHÔNG bỏ mạo từ (đã chốt ở GĐ4), nên các câu này tính là sai; hàm này đếm chúng để thấy EM nghiêm khắc cỡ nào."""
    p = normalize(pred)
    for r in map(normalize, refs):
        if p and r and p != r and any(p == f"{a} {r}" or r == f"{a} {p}" for a in ARTICLES):
            return True
    return False


def answer_in_passage(refs, passage_text):
    """Có tham chiếu nào xuất hiện (sau chuẩn hóa) trong đoạn này không."""
    t = normalize(passage_text)
    return any(_contains(t, normalize(r)) for r in refs)


def classify(row, gold_pids, refs, passages, reader_top_k=5):
    """row: một dòng file dự đoán (answer, passage_id, retrieved). Trả về dict mô tả lỗi.
    answer_in_selected_passage: đoạn reader chọn có chứa một tham chiếu -> có thể đáp án cũng đúng (qrels chưa đủ).
    answer_in_other_retrieved: tham chiếu xuất hiện trong một đoạn truy hồi KHÔNG nằm trong qrels."""
    ids = [h["passage_id"] for h in row["retrieved"]]
    gold = set(gold_pids)
    others = [pid for pid in ids if pid not in gold]
    return {
        "stage": stage(ids, row["passage_id"], gold, reader_top_k),
        "relation": relation(row["answer"], refs),
        "f1": max(f1_score(row["answer"], r) for r in refs),
        "article_only": article_only(row["answer"], refs),
        "answer_in_selected_passage": bool(row["passage_id"]) and row["passage_id"] not in gold
                                       and answer_in_passage(refs, passages[row["passage_id"]]),
        "answer_in_other_retrieved": any(answer_in_passage(refs, passages[pid]) for pid in others),
    }


# ---- Nhãn nguyên nhân gốc cho mẫu đọc tay (GĐ7 "lấy mẫu -> nhóm -> đếm", thêm 2026-11-13) ----
# Bộ mã ban đầu; người đọc gán một nhãn chính (primary) và tùy chọn nhãn phụ (secondary, đa nhãn).
ROOT_CAUSES = {
    "lexical_mismatch": "câu hỏi và đoạn đúng diễn đạt bằng từ khác (đồng nghĩa, diễn đạt lại)",
    "compound_or_inflection": "từ ghép hoặc biến cách tiếng Đức làm lệch từ khóa",
    "same_article_wrong_passage": "tìm/chọn đúng bài nhưng sai đoạn",
    "answer_in_unlabeled_passage": "đoạn ngoài qrels cũng chứa đáp án đúng (qrels thiếu)",
    "distractor_passage": "đoạn khác có ứng viên trông hợp lý hơn, reader bị đánh lừa",
    "span_too_long": "dự đoán chứa đáp án nhưng thừa chữ",
    "span_too_short": "dự đoán nằm trong đáp án nhưng thiếu chữ",
    "article_or_format": "chỉ khác mạo từ, dấu câu hoặc cách viết",
    "wrong_entity": "chọn sai thực thể/con số trong đúng đoạn",
    "question_ambiguous_or_label_noise": "câu hỏi mơ hồ hoặc nhãn vàng có vấn đề",
    "other": "không thuộc nhóm nào, ghi rõ ở note",
}


def wilson_ci(k, n, z=1.96):
    """Khoảng Wilson 95% cho tỉ lệ k/n: dùng cho tỉ lệ trong MẪU 40 câu, không phải tỉ lệ cả benchmark."""
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def validate_labels(labels):
    """Liệt kê câu chưa gán nhãn chính và nhãn ngoài bộ mã. labels: [{question_id, stage, primary, secondary, note}]."""
    unlabeled = [x["question_id"] for x in labels if not x.get("primary")]
    unknown = [x["question_id"] for x in labels
               if (x.get("primary") and x["primary"] not in ROOT_CAUSES)
               or any(s not in ROOT_CAUSES for s in x.get("secondary") or [])]
    bad_stage = [x["question_id"] for x in labels if x.get("stage") not in STAGES]
    return {"unlabeled": unlabeled, "unknown_cause": unknown, "bad_stage": bad_stage}


def count_labels(labels):
    """Đếm nhãn theo stage: số câu, số và tỉ lệ nhãn chính (kèm khoảng Wilson), số câu có nhãn (chính hoặc phụ).
    Từ chối khi còn câu chưa gán hoặc nhãn lạ: một bảng đếm thiếu nhãn sẽ lệch về các câu dễ gán."""
    problems = validate_labels(labels)
    if problems["unlabeled"]:
        raise ValueError(f"còn {len(problems['unlabeled'])} câu chưa gán nhãn chính, ví dụ {problems['unlabeled'][:3]}")
    if problems["unknown_cause"] or problems["bad_stage"]:
        raise ValueError(f"nhãn hoặc stage không hợp lệ: {problems}")
    out = {}
    for s in STAGES:
        rows = [x for x in labels if x["stage"] == s]
        if not rows:
            continue
        n = len(rows)
        prim = {c: sum(x["primary"] == c for x in rows) for c in ROOT_CAUSES}
        out[s] = {"n": n, "coarse": COARSE_STAGE[s],
                  "primary": {c: {"count": k, "share": k / n, "wilson95": wilson_ci(k, n)} for c, k in prim.items() if k},
                  "any": {c: k for c in ROOT_CAUSES
                          if (k := sum(x["primary"] == c or c in (x.get("secondary") or []) for x in rows))}}
    return out
