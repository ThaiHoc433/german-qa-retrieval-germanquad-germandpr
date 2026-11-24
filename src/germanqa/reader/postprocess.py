
"""Từ logit start/end của các cửa sổ ra một câu trả lời cho mỗi câu hỏi (GĐ4.4)."""
from collections import defaultdict

import numpy as np


def best_answers(examples, feats, start_logits, end_logits, n_best=20, max_answer_tokens=30,
                 score_mode="raw"):
    """Với mỗi ví dụ, thử mọi cặp (start, end) trong top-n_best của từng cửa sổ và giữ cặp tốt nhất.

    Cặp hợp lệ: cả hai nằm trong context, end không trước start, độ dài <= max_answer_tokens.
    score_mode: "raw"       -> điểm = start_logit + end_logit
                "minus_cls" -> trừ điểm [CLS] của cửa sổ đó (điểm 'không có đáp án ở cửa sổ này')
    Văn bản trả về được CẮT từ context gốc bằng offset, không ghép từ token.
    """
    windows = defaultdict(list)
    for fi, f in enumerate(feats):
        windows[f["example_index"]].append(fi)

    out = {}
    for ei, ex in enumerate(examples):
        best = {"text": "", "start": -1, "end": -1, "score": float("-inf")}
        for fi in windows[ei]:
            s_log, e_log, offs = start_logits[fi], end_logits[fi], feats[fi]["offsets"]
            n = len(offs)
            bias = (s_log[0] + e_log[0]) if score_mode == "minus_cls" else 0.0
            for si in np.argsort(s_log[:n])[::-1][:n_best]:
                for ej in np.argsort(e_log[:n])[::-1][:n_best]:
                    if offs[si] is None or offs[ej] is None:
                        continue
                    if ej < si or ej - si + 1 > max_answer_tokens:
                        continue
                    score = float(s_log[si] + e_log[ej] - bias)
                    if score > best["score"]:
                        a, b = offs[si][0], offs[ej][1]
                        best = {"text": ex["context"][a:b], "start": a, "end": b, "score": score}
        out[ex["id"]] = best
    return out
