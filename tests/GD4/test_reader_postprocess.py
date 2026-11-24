import numpy as np

from germanqa.reader.postprocess import best_answers
from germanqa.reader.qa_metrics import evaluate_answers, exact_match, f1_score, normalize

CONTEXT = "Der Hund, bellt laut."
#          0123456789...  token context: Der(0,3) Hund(4,8) ,(8,9) bellt(10,15) laut(16,20) .(20,21)
OFFS = [None, (0, 3), (4, 8), (8, 9), (10, 15), (16, 20), (20, 21), None]
EX = [{"id": "q", "question": "Wer bellt ?", "context": CONTEXT, "answers": [{"text": "Hund", "start": 4}]}]


def feat(offs=OFFS, example_index=0):
    return {"offsets": offs, "example_index": example_index}


def logits(n, **peaks):
    arr = np.full(n, -10.0)
    for i, v in peaks.items():
        arr[int(i[1:])] = v
    return arr


def test_cat_van_ban_tu_context_goc_giu_dau_cau_va_chu_hoa():
    s, e = logits(8, t2=5.0), logits(8, t4=5.0)
    out = best_answers(EX, [feat()], [s], [e])
    assert out["q"]["text"] == "Hund, bellt"               # nguyên văn, kể cả dấu phẩy
    assert (out["q"]["start"], out["q"]["end"]) == (4, 15)


def test_khong_chon_end_truoc_start():
    # cặp (start=4, end=2) có điểm 10 nhưng end đứng trước start nên bị loại; còn lại (2, 2) điểm 6
    s, e = logits(8, t4=5.0, t2=1.0), logits(8, t2=5.0)
    out = best_answers(EX, [feat()], [s], [e])
    assert out["q"]["text"] == "Hund"


def test_khong_chon_token_ngoai_context():
    s, e = logits(8, t0=9.0, t2=1.0), logits(8, t7=9.0, t2=1.0)       # CLS và SEP điểm cao nhưng không hợp lệ
    out = best_answers(EX, [feat()], [s], [e])
    assert out["q"]["text"] == "Hund"


def test_gioi_han_do_dai_dap_an():
    s, e = logits(8, t1=9.0, t2=0.0), logits(8, t6=9.0, t2=0.0)       # cặp tốt nhất dài 6 token
    assert best_answers(EX, [feat()], [s], [e], max_answer_tokens=6)["q"]["text"] == CONTEXT
    assert best_answers(EX, [feat()], [s], [e], max_answer_tokens=3)["q"]["text"] == "Der Hund"   # cặp (1, 2) điểm 9


def test_chon_cua_so_co_diem_cao_nhat():
    f1_, f2_ = feat(), feat()
    s1, e1 = logits(8, t2=3.0), logits(8, t2=3.0)
    s2, e2 = logits(8, t5=4.0), logits(8, t5=4.0)
    out = best_answers(EX, [f1_, f2_], [s1, s2], [e1, e2])
    assert out["q"]["text"] == "laut"


def test_minus_cls_dua_diem_cls_vao_so_sanh():
    # cửa sổ 1: điểm span 6 nhưng [CLS] cũng rất cao (mô hình nghĩ "đáp án không ở đây")
    s1, e1 = logits(8, t2=3.0, t0=8.0), logits(8, t2=3.0, t0=8.0)
    s2, e2 = logits(8, t5=2.0, t0=0.0), logits(8, t5=2.0, t0=0.0)
    feats = [feat(), feat()]
    assert best_answers(EX, feats, [s1, s2], [e1, e2], score_mode="raw")["q"]["text"] == "Hund"
    assert best_answers(EX, feats, [s1, s2], [e1, e2], score_mode="minus_cls")["q"]["text"] == "laut"


def test_khong_co_cap_hop_le_tra_chuoi_rong():
    offs = [None] * 8
    out = best_answers(EX, [feat(offs)], [logits(8)], [logits(8)])
    assert out["q"]["text"] == ""


def test_normalize_va_em():
    assert normalize("  Der  „Hund“, bellt! ") == "der hund bellt"
    assert exact_match("Der Hund.", "der hund") == 1.0
    assert normalize("die Katze") == "die katze"              # mạo từ KHÔNG bị bỏ


def test_f1_tinh_tay():
    # pred = "der große hund" (3 từ), ref = "der hund" (2 từ): chung 2 -> P=2/3, R=1 -> F1=0.8
    assert abs(f1_score("der große hund", "der hund") - 0.8) < 1e-12
    assert f1_score("katze", "hund") == 0.0
    assert f1_score("", "") == 1.0


def test_nhieu_dap_an_tham_chieu_lay_diem_cao_nhat():
    ex = [{"id": "q", "answers": [{"text": "der Hund"}, {"text": "Hund"}, {"text": "ein Tier"}]}]
    m = evaluate_answers({"q": {"text": "Hund"}}, ex)
    assert m["em"] == 1.0 and m["f1"] == 1.0
    m = evaluate_answers({"q": {"text": "Katze"}}, ex)
    assert m["em"] == 0.0 and m["f1"] == 0.0