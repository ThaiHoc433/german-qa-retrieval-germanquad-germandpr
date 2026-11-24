"""GĐ11.6 (bổ sung): mọi mã B0/B1/H0/H1 phải dùng CHUNG một evaluator và một quy tắc chuẩn hóa chữ.
Ba lớp chặn: (1) số đã lưu tái lập được bằng evaluator hiện tại; (2) không có bản sao evaluator ở chỗ khác; (3) quy tắc chuẩn hóa không đổi."""
import ast
import json
from pathlib import Path

import pytest

from germanqa.config import PROCESSED_DIR, REPORT_DIR, ROOT, RESULTS_DIR
from germanqa.data.io_utils import read_jsonl
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import evaluate_answers, exact_match, f1_score, normalize, score_answers

EVAL_NAMES = {"normalize", "exact_match", "f1_score", "score_answers", "evaluate_answers"}


def _saved(path, *keys):
    d = json.loads((ROOT / path).read_text(encoding="utf-8-sig"))["metrics"]
    for k in keys:
        d = d[k]
    return d["first_ref"]


# ---------- (1) số đã lưu tái lập được ----------
@pytest.mark.parametrize("name,pred_file,eval_file,keys", [
    ("B0", "pipeline_dev_predictions.jsonl", "results/eval_dev_20261103T163556Z_cb4749bc.json", ("end_to_end",)),
    ("B1", "pipeline_dev_dense_predictions.jsonl", "results/eval_dev_dense_20261107T072837Z_b97b5c1e.json", ("dev", "dense")),
    ("H0", "pipeline_dev_hybrid_predictions.jsonl", "results/eval_dev_hybrid_20261107T135423Z_598ddf6f.json", ("dev", "hybrid")),
])
def test_chay_lai_evaluator_ra_dung_so_da_luu(name, pred_file, eval_file, keys):
    """Chấm lại dự đoán đã lưu bằng evaluator hiện tại phải ra đúng EM/F1 trong tệp kết quả. Đổi evaluator hay quy tắc chuẩn hóa -> test này vỡ."""
    exs = load_examples(PROCESSED_DIR, "dev")
    preds = {r["question_id"]: {"text": r["answer"]} for r in read_jsonl(REPORT_DIR / pred_file)}
    got = evaluate_answers(preds, exs)
    want = _saved(eval_file, *keys)
    assert got["queries"] == want["queries"] == 1176
    assert got["em"] == pytest.approx(want["em"], abs=1e-12) and got["f1"] == pytest.approx(want["f1"], abs=1e-12), name


def test_evaluate_answers_va_score_answers_cung_mot_so():
    exs = load_examples(PROCESSED_DIR, "dev")[:200]
    preds = {r["question_id"]: {"text": r["answer"]} for r in read_jsonl(REPORT_DIR / "pipeline_dev_hybrid_predictions.jsonl")}
    per = score_answers(preds, exs)
    agg = evaluate_answers(preds, exs)
    assert agg["em"] == pytest.approx(sum(s[0] for s in per) / len(per)) and agg["f1"] == pytest.approx(sum(s[1] for s in per) / len(per))


# ---------- (2) không có bản sao evaluator ----------
def test_chi_qa_metrics_dinh_nghia_evaluator():
    """Không file nào khác trong src/ hoặc scripts/ được định nghĩa lại các hàm chấm điểm."""
    owner = ROOT / "src" / "germanqa" / "reader" / "qa_metrics.py"
    stray = []
    for base in (ROOT / "src", ROOT / "scripts"):
        for p in base.rglob("*.py"):
            if p == owner or "__pycache__" in p.parts:
                continue
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in EVAL_NAMES:
                    stray.append(f"{p.relative_to(ROOT)}:{n.name}")
    assert stray == [], stray


def test_script_danh_gia_import_tu_qa_metrics():
    """Mọi nơi gọi các hàm chấm điểm phải import chúng từ germanqa.reader.qa_metrics (không từ nơi khác)."""
    bad = []
    for base in (ROOT / "src", ROOT / "scripts"):
        for p in base.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(n, ast.ImportFrom) and any(a.name in EVAL_NAMES for a in n.names) and n.module != "germanqa.reader.qa_metrics":
                    bad.append(f"{p.relative_to(ROOT)} import từ {n.module}")
    assert bad == [], bad


# ---------- (3) quy tắc chuẩn hóa ----------
@pytest.mark.parametrize("pred,ref,em", [
    ("Der Rhein.", "der rhein", 1.0),                    # hoa/thường và dấu câu không tính
    ("  der   Rhein ", "der Rhein", 1.0),                # khoảng trắng gộp lại
    ("„Der Rhein“", "Der Rhein", 1.0),                   # ngoặc kép kiểu Đức
    ("»Der Rhein«", "Der Rhein", 1.0),
    ("Rhein", "der Rhein", 0.0),                         # KHÔNG bỏ mạo từ
    ("Straße", "Strasse", 0.0),                          # KHÔNG gộp ß/ss
    ("Köln", "Koln", 0.0),                               # KHÔNG bỏ dấu
    ("1993", "am 24. Mai 1993", 0.0),                    # lệch ranh giới là sai EM
    ("", "Rhein", 0.0),
])
def test_chuan_hoa_chu_khong_doi(pred, ref, em):
    assert exact_match(pred, ref) == em


def test_f1_theo_token_va_truong_hop_rong():
    assert f1_score("der Rhein", "Rhein") == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert f1_score("", "") == 1.0 and f1_score("", "Rhein") == 0.0 and f1_score("Elbe", "Rhein") == 0.0
    assert normalize("Aber, Hallo!  Welt.") == "aber hallo welt"


def test_nhieu_dap_an_tham_chieu_lay_diem_cao_nhat_va_thieu_du_doan_la_rong():
    exs = [{"id": "a", "answers": [{"text": "Bonn"}, {"text": "Köln"}]}, {"id": "b", "answers": [{"text": "Wien"}]}]
    sc = score_answers({"a": {"text": "köln"}}, exs)                  # 'b' không có dự đoán -> chuỗi rỗng
    assert sc[0] == (1.0, 1.0) and sc[1] == (0.0, 0.0)
