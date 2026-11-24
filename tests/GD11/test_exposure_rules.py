import json
import re
from pathlib import Path

import pytest

from germanqa import experiment_rules as R

CALL = re.compile("(guard_or_exit|log_test_metadata)" + chr(92) + "(")
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_khong_co_co_thi_dung_va_khong_ghi_so(tmp_path):
    led = tmp_path / "l.jsonl"
    with pytest.raises(R.TestAccessError):
        R.require_test_access("x.py", "thử", confirm=False, ledger_path=led)
    assert not led.exists()                            # chưa mở gì thì không có gì để ghi


def test_co_co_thi_ghi_mot_dong(tmp_path):
    led = tmp_path / "l.jsonl"
    e = R.require_test_access("x.py", "ma trận cuối", confirm=True, files=["qa_test.jsonl"], codes=["B0"], ledger_path=led)
    rows = R.read_ledger(led)
    assert len(rows) == 1 and rows[0]["script"] == "x.py" and rows[0]["codes"] == ["B0"] and rows[0]["source"] == "live"
    assert e["confirmed"] is True


def test_so_chi_them_dong_khong_ghi_de(tmp_path):
    led = tmp_path / "l.jsonl"
    for i in range(3):
        R.require_test_access("x.py", f"lần {i}", True, ledger_path=led)
    first = led.read_text(encoding="utf-8").splitlines()[0]
    R.require_test_access("x.py", "lần 4", True, ledger_path=led)
    lines = led.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and lines[0] == first       # dòng cũ nguyên vẹn
    assert not any(n in dir(R) for n in ("delete_ledger", "edit_ledger", "rewrite_ledger"))


def test_ma_la_va_loai_la_bi_tu_choi(tmp_path):
    with pytest.raises(ValueError):
        R.require_test_access("x.py", "thử", True, codes=["Z9"], ledger_path=tmp_path / "l.jsonl")
    with pytest.raises(ValueError):
        R.append_ledger({"script": "x", "purpose": "y", "kind": "khac"}, tmp_path / "l.jsonl")
    with pytest.raises(ValueError):
        R.append_ledger({"script": "", "purpose": "y", "kind": "metric_run"}, tmp_path / "l.jsonl")


def test_metadata_khong_can_co_nhung_van_ghi(tmp_path):
    led = tmp_path / "l.jsonl"
    R.log_test_metadata("06a.py", "kiểm tra rò rỉ", ledger_path=led)
    r = R.read_ledger(led)[0]
    assert r["kind"] == "metadata_only" and r["confirmed"] is False
    with pytest.raises(ValueError):                    # không lách bằng cách khai metadata_only qua cổng có cờ
        R.require_test_access("x.py", "y", True, kind="metadata_only", ledger_path=led)


def test_guard_or_exit_thoat_gon_khong_traceback(tmp_path):
    with pytest.raises(SystemExit) as ei:
        R.guard_or_exit("x.py", "y", False, ledger_path=tmp_path / "l.jsonl")
    assert "--confirm-test" in str(ei.value)


def test_doan_cong_khai_sinh_tu_so(tmp_path):
    led = tmp_path / "l.jsonl"
    assert "chưa bị mở" in R.disclosure_text(led)
    R.append_ledger({"ts": "2026-11-04T00:05:42", "source": "reconstructed", "script": "07a", "kind": "metric_run",
                     "purpose": "B0 test"}, led)
    R.log_test_metadata("06a.py", "rò rỉ", ledger_path=led)
    t = R.disclosure_text(led)
    assert "2 lần truy cập" in t and "1 dòng được dựng lại" in t and "2026-11-04 07a" in t


def test_ma_thi_nghiem_doi_theo_cau_hinh():
    a, b = R.baseline_id("B0", {"k1": 0.7}), R.baseline_id("B0", {"k1": 0.9})
    assert a.startswith("B0_") and a != b and a == R.baseline_id("B0", {"k1": 0.7})
    with pytest.raises(ValueError):
        R.baseline_id("B9", {})


def test_moc_khong_bi_ghi_de(tmp_path):
    p = tmp_path / "B0.json"
    R.write_once(p, {"v": 1})
    with pytest.raises(FileExistsError):
        R.write_once(p, {"v": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"v": 1}


def test_moi_script_cham_test_deu_di_qua_rao_chan():
    """Quét tĩnh: script nào nhắc tới tệp/dự đoán test phải import guard_or_exit hoặc log_test_metadata.
    02_preprocess TẠO ra tệp test (không phải truy cập kết quả) nên được miễn; GD11 chỉ nhắc tới test để KHÔNG mở."""
    pat = re.compile(r"qa_test|qrels_test|test_predictions|\"test\"\)|split == \"test\"|--test|--confirm-test")
    # 11a tự ghi sổ (băm tệp KẾT QUẢ cũ, không mở dữ liệu test); 11c chỉ nhắc "không đọc qa_test" trong docstring
    exempt = {"02_preprocess.py", "00_init_project.py", "01_download_data.py", "11a_backfill_ledger.py", "11c_select_set.py"}
    offenders = []
    for p in SCRIPTS.rglob("*.py"):
        if p.name in exempt:
            continue
        src = p.read_text(encoding="utf-8")
        if pat.search(src) and not CALL.search(src):     # phải có LỜI GỌI, không chỉ import
            offenders.append(str(p.relative_to(SCRIPTS)))
    assert offenders == [], f"script nhắc tới test nhưng không qua rào chắn: {offenders}"
