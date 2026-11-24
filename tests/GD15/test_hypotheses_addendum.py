from germanqa.config import RESULTS_DIR, ROOT
from germanqa.data.io_utils import file_sha1, read_json
from germanqa.experiment_rules import read_ledger


def test_phu_luc_chot_loai_em_truoc_khi_chay_test():
    add = read_json(RESULTS_DIR / "15_hypotheses_addendum.json")
    base = read_json(RESULTS_DIR / "15_hypotheses.json")
    assert add["amends_sha1"] == file_sha1(RESULTS_DIR / "15_hypotheses.json")        # phụ lục gắn đúng với bản ghi trước
    assert add["decision"]["primary_reference_mode"] == "first_ref"
    assert add["created_at"] >= base["created_at"]
    assert set(add["decision"]["applies_to"]) >= {"HA", "HB"}


def test_phu_luc_ghi_truoc_moi_lan_chay_test_cho_b1_h0_h1():
    add = read_json(RESULTS_DIR / "15_hypotheses_addendum.json")
    runs = [r for r in read_ledger() if r["kind"] == "metric_run" and set(r.get("codes", [])) & {"B1", "H0", "H1"}]
    for r in runs:
        assert add["created_at"].replace(" ", "T") < r["ts"], "phụ lục phải được ghi trước khi chạy test"


def test_phu_luc_khong_doi_ba_gia_thuyet():
    base = read_json(RESULTS_DIR / "15_hypotheses.json")
    assert set(base["hypotheses"]) == {"HA", "HB", "HC"}
    assert base["code_sha1"]["src/germanqa/evaluation/gd15_hypotheses.py"] == file_sha1(ROOT / "src/germanqa/evaluation/gd15_hypotheses.py")
