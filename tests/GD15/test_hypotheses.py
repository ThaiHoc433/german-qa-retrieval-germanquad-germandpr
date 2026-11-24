import pytest

from germanqa.config import RESULTS_DIR, ROOT
from germanqa.data.io_utils import file_sha1, read_json
from germanqa.evaluation import gd15_hypotheses as H
from germanqa.experiment_rules import read_ledger


def test_closer_to_unseen_tinh_tay():
    assert H.closer_to_unseen(0.03, 0.026, 0.103) is True        # gần unseen
    assert H.closer_to_unseen(0.09, 0.026, 0.103) is False       # gần toàn dev
    assert H.closer_to_unseen(0.0645, 0.026, 0.103) is False     # đúng giữa -> không tính là ủng hộ
    assert H.delta(0.34, 0.24) == pytest.approx(0.10)


def test_judge_ha_ba_ket_luan():
    full = {"B0": 0.236, "B1": 0.339, "H0": 0.329}
    unseen = {"B0": 0.171, "B1": 0.197, "H0": 0.209}
    near_unseen = {"B0": 0.20, "B1": 0.23, "H0": 0.24}            # chênh nhỏ như dev-unseen
    near_full = {"B0": 0.20, "B1": 0.30, "H0": 0.29}              # chênh lớn như toàn dev
    mixed = {"B0": 0.20, "B1": 0.23, "H0": 0.29}
    assert H.judge_ha(near_unseen, unseen, full)["verdict"] == "ủng hộ"
    assert H.judge_ha(near_full, unseen, full)["verdict"] == "bác bỏ"
    r = H.judge_ha(mixed, unseen, full)
    assert r["verdict"] == "hỗn hợp" and r["B1"] is True and r["H0"] is False


def test_judge_hb_va_hc():
    assert H.judge_hb(0.0)["holds"] is True and H.judge_hb(-0.001)["holds"] is False
    assert H.judge_hc((0.07, 0.03, 0.11), (-0.004, -0.04, 0.03))["verdict"] == "ủng hộ"
    assert H.judge_hc((0.07, -0.01, 0.11), (0.05, 0.02, 0.08))["verdict"] == "bác bỏ"
    assert H.judge_hc((0.07, 0.03, 0.11), (0.05, 0.02, 0.08))["verdict"] == "hỗn hợp"       # Hit@5 đúng nhưng EM tăng thật


def test_tep_ghi_truoc_ton_tai_va_so_lieu_khop_nguon():
    doc = read_json(RESULTS_DIR / "15_hypotheses.json")
    for k, f in {"B0": "results/eval_dev_20261103T163556Z_cb4749bc.json", "B1": "results/eval_dev_dense_20261107T072837Z_b97b5c1e.json",
                 "H0": "results/eval_dev_hybrid_20261107T135423Z_598ddf6f.json"}.items():
        assert doc["inputs_sha1"][k] == file_sha1(ROOT / f), k
    assert doc["code_sha1"]["src/germanqa/evaluation/gd15_hypotheses.py"] == file_sha1(ROOT / "src/germanqa/evaluation/gd15_hypotheses.py")
    full, unseen = doc["reference_dev_em"]["full_dev_1176"], doc["reference_dev_em"]["dev_unseen_234"]
    assert doc["reference_delta_vs_B0"]["full_dev"]["B1"] == pytest.approx(full["B1"] - full["B0"], abs=1e-6)
    assert doc["reference_delta_vs_B0"]["dev_unseen"]["H0"] == pytest.approx(unseen["H0"] - unseen["B0"], abs=1e-6)
    assert set(doc["hypotheses"]) == {"HA", "HB", "HC"}


def test_ghi_truoc_khi_chay_test_cho_b1_h0_h1():
    """Lời ghi trước chỉ có nghĩa nếu lúc ghi chưa có lần tính chỉ số test cho B1/H0/H1. Test này nhắc: nếu sổ đã có lần như vậy,
    tệp 15_hypotheses.json phải có created_at SỚM HƠN lần đầu tiên."""
    doc = read_json(RESULTS_DIR / "15_hypotheses.json")
    runs = [r for r in read_ledger() if r["kind"] == "metric_run" and set(r.get("codes", [])) & {"B1", "H0", "H1"}]
    for r in runs:
        assert doc["created_at"].replace(" ", "T") < r["ts"], "giả thuyết phải được ghi trước khi chạy test"


def test_hoa_chinh_xac_khong_tinh_la_ung_ho():
    """0,5 cách 0,25 và 0,75 đúng bằng nhau (biểu diễn chính xác trong float): hòa phải là Sai."""
    assert H.closer_to_unseen(0.5, 0.25, 0.75) is False
    assert H.closer_to_unseen(0.49, 0.25, 0.75) is True


def test_hc_khoang_tin_cay_em_nam_han_duoi_khong_la_chua_0():
    r = H.judge_hc((0.07, 0.03, 0.11), (-0.05, -0.08, -0.02))        # EM GIẢM thật: khoảng không chứa 0
    assert r["em_ci_contains_zero"] is False and r["verdict"] == "hỗn hợp"
    r2 = H.judge_hc((0.07, 0.03, 0.11), (0.0, -0.03, 0.03))
    assert r2["em_ci_contains_zero"] is True and r2["verdict"] == "ủng hộ"
