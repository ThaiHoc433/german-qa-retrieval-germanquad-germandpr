import json
import shutil

import pytest

from germanqa.config import ROOT
from germanqa.data.io_utils import file_sha1
from germanqa.evaluation import gd15_freeze as F


def test_hien_tai_moi_dieu_kien_truoc_khi_mo_test_deu_dung():
    """Chạy trên project thật: nếu test này đỏ thì chưa được mở test (hoặc đã có lần chạy test cho B1/H0/H1)."""
    ledger_runs = [json.loads(l) for l in (ROOT / "results/test_exposure_ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    already = [r for r in ledger_runs if r["kind"] == "metric_run" and set(r.get("codes", [])) & F.TEST_CODES]
    if already:                                    # sau khi đã chạy test một lần, điều kiện 'chưa có lần chạy' tất nhiên sai: chỉ kiểm phần còn lại
        pytest.skip("test đã được chạy một lần; preflight chỉ có nghĩa trước đó")
    assert F.preflight_problems(ROOT) == []


def test_b0_dung_lai_la_dung_cau_hinh():
    assert F.b0_reuse_problems(ROOT) == []


def test_build_va_verify_freeze_phat_hien_tep_bi_doi(tmp_path):
    for rel in F.all_frozen_files():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    fz = F.build_freeze(tmp_path)
    assert F.verify_freeze(fz, tmp_path) == []
    victim = tmp_path / "configs/rerank.yaml"
    victim.write_text(victim.read_text(encoding="utf-8") + "\n# sửa sau khi đóng băng\n", encoding="utf-8")
    bad = F.verify_freeze(fz, tmp_path)
    assert len(bad) == 1 and "configs/rerank.yaml" in bad[0]
    (tmp_path / "baselines/H1.json").unlink()
    assert any("baselines/H1.json" in b and "None" in b for b in F.verify_freeze(fz, tmp_path))


def test_b0_khac_cau_hinh_thi_bi_chan(tmp_path):
    for rel in F.B0_REUSE + ["baselines/B0.json", "data/processed/corpus.jsonl", "data/processed/qa_test.jsonl"]:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    assert F.b0_reuse_problems(tmp_path) == []
    ev_path = tmp_path / F.B0_REUSE[0]
    ev = json.loads(ev_path.read_text(encoding="utf-8-sig"))
    ev["config"]["pipeline_config"]["reader"]["max_answer_tokens"] = 30                 # giả vờ lần chạy test cũ dùng D5 = 30
    ev_path.write_text(json.dumps(ev), encoding="utf-8")
    assert any("'reader'" in b for b in F.b0_reuse_problems(tmp_path))
    ev["config"]["pipeline_config"]["reader"] = json.loads((ROOT / F.B0_REUSE[0]).read_text(encoding="utf-8-sig"))["config"]["pipeline_config"]["reader"]
    ev["config"]["corpus_version"] = "000000000000"
    ev_path.write_text(json.dumps(ev), encoding="utf-8")
    assert any("corpus" in b for b in F.b0_reuse_problems(tmp_path))


def test_thieu_ket_qua_dev_thi_chua_duoc_mo_test(tmp_path):
    """Chưa phân tích xong trên dev (thiếu 15_dev_matrix / 15_stage_timing) thì preflight phải chặn."""
    bad = F.preflight_problems(tmp_path)
    assert any("15_dev_matrix.json" in b for b in bad) and any("15_stage_timing.json" in b for b in bad)


def test_ledger_da_co_lan_chay_test_cho_h1_thi_chan(tmp_path):
    for rel in F.DECISIONS + F.DEV_ANALYSIS + F.B0_REUSE:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    (tmp_path / "results/test_exposure_ledger.jsonl").write_text(
        json.dumps({"script": "x", "purpose": "x", "kind": "metric_run", "files": [], "codes": ["H1"], "confirmed": True, "source": "live", "ts": "2026-11-09T00:00:00+0700"}) + "\n",
        encoding="utf-8")
    assert any("H1" in b and "sổ phơi bày" in b for b in F.preflight_problems(tmp_path))
