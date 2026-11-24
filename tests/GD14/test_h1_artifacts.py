import copy

import pytest

from germanqa.config import RESULTS_DIR, ROOT
from germanqa.data.io_utils import file_sha1, read_json
from germanqa.evaluation.fusion_select import require_prereg
from germanqa.evaluation.manifest import load_manifest, verify_manifest, verify_manifest_detail
from germanqa.experiment import load_config


def test_prereg_con_khop_tep_that_va_model_da_khoa():
    pre = require_prereg(RESULTS_DIR / "14_prereg.json", ROOT)
    assert pre["max_configs_tried"] == 6 and len(pre["models"]) == 3
    assert pre["splits"]["test"] == "KHÔNG mở"


def test_cau_hinh_rerank_khop_ket_qua_chon_va_prereg():
    sel = read_json(RESULTS_DIR / "14_selection.json")
    cfg = load_config("rerank")["rerank"]
    pre = read_json(RESULTS_DIR / "14_prereg.json")
    assert (cfg["model"], cfg["overflow"]) == (sel["chosen"]["model"], sel["chosen"]["overflow"]) == ("bge_m3", "windows")
    assert sel["configs_tried"] == pre["max_configs_tried"] == len(sel["table"])
    assert sel["split"] == "train-select" and sel["n"] == 1989
    assert cfg["commit"] == pre["models"]["bge_m3"]["commit"]
    for name, h in pre["models"]["bge_m3"]["files_sha1"].items():             # bản sao trong models/ đúng từng bit với bản đã khóa
        assert file_sha1(ROOT / cfg["checkpoint"] / name) == h, name


def test_chon_theo_quy_tac_lai_tu_bang_ket_qua():
    from germanqa.rerank.select import choose_config
    sel = read_json(RESULTS_DIR / "14_selection.json")
    assert choose_config(sel["table"]) == sel["chosen"]
    assert sel["adopted"] is True and sel["chosen_vs_h0"]["hit@5"]["lo"] > 0 and sel["chosen_vs_h0"]["hit@5_never_group"]["lo"] >= -0.01


def test_xac_nhan_dev_unseen_mot_lan_va_khop_ket_luan():
    c = read_json(RESULTS_DIR / "14_confirmation.json")
    assert c["split"] == "dev-unseen" and c["n"] == 234
    assert c["config"]["model"] == "bge_m3"
    assert (c["paired_h1_minus_h0"]["hit@5"]["lo"] > 0) == c["verdict"].startswith("H1 hơn H0")
    assert c["paired_h1_minus_h0"]["hit@20"]["diff"] == 0.0                     # reranker không đổi tập 20 đoạn


def test_h1_dau_cuoi_hoi_quy_h0_khong_doi_va_khop_xac_nhan():
    r = read_json(RESULTS_DIR / "14f_H1_dev_unseen.json")
    assert r["regression_H0_vs_saved_GD13"] == {"answer": 0, "passage_id": 0, "reader_score": 0.0, "retrieved_ids": 0}
    c = read_json(RESULTS_DIR / "14_confirmation.json")
    assert r["H1"]["evidence_in_top5"] == pytest.approx(c["h1"]["hit@5"]) and r["H0"]["evidence_in_top5"] == pytest.approx(c["h0"]["hit@5"])
    x = read_json(RESULTS_DIR / "14i_crosscheck.json")
    assert x["rerank_score_abs_diff_on_same_pool"]["max"] == 0.0 and x["same_pool"] + x["pool_order_differs_same_set"] + x["pool_set_differs"] == 234


def test_manifest_H1_kiem_duoc_nhieu_sha1_va_phat_hien_doi():
    man = load_manifest("H1")
    bad, n = verify_manifest_detail(man)
    assert bad == [] and n >= 30, n
    assert man["code"] == "H1" and man["rerank"]["config"] == load_config("rerank")["rerank"]
    assert man["reference_dev_run"]["split"] == "dev-unseen" and man["reference_dev_run"]["end_to_end_first_ref"]["queries"] == 234
    assert man["reader"]["d5_max_answer_tokens"]["effective_max_answer_tokens"] == 80
    t = copy.deepcopy(man)
    t["rerank"]["sha1"]["model.safetensors"] = "3" * 40
    t["rerank"]["results_sha1"]["14_prereg.json"] = "4" * 40
    msgs = verify_manifest(t)
    assert any("rerank/model.safetensors" in m for m in msgs) and any("14_prereg.json" in m for m in msgs)


def test_ma_cau_hinh_H1_khac_H0():
    assert load_manifest("H1")["config_id"] != load_manifest("H0")["config_id"]
    assert load_manifest("H1")["config_id"].startswith("H1_")
