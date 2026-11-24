"""Test /experiments (GĐ8): chỉ đọc, bỏ file hỏng, không lộ đường dẫn máy."""
import json

import pytest
from fastapi.testclient import TestClient

from germanqa.api.app import create_app
from germanqa.api.experiments import list_experiments, summarize_file
from germanqa.api.settings import ApiSettings

HIT = {"hit@1": 0.3, "hit@5": 0.6, "hit@20": 0.8, "mrr@10": 0.45, "queries": 10}
CELL = {"em": 0.25, "f1": 0.4, "em_ci": [0.2, 0.3], "queries": 10}


def put(d, name, obj):
    (d / name).write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")


@pytest.fixture
def results(tmp_path):
    put(tmp_path, "eval_dev_20261103T163556Z_cb4749bc.json", {
        "experiment_id": "eval_dev_20261103T163556Z_cb4749bc", "finished_at": "2026-11-03 23:35:56",
        "config": {"split": "dev", "checkpoint": "reader_x", "corpus_version": "abc123"},
        "metrics": {"end_to_end": {"first_ref": CELL, "all_refs": CELL}, "retrieval": HIT,
                    "timing": {"total_time": {"p50_ms": 57.4}}}})
    put(tmp_path, "eval_dev_dense_20261104T121120Z_57716bd8.json", {
        "experiment_id": "eval_dev_dense_20261104T121120Z_57716bd8",
        "config": {"split": "dev", "retriever": "dense", "checkpoint": "reader_x"},
        "metrics": {"dev": {"dense": {"first_ref": {"em": 0.34, "f1": 0.51}}, "bm25": {"first_ref": CELL},
                            "retrieval_dense": HIT, "retrieval_bm25": HIT},
                    "dev_unseen": {"dense": {"first_ref": {"em": 0.2, "f1": 0.42}}, "retrieval_dense": HIT}}})
    put(tmp_path, "reader_20261101T213758Z_96717e35.json", {
        "experiment_id": "reader_20261101T213758Z_96717e35", "checkpoint": "D:\\Project\\secret\\models\\reader_x",
        "config": {}, "metrics": {"history": [{"epoch": 1, "dev": {"em": 0.40, "f1": 0.62}}, {"epoch": 2, "dev": {"em": 0.45, "f1": 0.67, "queries": 9}}]}})
    put(tmp_path, "bm25_20261031T203636Z_feb3f3b7.json", {
        "experiment_id": "bm25_20261031T203636Z_feb3f3b7", "config": {},
        "metrics": {"dev": {"metrics": HIT}, "test": {"metrics": HIT}}})
    put(tmp_path, "hong.json", "{ not json")
    put(tmp_path, "mang.json", "[1, 2]")
    put(tmp_path, "thieu_id.json", {"metrics": {}})
    put(tmp_path, "dev_unseen_question_ids.json", ["q_1", "q_2"])        # file không phải thí nghiệm nằm chung thư mục
    return tmp_path


def test_liet_ke_moi_nhat_truoc_va_dem_file_bi_bo_qua(results):
    items, skipped = list_experiments(results)
    assert [i["kind"] for i in items] == ["eval_dev_dense", "eval_dev", "reader", "bm25"]
    assert skipped == 4


def test_dense_dev_co_ca_dev_unseen_va_canh_bao_nhiem(results):
    it = summarize_file(results / "eval_dev_dense_20261104T121120Z_57716bd8.json")
    assert set(it["metrics"]) == {"dev", "dev_unseen"} and it["metrics"]["dev_unseen"]["em"] == 0.2
    assert it["metrics"]["dev"]["bm25_em"] == 0.25 and it["retriever"] == "dense"
    assert it["caveats"] and "dev_unseen" in it["caveats"][0]


def test_so_do_chi_gom_so_va_khong_co_khoang_tin_cay_tho(results):
    m = summarize_file(results / "eval_dev_20261103T163556Z_cb4749bc.json")["metrics"]["dev"]
    assert m == {"em": 0.25, "f1": 0.4, "hit@1": 0.3, "hit@5": 0.6, "hit@20": 0.8, "mrr@10": 0.45, "total_p50_ms": 57.4}


def test_khoang_tin_cay_em_f1_chi_co_o_file_eval(results):
    it = summarize_file(results / "eval_dev_20261103T163556Z_cb4749bc.json")
    assert it["intervals"] == {"dev": {"em": [0.2, 0.3]}}                    # fixture chỉ có em_ci
    assert summarize_file(results / "bm25_20261031T203636Z_feb3f3b7.json")["intervals"] == {}
    bad = results / "eval_dev_20269999T000000Z_00000000.json"
    put(results, bad.name, {"experiment_id": bad.stem, "config": {"split": "dev"},
                            "metrics": {"end_to_end": {"first_ref": {"em": 0.1, "em_ci": [0.1, "x"], "f1_ci": [1, 2, 3]}}}})
    assert summarize_file(bad)["intervals"] == {}                            # khoảng sai hình dạng thì bỏ, không làm lỗi


def test_reader_lay_em_f1_dev_o_epoch_cuoi(results):
    it = summarize_file(results / "reader_20261101T213758Z_96717e35.json")
    assert it["metrics"] == {"dev_last_epoch": {"em": 0.45, "f1": 0.67}}


def test_khong_lo_duong_dan_may(results):
    blob = json.dumps(list_experiments(results)[0])
    assert "secret" not in blob and "D:\\" not in blob
    assert summarize_file(results / "reader_20261101T213758Z_96717e35.json")["checkpoint"] == "reader_x"


def test_created_at_la_utc_tu_ma_thi_nghiem_nen_sap_xep_nhat_quan(results):
    it = summarize_file(results / "eval_dev_20261103T163556Z_cb4749bc.json")
    assert it["created_at"] == "2026-11-03T16:35:56+00:00"          # không dùng finished_at (giờ địa phương)


def test_thu_muc_khong_ton_tai_thi_rong(tmp_path):
    assert list_experiments(tmp_path / "khong_co") == ([], 0)


def test_endpoint_va_loc_theo_kind_va_khong_ghi_gi(results):
    before = sorted(p.name for p in results.iterdir())
    s = ApiSettings(results_dir=str(results))
    with TestClient(create_app(s, lambda _: ({}, {}))) as c:
        r = c.get("/api/v1/experiments")
        assert r.status_code == 200 and r.json()["count"] == 4 and r.json()["skipped"] == 4
        assert [i["kind"] for i in c.get("/api/v1/experiments?kind=bm25").json()["items"]] == ["bm25"]
    assert sorted(p.name for p in results.iterdir()) == before            # API không ghi gì
