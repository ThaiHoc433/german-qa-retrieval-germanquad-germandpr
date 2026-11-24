"""Trang Số đo (GĐ11.5 + quyết định C): mã B0/B1/H0/H1, ma trận từ baselines/*.json, nhánh hybrid, all_refs, gộp lần chạy trùng.
Chỉ đọc, không lộ đường dẫn máy, thiếu dữ liệu thì bỏ hàng chứ không làm lỗi."""
import json

import pytest
from fastapi.testclient import TestClient

from germanqa.api.app import create_app
from germanqa.api.experiments import MATRIX_CAVEATS, list_experiments, list_matrix, summarize_file
from germanqa.api.settings import ApiSettings
from germanqa.config import RESULTS_DIR, ROOT

CELL = lambda em, f1, n=10, ci=True: {"em": em, "f1": f1, "queries": n, **({"em_ci": [em - 0.05, em + 0.05], "f1_ci": [f1 - 0.05, f1 + 0.05]} if ci else {})}
RET = lambda h5: {"hit@1": h5 - 0.2, "hit@5": h5, "hit@20": min(1.0, h5 + 0.1), "mrr@10": h5 - 0.1, "queries": 10}


def put(d, name, obj):
    (d / name).write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")


def hybrid_file(tag, em):
    split = {"hybrid": {"first_ref": CELL(em, 0.5), "all_refs": CELL(em, 0.5)}, "retrieval_hybrid": RET(0.9),
             "B0_bm25": {"first_ref": CELL(0.2, 0.35)}, "retrieval_B0_bm25": RET(0.65),
             "B1_dense384": {"first_ref": CELL(0.3, 0.45)}, "retrieval_B1_dense384": RET(0.8)}
    return {"experiment_id": f"eval_dev_hybrid_{tag}_598ddf6f", "config": {"split": "dev", "retriever": "hybrid"},
            "metrics": {"dev": split, "dev_unseen": split, "timing": {"total_time": {"p50_ms": 90.0}}}}


@pytest.fixture
def env(tmp_path):
    res, base = tmp_path / "results", tmp_path / "baselines"
    res.mkdir(), base.mkdir()
    put(res, "eval_dev_hybrid_20261107T135423Z_598ddf6f.json", hybrid_file("20261107T135423Z", 0.33))
    put(res, "eval_dev_hybrid_20261107T131435Z_598ddf6f.json", hybrid_file("20261107T131435Z", 0.25))      # lần chạy thử cũ, cùng cấu hình
    put(res, "14f_H1_dev_unseen.json", {"H1": {"first_ref": CELL(0.2051, 0.4417, 234)}, "H0": {"first_ref": CELL(0.2094, 0.4327, 234)}})
    man = lambda code, ref, **kw: {"code": code, "config_id": f"{code}_abcd1234", "frozen_on": "2026-11-08", "status": "ĐÃ ĐÓNG BĂNG", "reference_dev_run": {"file": f"results/{ref}.json", **kw}}
    put(base, "H0.json", man("H0", "eval_dev_hybrid_20261107T135423Z_598ddf6f"))
    put(base, "B0.json", man("B0", "khong_con_file"))                                                       # tệp tham chiếu đã mất
    put(base, "B1.json", {"code": "B1", "config_id": "B1_cu"})                                               # bản cũ, không có reference_dev_run
    put(base, "B1.v2.json", man("B1", "eval_dev_dense_20261107T072837Z_b97b5c1e"))             # tệp tham chiếu của B1 (không cần tồn tại: hàng B1 lấy số từ tệp H0)
    put(base, "H1.json", man("H1", "14f_H1_dev_unseen", retrieval={"hit@5": 0.9615, "hit@20": 0.9829}))
    return res, base


def test_hybrid_co_so_do_thay_vi_chi_thoi_gian(env):
    res, _ = env
    it = summarize_file(res / "eval_dev_hybrid_20261107T135423Z_598ddf6f.json")
    assert set(it["metrics"]) == {"dev", "dev_unseen"}
    m = it["metrics"]["dev_unseen"]
    assert m["em"] == 0.33 and m["hit@5"] == 0.9 and m["bm25_em"] == 0.2 and m["dense_em"] == 0.3 and "bm25_hit@5" in m and "dense_hit@5" in m
    assert it["metrics"]["dev"]["total_p50_ms"] == 90.0
    assert it["intervals"]["dev"]["em"] == [pytest.approx(0.28), pytest.approx(0.38)]
    assert it["caveats"] and "dev_unseen" in it["caveats"][0]


def test_all_refs_chi_hien_khi_khac_first_ref():
    same = {"end_to_end": {"first_ref": CELL(0.3, 0.4), "all_refs": CELL(0.3, 0.4)}}
    diff = {"end_to_end": {"first_ref": CELL(0.22, 0.39), "all_refs": CELL(0.33, 0.45)}}
    from germanqa.api.experiments import _metrics
    assert "em_all_refs" not in _metrics("eval_dev", {"split": "dev"}, same)["dev"]
    t = _metrics("eval_test", {"split": "test"}, diff)["test"]
    assert t["em"] == 0.22 and t["em_all_refs"] == 0.33 and t["f1_all_refs"] == 0.45


def test_ma_ma_tran_va_gop_lan_chay_trung(env):
    res, base = env
    items, _ = list_experiments(res, None, base)
    by = {i["experiment_id"]: i for i in items}
    ref, smoke = by["eval_dev_hybrid_20261107T135423Z_598ddf6f"], by["eval_dev_hybrid_20261107T131435Z_598ddf6f"]
    assert ref["code"] == "H0"
    assert ref["duplicate_of"] is None and smoke["duplicate_of"] == ref["experiment_id"] and smoke["code"] is None


def test_khong_co_thu_muc_baselines_thi_van_liet_ke_binh_thuong(env):
    res, _ = env
    items, skipped = list_experiments(res, None, None)
    assert all(i["code"] is None for i in items)
    assert [i["experiment_id"] for i in items if i["duplicate_of"]]          # vẫn gộp được lần trùng khi không có mã
    assert list_experiments(res)[0]


def test_ma_tran_hang_dung_nguon_va_bo_hang_thieu_du_lieu(env):
    res, base = env
    rows = {r["code"]: r for r in list_matrix(base, res)}
    assert set(rows) == {"B0", "B1", "H0", "H1"}                            # B0/B1/H0 lấy số từ tệp so sánh của H0, nên B0 không cần tệp riêng
    assert rows["B0"]["splits"]["dev_unseen"]["em"] == 0.2 and rows["B0"]["splits"]["dev_unseen"]["hit@5"] == 0.65
    assert rows["B1"]["config_id"] == "B1_abcd1234"                         # ưu tiên B1.v2 hơn B1 cũ
    assert set(rows["H0"]["splits"]) == {"dev", "dev_unseen"} and rows["H0"]["splits"]["dev_unseen"]["hit@5"] == 0.9
    assert rows["H1"]["splits"] == {"dev_unseen": {"em": 0.2051, "f1": 0.4417, "hit@5": 0.9615, "hit@20": 0.9829}}
    assert rows["H1"]["queries"] == {"dev_unseen": 234} and "dev" not in rows["H1"]["splits"]
    assert rows["H1"]["reference_file"] == "14f_H1_dev_unseen.json"


def test_ma_tran_khong_doc_tep_ngoai_thu_muc_ket_qua(env, tmp_path):
    res, base = env
    put(tmp_path, "secret.json", {"H1": {"first_ref": CELL(0.99, 0.99, 234)}})             # tệp nằm NGOÀI thư mục kết quả
    put(base, "H1.json", {"code": "H1", "reference_dev_run": {"file": "../secret.json", "retrieval": {"hit@5": 0.9}}})
    rows = {r["code"]: r for r in list_matrix(base, res)}
    # chỉ lấy TÊN tệp rồi tìm trong thư mục kết quả -> không thấy tệp ngoài -> chỉ còn số nằm sẵn trong manifest
    assert rows["H1"]["splits"] == {"dev": {"hit@5": 0.9}}
    assert "0.99" not in json.dumps(list(rows.values()))


def test_ma_tran_khong_lo_duong_dan_may(env):
    res, base = env
    blob = json.dumps(list_matrix(base, res))
    assert "D:" not in blob and "baselines" not in blob and "/" not in blob.replace("→", "")
    assert all("/" not in (r["reference_file"] or "") and "\\" not in (r["reference_file"] or "") for r in list_matrix(base, res))


def test_endpoint_matrix_chi_doc_va_co_canh_bao(env):
    res, base = env
    before = sorted(p.name for p in res.iterdir()) + sorted(p.name for p in base.iterdir())
    s = ApiSettings(results_dir=str(res), baselines_dir=str(base))
    with TestClient(create_app(s, lambda _: ({}, {}))) as c:
        r = c.get("/api/v1/matrix")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 4 and body["caveats"] == MATRIX_CAVEATS and {x["code"] for x in body["rows"]} == {"B0", "B1", "H0", "H1"}
        e = c.get("/api/v1/experiments").json()
        assert {i["code"] for i in e["items"] if i["code"]} and any(i["duplicate_of"] for i in e["items"])
    assert sorted(p.name for p in res.iterdir()) + sorted(p.name for p in base.iterdir()) == before


def test_du_lieu_that_cua_du_an():
    """Hồi quy cho lỗi đã thấy: tệp eval_dev_hybrid thật từng chỉ hiện thời gian p50, không EM/F1/Hit@k."""
    rows = {r["code"]: r for r in list_matrix(ROOT / "baselines", RESULTS_DIR)}
    assert {"B0", "B1", "H0", "H1"} <= set(rows)
    for code in ("B0", "B1", "H0"):
        assert set(rows[code]["splits"]) == {"dev", "dev_unseen"}, code
        assert {"em", "f1", "hit@5", "hit@20"} <= set(rows[code]["splits"]["dev_unseen"])
    assert set(rows["H1"]["splits"]) == {"dev_unseen"} and rows["H1"]["queries"]["dev_unseen"] == 234
    h0 = rows["H0"]["splits"]["dev_unseen"]
    assert h0["hit@5"] == pytest.approx(0.8889, abs=1e-4) and rows["H1"]["splits"]["dev_unseen"]["hit@5"] == pytest.approx(0.9615, abs=1e-4)
    items, _ = list_experiments(RESULTS_DIR, "eval_dev_hybrid", ROOT / "baselines")
    ref = [i for i in items if i["code"] == "H0"]
    assert len(ref) == 1 and {"em", "f1", "hit@5", "hit@20"} <= set(ref[0]["metrics"]["dev_unseen"])
    assert len({"eval_dev_hybrid_20261107T135423Z_598ddf6f", ref[0]["experiment_id"]}) == 1


def test_du_phong_trong_manifest_chuan_hoa_ten_tap_dev_gach_noi(tmp_path):
    """Manifest H1 thật ghi split là 'dev-unseen' (gạch nối). Không chuẩn hóa thì số của H1 bị dán nhãn nhầm là 'dev'."""
    res, base = tmp_path / "r", tmp_path / "b"
    res.mkdir(), base.mkdir()
    put(base, "H1.json", {"code": "H1", "reference_dev_run": {"file": "results/mat.json", "split": "dev-unseen",
                                                               "end_to_end_first_ref": {"em": 0.2, "f1": 0.44, "queries": 234}, "retrieval": {"hit@5": 0.96}}})
    row = list_matrix(base, res)[0]
    assert set(row["splits"]) == {"dev_unseen"} and row["queries"] == {"dev_unseen": 234} and row["splits"]["dev_unseen"]["hit@5"] == 0.96


def test_khong_co_manifest_nao_thi_ma_tran_rong_khong_loi(tmp_path):
    base = tmp_path / "b"
    base.mkdir()
    put(base, "B1.json", {"code": "B1", "config_id": "B1_cu"})              # bản cũ không có reference_dev_run
    put(base, "hong.json", "{ khong phai json")
    assert list_matrix(base, tmp_path) == [] and list_matrix(tmp_path / "khong_co", tmp_path) == [] and list_matrix(None, tmp_path) == []



def test_khac_cau_hinh_thi_khong_bi_gop_nham(tmp_path):
    """Cùng loại nhưng khác đuôi băm cấu hình = hai thí nghiệm khác nhau: không được đánh dấu trùng, nếu không sẽ giấu mất một thí nghiệm thật."""
    res = tmp_path / "r"
    res.mkdir()
    for stamp, digest in (("20261107T135423Z", "aaaaaaaa"), ("20261107T131435Z", "bbbbbbbb")):
        f = hybrid_file(stamp, 0.33)
        f["experiment_id"] = f"eval_dev_hybrid_{stamp}_{digest}"
        put(res, f"eval_dev_hybrid_{stamp}_{digest}.json", f)
    items, _ = list_experiments(res)
    assert len(items) == 2 and all(i["duplicate_of"] is None for i in items)
