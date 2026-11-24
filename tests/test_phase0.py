"""GĐ0: kiểm tra hợp đồng dữ liệu (schema, offset UTF-16) và công cụ tái lập (mã thí nghiệm, cấu hình)."""
from germanqa.config import DATASETS, DEV_RATIO, PROCESSED_DIR, RAW_DIR, ROOT, SEED
from germanqa.experiment import load_config, new_experiment_id, save_run
from germanqa.schemas import Answer, QAExample, QAResponse, to_utf16_offset


def test_duong_dan_tinh_tu_goc_du_an():
    assert RAW_DIR == ROOT / "data" / "raw"
    assert PROCESSED_DIR == ROOT / "data" / "processed"
    assert (ROOT / "src" / "germanqa" / "config.py").exists()


def test_hang_so_thi_nghiem():
    assert SEED == 42
    assert 0 < DEV_RATIO < 0.5
    assert DATASETS["germanquad"].expected == {"train": 11518, "test": 2204}


def test_offset_utf16_chu_thuong_khong_doi():
    assert to_utf16_offset("Häuser über", 7) == 7


def test_offset_utf16_ky_tu_ngoai_bmp():
    text = "ab\U0001F600cd"            # emoji: Python đếm 1 ký tự, UTF-16 đếm 2
    assert len(text) == 5
    assert to_utf16_offset(text, 3) == 4    # sau emoji: lệch thêm 1
    assert to_utf16_offset(text, 2) == 2    # trước emoji: chưa lệch


def test_response_ra_dict_du_truong():
    r = QAResponse("Wer?", "Anna", "p_1", "Anna kam.", 0, 4, 1.0, 2.0, 0.001, 0.2, 0.3,
                   "reader-v0", "corpus-v0", "exp_1")
    d = r.to_dict()
    assert d["answer"] == "Anna" and d["answer_end"] - d["answer_start"] == 4
    assert "reader_score" in d and "total_time" in d


def test_qa_example_giu_answer_start_none():
    ex = QAExample("q_1", "Wer?", "p_1", [Answer("Anna", None)], "germandpr")
    assert ex.answers[0].start is None


def test_ma_thi_nghiem_cung_cau_hinh_cung_duoi_bam():
    cfg = {"bm25": {"k1": 1.5, "b": 0.75}}
    a, b = new_experiment_id("bm25", cfg), new_experiment_id("bm25", cfg)
    assert a.split("_")[-1] == b.split("_")[-1]
    other = new_experiment_id("bm25", {"bm25": {"k1": 0.7, "b": 0.2}})
    assert other.split("_")[-1] != a.split("_")[-1]


def test_doc_cau_hinh_yaml():
    cfg = load_config("baseline")
    assert cfg["bm25"]["k1"] > 0 and cfg["evaluation"]["ks"] == [1, 5, 10, 20]


def test_save_run_ghi_cau_hinh_va_moi_truong(tmp_path, monkeypatch):
    import germanqa.experiment as ex
    monkeypatch.setattr(ex, "RESULTS_DIR", tmp_path)
    path = save_run("exp_x", {"a": 1}, {"mrr": 0.5})
    import json
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["config"] == {"a": 1} and rec["metrics"] == {"mrr": 0.5}
    assert "python" in rec["environment"]
