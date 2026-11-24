import copy
import json

import pytest

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, read_jsonl, write_jsonl
from germanqa.evaluation.manifest import load_manifest, verify_manifest, verify_manifest_detail
from germanqa.experiment import load_config
from germanqa.retrieval import bm25_store


# ---------------- D5: max_answer_tokens ----------------
def test_pipeline_yaml_khop_reader_gold_context_chosen():
    chosen = read_json(REPORT_DIR / "reader_gold_context.json")["chosen"]
    assert load_config("pipeline")["reader"]["max_answer_tokens"] == chosen["max_answer_tokens"] == 80
    assert load_config("pipeline")["reader"]["score_mode"] == chosen["score_mode"]


def test_manifest_ghi_dung_gia_tri_hieu_luc_va_lech_30_80():
    for code in ("B0", "H0"):
        d5 = load_manifest(code)["reader"]["d5_max_answer_tokens"]
        assert d5["effective_max_answer_tokens"] == load_config("pipeline")["reader"]["max_answer_tokens"] == 80
        assert d5["reader_yaml_max_answer_tokens"] == load_config("reader")["reader"]["max_answer_tokens"] == 30
        assert d5["reader_gold_context_chosen"] == 80


def test_pipeline_truyen_cho_reader_dung_gia_tri_trong_manifest(tmp_path, monkeypatch):
    """Dựng pipeline thật với Reader giả để bắt tham số: giá trị reader nhận phải bằng giá trị trong manifest B0."""
    corpus = [{"passage_id": f"p{i}", "retrieval_text": t, "original_text": t}
              for i, t in enumerate(["Bonn war Hauptstadt", "Köln hat einen Dom"])]
    write_jsonl(tmp_path / "corpus.jsonl", corpus)
    seen = {}

    class FakeReader:
        def __init__(self, checkpoint, max_length, stride, n_best, max_answer_tokens, score_mode, batch_size, **kw):
            seen.update(max_answer_tokens=max_answer_tokens, score_mode=score_mode)

    monkeypatch.setattr("germanqa.reader.inference.Reader", FakeReader)
    from germanqa.pipeline import build_pipeline
    build_pipeline(None, tmp_path, "bm25", bm25_store_dir=tmp_path / "store")
    assert seen["max_answer_tokens"] == load_manifest("B0")["reader"]["d5_max_answer_tokens"]["effective_max_answer_tokens"]


def test_pipeline_dung_chi_muc_luu_lan_hai(tmp_path, monkeypatch):
    corpus = [{"passage_id": "p0", "retrieval_text": "Bonn Hauptstadt", "original_text": "x"}]
    write_jsonl(tmp_path / "corpus.jsonl", corpus)
    from germanqa.pipeline import build_retriever
    srcs = []
    real = bm25_store.load_or_build

    def spy(*a, **k):
        r, info = real(*a, **k)
        srcs.append(info["source"])
        return r, info
    monkeypatch.setattr(bm25_store, "load_or_build", spy)
    build_retriever("bm25", corpus, tmp_path, bm25_store_dir=tmp_path / "s")
    build_retriever("bm25", corpus, tmp_path, bm25_store_dir=tmp_path / "s")
    assert srcs == ["built", "loaded"]


# ---------------- manifest ----------------
@pytest.mark.parametrize("code", ["B0", "B1", "H0"])
def test_manifest_con_khop_tep_that(code):
    assert verify_manifest(load_manifest(code)) == []


@pytest.mark.parametrize("code,min_checks", [("B0", 14), ("B1", 13), ("H0", 22)])
def test_manifest_that_su_kiem_duoc_nhieu_sha1(code, min_checks):
    """Chặn lỗi 'rỗng = khớp' vì không kiểm gì: mỗi manifest phải kiểm ít nhất chừng này SHA-1."""
    bad, n = verify_manifest_detail(load_manifest(code))
    assert bad == [] and n >= min_checks, (code, n)


def test_manifest_B1_cu_phat_hien_artifact_bi_doi():
    man = copy.deepcopy(load_manifest("B1"))
    man["artifacts"]["text384"]["sha1"]["index.faiss"] = "2" * 40
    assert any("index.faiss" in b for b in verify_manifest(man))


def test_manifest_phat_hien_tep_bi_doi():
    man = copy.deepcopy(load_manifest("B0"))
    man["data"]["corpus.jsonl"] = "0" * 40
    man["reader"]["sha1"]["model.safetensors"] = "1" * 40
    bad = verify_manifest(man)
    assert any("corpus" in b for b in bad) and any("model.safetensors" in b for b in bad)


def test_manifest_co_thiet_bi_va_precision():
    for code in ("B0", "H0"):
        env = load_manifest(code)["environment"]
        assert env["gpu"] and env["torch"] and env["cuda_runtime"] and "reader_fp16" in env["precision"]


def test_bm25_trong_manifest_trung_dau_van_tay_chi_muc_dang_luu():
    man = load_manifest("B0")["bm25"]
    meta = json.loads((bm25_store.DEFAULT_DIR / "meta.json").read_text(encoding="utf-8"))
    assert man["fingerprint_sha1"] == meta["fingerprint"]["sha1"] and man["index_npz_sha1"] == meta["index_npz_sha1"]


# ---------------- tập chọn (11.6) ----------------
def test_tap_chon_dung_so_va_khong_chua_test():
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    tr, dv = set(sel["train"]), set(sel["dev"])
    assert (len(tr), len(dv), len(tr) + len(dv)) == (1989, 234, 2223) and not (tr & dv)
    splits = {s: {r["question_id"] for r in read_jsonl(PROCESSED_DIR / f"qa_{s}.jsonl")} for s in ("train", "dev")}
    assert tr <= splits["train"] and dv <= splits["dev"]
    assert dv == set(read_json(RESULTS_DIR / "dev_unseen_question_ids.json"))
    assert sel["counts"]["total"] == 2223


def test_so_phoi_bay_chi_co_toi_da_mot_lan_mo_test_cua_phan_b():
    """Phần B chỉ được mở test MỘT lần (GĐ15.6): hoặc chưa có lần nào, hoặc đúng một lần là 15f_run_test.py cho B1, H0, H1.
    Bất kỳ lần tính chỉ số/phân tích lỗi test nào khác phát sinh từ lúc có sổ đều làm test này đỏ."""
    from germanqa.experiment_rules import read_ledger
    rows = read_ledger()
    live = [r for r in rows if r["source"] == "live" and r["kind"] in ("metric_run", "error_analysis")]
    allowed = [("15f_run_test.py", "metric_run", ("B1", "H0", "H1"))]
    got = [(r["script"], r["kind"], tuple(sorted(r["codes"]))) for r in live]
    assert got == [] or got == allowed, got
