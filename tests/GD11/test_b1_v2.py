import json

from germanqa.config import ROOT
from germanqa.evaluation.manifest import load_manifest, verify_manifest_detail


def test_b1_v2_day_du_va_khop_tep_that():
    man = load_manifest("B1.v2")
    bad, n = verify_manifest_detail(man)
    assert bad == [] and n >= 17, n                                    # 13 của B1 cũ + 4 tệp reader
    assert set(man["reader"]["sha1"]) == {"model.safetensors", "config.json", "reader_config.json", "vocab.txt"}
    assert man["reader"]["d5_max_answer_tokens"]["effective_max_answer_tokens"] == 80
    assert man["config_id"].startswith("B1_") and man["code"] == "B1"
    assert "dense" in man and "bm25" not in man                         # B1 = dense thuần, không có chỉ mục BM25


def test_b1_v2_cung_so_do_tham_chieu_voi_b1_cu_va_b1_cu_khong_doi():
    old, new = load_manifest("B1"), load_manifest("B1.v2")
    ref = json.loads((ROOT / new["reference_dev_run"]["file"]).read_text(encoding="utf-8-sig"))["metrics"]["dev"]["dense"]["all_refs"]
    assert new["reference_dev_run"]["end_to_end_all_refs"]["em"] == ref["em"]
    assert old["end_to_end_dev"]["em"] == ref["em"] if "em" in old["end_to_end_dev"] else True
    assert "reader" not in old and "config_id" not in old               # B1.json cũ giữ nguyên dạng GĐ12.5 (mốc không bị ghi đè)
    assert old["corpus_sha1"] == new["data"]["corpus.jsonl"]
