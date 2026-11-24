"""Nghiệm thu GĐ2 trên dữ liệu thật. Bỏ qua nếu chưa chạy scripts/02_preprocess.py."""
import pytest

from germanqa.config import PROCESSED_DIR
from germanqa.data.io_utils import read_json, read_jsonl

pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "corpus.jsonl").exists(),
                                reason="chưa có data/processed")


@pytest.fixture(scope="module")
def data():
    d = PROCESSED_DIR
    return {
        "corpus": {r["passage_id"]: r for r in read_jsonl(d / "corpus.jsonl")},
        "qa": {s: read_jsonl(d / f"qa_{s}.jsonl") for s in ("train", "dev", "test")},
        "qrels": {s: read_json(d / f"qrels_{s}.json") for s in ("train", "dev", "test")},
    }


def test_passage_id_khong_trung(data):
    ids = list(data["corpus"])
    assert len(ids) == len(set(ids))


def test_qrels_va_qa_tro_dung_vao_corpus(data):
    for s in ("train", "dev", "test"):
        for r in data["qa"][s]:
            assert r["passage_id"] in data["corpus"]
        for pids in data["qrels"][s].values():
            assert all(p in data["corpus"] for p in pids)


def test_offset_dap_an_khop_original_text(data):
    for s in ("train", "dev", "test"):
        for r in data["qa"][s]:
            text = data["corpus"][r["passage_id"]]["original_text"]
            for a in r["answers"]:
                if a["start"] is not None:
                    assert text[a["start"]:a["start"] + len(a["text"])] == a["text"]


def test_khong_overlap_cau_hoi_giua_cac_tap(data):
    ids = {s: {r["question_id"] for r in data["qa"][s]} for s in data["qa"]}
    assert ids["train"].isdisjoint(ids["dev"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["dev"].isdisjoint(ids["test"])


def test_khong_ro_ri_doan_van(data):
    pids = {s: {r["passage_id"] for r in data["qa"][s]} for s in data["qa"]}
    assert pids["train"].isdisjoint(pids["dev"])      # nhờ chia theo nhóm
    assert pids["train"].isdisjoint(pids["test"])     # nhờ luật R3


def test_original_text_khong_bi_sua(data):
    for r in list(data["corpus"].values())[:200]:
        assert r["retrieval_text"] == r["retrieval_text"].lower()
        assert "\u200b" not in r["retrieval_text"]
