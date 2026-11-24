import numpy as np
import pytest

from germanqa.retrieval.dense import DenseRetriever, check_manifest_compat, save_artifacts

CFG = {"ctx_encoder": "r/ctx", "question_encoder": "r/q", "passage_field": "original_text",
       "passage_input": "text", "similarity": "dot"}
MAN = {"ctx_encoder": {"repo": "r/ctx", "revision": "aaa"}, "question_encoder": {"repo": "r/q", "revision": "bbb"},
       "passage_field": "original_text", "passage_input": "text", "similarity": "dot"}
REV = {"ctx_encoder": {"repo": "r/ctx", "revision": "aaa"}, "question_encoder": {"repo": "r/q", "revision": "bbb"}}


class FakeEncoder:
    def __init__(self, revisions):
        self._rev = revisions

    def revisions(self):
        return self._rev


def test_tuong_thich_thi_khong_loi():
    check_manifest_compat(MAN, CFG, REV)
    check_manifest_compat(MAN, CFG)            # không có revisions: chỉ so cấu hình


@pytest.mark.parametrize("key,val", [("passage_field", "retrieval_text"), ("passage_input", "title_text"), ("similarity", "cosine")])
def test_doi_dinh_dang_thi_bi_tu_choi(key, val):
    with pytest.raises(ValueError, match=key):
        check_manifest_compat(MAN, {**CFG, key: val}, REV)


def test_doi_encoder_hoac_revision_thi_bi_tu_choi():
    with pytest.raises(ValueError, match="ctx_encoder"):
        check_manifest_compat(MAN, {**CFG, "ctx_encoder": "r/other"}, REV)
    bad = {**REV, "question_encoder": {"repo": "r/q", "revision": "zzz"}}
    with pytest.raises(ValueError, match="question_encoder revision"):
        check_manifest_compat(MAN, CFG, bad)


def test_from_dir_chi_so_khi_duoc_yeu_cau(tmp_path):
    vecs = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    save_artifacts(tmp_path, vecs, ["a", "b", "c", "d"], MAN)
    DenseRetriever.from_dir(tmp_path, FakeEncoder(REV))                                   # hành vi cũ: không kiểm
    DenseRetriever.from_dir(tmp_path, FakeEncoder(REV), expected_cfg=CFG)                 # có kiểm: khớp thì qua
    with pytest.raises(ValueError):
        DenseRetriever.from_dir(tmp_path, FakeEncoder(REV), expected_cfg={**CFG, "passage_input": "title_text"})
