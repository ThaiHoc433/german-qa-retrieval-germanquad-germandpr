import pytest
import torch

from germanqa.rerank import cross_encoder as C

transformers = pytest.importorskip("transformers")


def make_ckpt(path, num_labels=1, seed=0):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "bonn", "köln", "dom", "haus", "wort", "rhein", "stadt", "fluss", "?"]
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizerFast(str(path / "vocab.txt"), do_lower_case=True).save_pretrained(path)
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=256, num_labels=num_labels)
    BertForSequenceClassification(cfg).save_pretrained(path)
    return path


def reranker(path, **kw):
    kw.setdefault("max_length", 32)
    kw.setdefault("fp16", False)
    return C.CrossEncoderReranker(path, device="cpu", **kw)


WORDS = ["bonn", "köln", "dom", "haus", "wort", "rhein", "stadt", "fluss"]
CANDS = [{"passage_id": f"p{i}", "text": " ".join(WORDS[i % 8:] + WORDS[:i % 8]), "rank": i + 1} for i in range(10)]


# ---------- cách tính điểm từ logit ----------
def test_combine_logits_mot_dau_ra_hai_nhan_ba_nhan():
    assert C.combine_logits(torch.tensor([[2.0], [-1.0]]), None).tolist() == [2.0, -1.0]
    two = torch.tensor([[1.0, 4.0], [3.0, 0.5]])
    assert C.combine_logits(two, 1).tolist() == [3.0, -2.5]               # logit[liên quan] - logit[không]
    assert C.combine_logits(two, 0).tolist() == [-3.0, 2.5]               # đổi nhãn liên quan thì điểm đổi dấu
    three = torch.tensor([[0.0, 1.0, 2.0]])
    want = 2.0 - torch.logsumexp(torch.tensor([0.0, 1.0]), 0)
    assert C.combine_logits(three, 2)[0].item() == pytest.approx(want.item())


def test_nhieu_nhan_ma_khong_chi_dinh_nhan_lien_quan_la_loi():
    with pytest.raises(ValueError):
        C.combine_logits(torch.zeros(1, 2), None)
    with pytest.raises(ValueError):
        C.combine_logits(torch.zeros(1, 2), 2)


# ---------- sắp xếp ----------
def test_sap_giam_dan_hoa_diem_giu_thu_tu_cu():
    rows = [{"passage_id": c} for c in "abcde"]
    out = C.stable_rank(rows, [0.5, 2.0, 0.5, 2.0, -1.0])
    assert [r["passage_id"] for r in out] == ["b", "d", "a", "c", "e"]     # b trước d, a trước c (thứ tự cũ)
    assert [r["rank"] for r in out] == [1, 2, 3, 4, 5]
    assert [r["rank_before"] for r in out] == [2, 4, 1, 3, 5]
    assert out[0]["rerank_score"] == 2.0


def test_rerank_dung_chieu_diem_va_khong_them_bot_ung_vien(tmp_path, monkeypatch):
    rr = reranker(make_ckpt(tmp_path / "m"))
    monkeypatch.setattr(rr, "score_texts", lambda q, texts: [float(i == 6) * 10 + i * 0.01 for i in range(len(texts))])
    out = rr.rerank("bonn ?", CANDS)
    assert out[0]["passage_id"] == "p6"                                     # điểm cao nhất lên đầu
    assert sorted(r["passage_id"] for r in out) == sorted(c["passage_id"] for c in CANDS)
    assert len({r["passage_id"] for r in out}) == len(CANDS)
    assert [r["passage_id"] for r in rr.rerank("bonn ?", CANDS, top_n=3)] == [r["passage_id"] for r in out[:3]]
    assert rr.rerank("bonn ?", []) == []


# ---------- mô hình thật nhỏ ----------
def test_diem_khong_phu_thuoc_kich_thuoc_batch_va_thu_tu_dau_vao(tmp_path):
    ck = make_ckpt(tmp_path / "m")
    texts = [c["text"] for c in CANDS]
    a = reranker(ck, batch_size=1).score_texts("bonn dom ?", texts)
    b = reranker(ck, batch_size=16).score_texts("bonn dom ?", texts)
    assert a == pytest.approx(b, abs=1e-5)
    perm = [3, 7, 0, 9, 1, 5, 2, 8, 4, 6]
    c = reranker(ck, batch_size=4).score_texts("bonn dom ?", [texts[i] for i in perm])
    assert [c[perm.index(i)] for i in range(10)] == pytest.approx(a, abs=1e-5)     # điểm của đoạn không phụ thuộc đoạn khác
    ra = reranker(ck, batch_size=1).rerank("bonn dom ?", CANDS)
    rb = reranker(ck, batch_size=16).rerank("bonn dom ?", CANDS)
    assert [r["passage_id"] for r in ra] == [r["passage_id"] for r in rb]


def test_doi_nhan_lien_quan_dao_nguoc_thu_hang(tmp_path):
    ck = make_ckpt(tmp_path / "m2", num_labels=2)
    with pytest.raises(ValueError):
        reranker(ck)                                                         # không chỉ định nhãn -> lỗi
    s1 = reranker(ck, relevant_label=1).score_texts("bonn ?", [c["text"] for c in CANDS])
    s0 = reranker(ck, relevant_label=0).score_texts("bonn ?", [c["text"] for c in CANDS])
    assert s0 == pytest.approx([-x for x in s1], abs=1e-5)
    o1 = [r["passage_id"] for r in reranker(ck, relevant_label=1).rerank("bonn ?", CANDS)]
    o0 = [r["passage_id"] for r in reranker(ck, relevant_label=0).rerank("bonn ?", CANDS)]
    assert o1 == o0[::-1] or len(set(s1)) < len(s1)                          # đảo đúng, trừ khi có điểm trùng


# ---------- cắt token / cửa sổ ----------
def test_cua_so_chia_doan_dai_va_lay_diem_lon_nhat(tmp_path):
    ck = make_ckpt(tmp_path / "m")
    long_text = " ".join(["haus"] * 60 + ["dom", "bonn", "rhein"])           # bằng chứng ở CUỐI đoạn
    short = reranker(ck, max_length=32, overflow="truncate")
    win = reranker(ck, max_length=32, overflow="windows", stride=8)
    st = short.score_texts("bonn ?", [long_text])
    sw = win.score_texts("bonn ?", [long_text])
    assert short.last_stats["windows"] == 1 and win.last_stats["windows"] > 1 and win.last_stats["pairs"] == 1
    assert sw[0] >= st[0] - 1e-6                                              # cửa sổ đầu chính là bản bị cắt
    enc, owner = win.encode_pairs("bonn ?", [long_text])
    assert set(owner) == {0} and all(len(x) <= 32 for x in enc["input_ids"])
    # cửa sổ cuối chứa phần đuôi bị cắt ở chế độ truncate
    tail_ids = enc["input_ids"][-1]
    assert win.tokenizer.convert_tokens_to_ids("rhein") in tail_ids
    assert win.tokenizer.convert_tokens_to_ids("rhein") not in short.encode_pairs("bonn ?", [long_text])[0]["input_ids"][0]


def test_diem_cua_so_dau_bang_diem_khi_chi_cat_cuoi(tmp_path):
    ck = make_ckpt(tmp_path / "m")
    texts = [" ".join(["haus"] * 60 + ["dom", "bonn"]), "bonn dom", " ".join(["köln"] * 90)]
    win = reranker(ck, max_length=32, overflow="windows", stride=8, batch_size=2).score_detail("bonn ?", texts)
    tr = reranker(ck, max_length=32, overflow="truncate", batch_size=3).score_detail("bonn ?", texts)
    assert win["first"] == pytest.approx(tr["score"], abs=1e-5)
    assert tr["windows"] == [1, 1, 1] and win["windows"][0] > 1 and win["windows"][1] == 1 and win["windows"][2] > 1
    assert all(m >= f - 1e-6 for m, f in zip(win["score"], win["first"]))
    assert win["score"][1] == pytest.approx(win["first"][1])                 # đoạn ngắn: một cửa sổ, hai điểm trùng


def test_cau_hoi_qua_dai_van_chay_khong_loi(tmp_path):
    rr = reranker(make_ckpt(tmp_path / "m"), max_length=128)
    q = " ".join(["köln"] * 200) + " ?"
    assert len(rr.score_texts(q, ["bonn dom", "haus"])) == 2
    assert len(rr._cut_query(q).split()) <= C.MAX_QUERY_TOKENS


def test_token_lengths_khong_cat(tmp_path):
    rr = reranker(make_ckpt(tmp_path / "m"), max_length=32)
    n = rr.token_lengths("bonn ?", [" ".join(["haus"] * 100), "dom"])
    assert n[0] > 100 and n[1] < 10


# ---------- định danh ----------
def test_dau_van_tay_doi_khi_doi_cau_hinh(tmp_path):
    ck = make_ckpt(tmp_path / "m2", num_labels=2)
    base = reranker(ck, relevant_label=1).fingerprint()["sha1"]
    assert reranker(ck, relevant_label=1).fingerprint()["sha1"] == base
    assert reranker(ck, relevant_label=0).fingerprint()["sha1"] != base
    assert reranker(ck, relevant_label=1, max_length=48).fingerprint()["sha1"] != base
    assert reranker(ck, relevant_label=1, overflow="windows").fingerprint()["sha1"] != base
    other = make_ckpt(tmp_path / "m3", num_labels=2, seed=1)
    assert reranker(other, relevant_label=1).fingerprint()["sha1"] != base      # trọng số khác


# ---------- gộp cửa sổ với model giả có điểm xác định ----------
class CountTokenModel:
    """Điểm = số lần token mục tiêu xuất hiện trong cặp: cho phép biết chính xác cửa sổ nào đạt điểm nào."""
    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, input_ids, attention_mask, token_type_ids=None):
        import types
        return types.SimpleNamespace(logits=(input_ids == self.token_id).sum(1).float().unsqueeze(1))


def stub_reranker(tmp_path, overflow):
    rr = reranker(make_ckpt(tmp_path / f"s_{overflow}"), max_length=32, overflow=overflow, stride=8)
    rr.model = CountTokenModel(rr.tokenizer.convert_tokens_to_ids("rhein"))
    return rr


def test_gop_cua_so_lay_diem_lon_nhat_khong_phai_nho_nhat_hay_cuoi(tmp_path):
    tail = " ".join(["haus"] * 60 + ["rhein"])                    # bằng chứng 'rhein' chỉ có ở CUỐI đoạn dài
    texts = [tail, "haus rhein", " ".join(["haus"] * 60)]
    win = stub_reranker(tmp_path, "windows").score_detail("bonn ?", texts)
    tr = stub_reranker(tmp_path, "truncate").score_detail("bonn ?", texts)
    assert tr["score"] == [0.0, 1.0, 0.0]                         # cắt cuối đoạn: đoạn 0 mất bằng chứng
    assert win["score"] == [1.0, 1.0, 0.0]                        # cửa sổ: thấy bằng chứng ở cuối -> lấy điểm lớn nhất
    assert win["first"] == [0.0, 1.0, 0.0]                        # điểm cửa sổ đầu == điểm khi chỉ cắt
    assert win["windows"][0] > 1 and win["windows"][1] == 1
    order_win = [r["passage_id"] for r in stub_reranker(tmp_path, "windows").rerank(
        "bonn ?", [{"passage_id": f"p{i}", "text": t} for i, t in enumerate(texts)])]
    order_tr = [r["passage_id"] for r in stub_reranker(tmp_path, "truncate").rerank(
        "bonn ?", [{"passage_id": f"p{i}", "text": t} for i, t in enumerate(texts)])]
    assert order_win == ["p0", "p1", "p2"] and order_tr == ["p1", "p0", "p2"]       # p0 chỉ lên đầu nhờ cửa sổ
