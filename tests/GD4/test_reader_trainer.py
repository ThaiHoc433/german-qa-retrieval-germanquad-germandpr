
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transformers import BertConfig, BertForQuestionAnswering          # noqa: E402

from germanqa.reader.features import prepare_eval_features, prepare_train_features   # noqa: E402
from germanqa.reader.trainer import collate, evaluate, predict_logits, train          # noqa: E402
from germanqa.tokenization import load_bert_tokenizer                                 # noqa: E402

WORDS = ["Wer", "bellt", "Was", "schläft", "frisst", "Der", "Hund", "laut", "Die", "Katze", "leise", "Maus", ".", "?"]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    path = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS), encoding="utf-8")
    return load_bert_tokenizer(path)


def examples():
    """Ba ví dụ có context khác nhau; đáp án là từ nằm ở chỗ khác nhau, mô hình nhỏ học thuộc được."""
    c1, c2, c3 = "Der Hund bellt laut .", "Die Katze schläft leise .", "Die Maus frisst ."
    return [
        {"id": "q1", "question": "Wer bellt ?", "context": c1, "answers": [{"text": "Hund", "start": c1.index("Hund")}]},
        {"id": "q2", "question": "Wer bellt ?", "context": c2, "answers": [{"text": "leise", "start": c2.index("leise")}]},
        {"id": "q3", "question": "Wer bellt ?", "context": c3, "answers": [{"text": "Maus", "start": c3.index("Maus")}]},
    ]


def tiny_model(tok):
    cfg = BertConfig(vocab_size=len(tok.get_vocab()), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    torch.manual_seed(0)
    return BertForQuestionAnswering(cfg)


def test_collate_dem_den_do_dai_lon_nhat_trong_lo():
    batch = [{"input_ids": [5, 6, 7], "token_type_ids": [0, 0, 1], "start_positions": 1, "end_positions": 2},
             {"input_ids": [5, 6], "token_type_ids": [0, 1], "start_positions": 0, "end_positions": 0}]
    out = collate(batch, pad_id=0)
    assert out["input_ids"].tolist() == [[5, 6, 7], [5, 6, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["start_positions"].tolist() == [1, 0]


def test_predict_logits_tra_dung_do_dai_tung_cua_so(tok):
    feats = prepare_eval_features(examples(), tok, max_length=16, stride=4)
    s, e = predict_logits(tiny_model(tok), feats, batch_size=2, device="cpu", fp16=False)
    assert len(s) == len(e) == len(feats)
    for f, a, b in zip(feats, s, e):
        assert len(a) == len(b) == len(f["input_ids"])
        assert np.isfinite(a).all()


def test_ket_qua_khong_phu_thuoc_kich_thuoc_lo(tok):
    feats = prepare_eval_features(examples(), tok, max_length=16, stride=4)
    model = tiny_model(tok)
    s1, _ = predict_logits(model, feats, batch_size=1, device="cpu", fp16=False)
    s3, _ = predict_logits(model, feats, batch_size=3, device="cpu", fp16=False)
    for a, b in zip(s1, s3):
        assert np.allclose(a, b, atol=1e-4)               # đệm không làm đổi logit của token thật


def test_huan_luyen_mo_hinh_nho_hoc_duoc_va_giu_checkpoint_tot_nhat(tok):
    exs = examples()
    feats = prepare_train_features(exs, tok, max_length=32, stride=8)
    dev_feats = prepare_eval_features(exs, tok, max_length=32, stride=8)
    model = tiny_model(tok)
    cfg = {"n_best": 10, "max_answer_tokens": 5, "eval_batch_size": 4,
           "train": {"epochs": 60, "lr": 3e-3, "batch_size": 1, "grad_accum": 1, "warmup_ratio": 0.1,
                     "weight_decay": 0.0, "max_grad_norm": 1.0, "fp16": False, "log_every": 1000}}
    before, _ = evaluate(model, exs, dev_feats, 10, 5, 4, "cpu", fp16=False)
    state, history = train(model, feats, exs, dev_feats, cfg, seed=1, device="cpu", log=lambda m: None)
    assert len(history) == 60 and history[0]["epoch"] == 1
    model.load_state_dict(state)
    after, _ = evaluate(model, exs, dev_feats, 10, 5, 4, "cpu", fp16=False)
    assert after["em"] == 1.0 and after["f1"] == 1.0         # học thuộc 3 ví dụ
    assert after["em"] > before["em"]
