"""Reader suy luận (GĐ5.1) với mô hình BERT nhỏ trên CPU."""
import pytest

torch = pytest.importorskip("torch")

from transformers import BertConfig, BertForQuestionAnswering          # noqa: E402

from germanqa.reader.inference import Reader                            # noqa: E402

WORDS = ["Wer", "bellt", "Was", "schläft", "frisst", "Der", "Hund", "laut", "Die", "Katze", "leise", "Maus", ".", "?"]
PASSAGES = [{"passage_id": "a", "text": "Der Hund bellt laut ."}, {"passage_id": "b", "text": "Die Katze schläft leise ."},
            {"passage_id": "c", "text": "Die Maus frisst ."}]


@pytest.fixture(scope="module")
def reader(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ckpt")
    (folder / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS), encoding="utf-8")
    cfg = BertConfig(vocab_size=5 + len(WORDS), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    torch.manual_seed(0)
    BertForQuestionAnswering(cfg).save_pretrained(folder)
    return Reader(folder, max_length=16, stride=4, n_best=10, max_answer_tokens=5, fp16=False, device="cpu")


def test_moi_doan_mot_ket_qua_dung_thu_tu(reader):
    out = reader.read("Wer bellt ?", PASSAGES)
    assert [o["passage_id"] for o in out] == ["a", "b", "c"]


def test_span_cat_tu_van_ban_goc(reader):
    for o, p in zip(reader.read("Wer bellt ?", PASSAGES), PASSAGES):
        assert o["start"] >= 0 and p["text"][o["start"]:o["end"]] == o["text"]


def test_diem_cua_mot_doan_khong_phu_thuoc_cac_doan_di_kem(reader):
    """Nền tảng để suy ra kết quả top-3, top-5 từ một lần chạy top-10: điểm từng đoạn độc lập với đoạn khác."""
    alone = {p["passage_id"]: reader.read("Wer bellt ?", [p])[0] for p in PASSAGES}
    together = {o["passage_id"]: o for o in reader.read("Wer bellt ?", PASSAGES)}
    for pid in alone:
        assert alone[pid]["text"] == together[pid]["text"]
        assert alone[pid]["score"] == pytest.approx(together[pid]["score"], abs=1e-4)


def test_khong_co_doan_nao_thi_tra_rong(reader):
    assert reader.read("Wer bellt ?", []) == []