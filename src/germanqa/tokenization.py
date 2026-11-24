"""Nạp tokenizer WordPiece của BERT từ vocab.txt, có kiểm tra từ vựng.

Lý do: ở transformers 5.17-5.22, BertTokenizerFast(vocab_file=...) bỏ qua từ vựng
mà không báo lỗi (mọi từ thành [UNK]). Nên dựng trực tiếp từ `tokenizers` và
kiểm tra kích thước từ vựng.
"""
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

SPECIALS = dict(unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
                cls_token="[CLS]", mask_token="[MASK]")


def load_bert_tokenizer(vocab_path, do_lower_case=False):
    """Trả về tokenizer nhanh có offset_mapping. gbert-base là cased: do_lower_case=False."""
    vocab_path = Path(vocab_path)
    lines = vocab_path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    vocab = {tok: i for i, tok in enumerate(lines)}
    for name in SPECIALS.values():
        if name not in vocab:
            raise ValueError(f"vocab.txt thiếu token đặc biệt {name}")

    tk = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    tk.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True,
        strip_accents=False, lowercase=do_lower_case)   # giữ ä ö ü ß
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    tk.decoder = decoders.WordPiece(prefix="##")

    tok = PreTrainedTokenizerFast(tokenizer_object=tk, model_max_length=512,
                                  padding_side="right",
                                  model_input_names=["input_ids", "token_type_ids", "attention_mask"],
                                  **SPECIALS)


    if len(tok) != len(vocab):
        raise RuntimeError(f"Từ vựng nạp thiếu: tokenizer {len(tok)} != vocab.txt {len(vocab)}")
    return tok