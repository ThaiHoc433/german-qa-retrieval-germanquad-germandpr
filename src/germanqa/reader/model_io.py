### File này làm ba việc: tìm hoặc tải mô hình, nạp tokenizer qua load_bert_tokenizer, nạp mô hình. Chép đúng nội dung này: ###

"""Nạp tokenizer và mô hình reader (GĐ4)."""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoModelForQuestionAnswering, BertForQuestionAnswering

from germanqa.tokenization import load_bert_tokenizer


def _download(repo_id, filenames):
    """Tải từng file riêng lẻ (không gọi API liệt kê repo, vốn hay bị ngắt kết nối)."""
    last = None
    for name in filenames:
        try:
            return Path(hf_hub_download(repo_id, name))
        except Exception as err:          # thử tên file kế tiếp
            last = err
    raise last


def resolve_dir(model_name):
    """Trả về thư mục chứa mô hình: đường dẫn cục bộ nếu có, nếu không thì tải từ Hub."""
    local = Path(model_name)
    if local.is_dir():
        return local
    config = _download(model_name, ["config.json"])
    _download(model_name, ["vocab.txt"])
    _download(model_name, ["model.safetensors", "pytorch_model.bin"])
    return config.parent


def get_tokenizer(model_name):
    """Tokenizer WordPiece phân biệt chữ hoa/thường, đúng với họ GBERT/GELECTRA."""
    folder = resolve_dir(model_name)
    return load_bert_tokenizer(folder / "vocab.txt")


def get_model(model_name):
    """Backbone tiền huấn luyện cộng đầu start/end (khởi tạo ngẫu nhiên nếu chưa học QA).

    config.json của deepset/gbert-base không có khóa "model_type"; transformers 5.x không còn đoán
    được từ tên thư mục như bản cũ, nên trường hợp đó chỉ định thẳng lớp BERT."""
    folder = resolve_dir(model_name)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if "model_type" not in config:
        return BertForQuestionAnswering.from_pretrained(str(folder))
    return AutoModelForQuestionAnswering.from_pretrained(str(folder))
