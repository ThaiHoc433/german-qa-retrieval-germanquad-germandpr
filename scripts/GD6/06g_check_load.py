"""GĐ6.0: nạp được cả hai encoder chưa? Trọng số có thật sự được nạp (không phải khởi tạo ngẫu nhiên)?"""
import torch
import transformers

from germanqa.experiment import load_config
from germanqa.retrieval.dense import snapshot_dir
from germanqa.tokenization import load_bert_tokenizer


def main():
    cfg = load_config("dense")["dense"]
    for cls_name, repo in (("DPRContextEncoder", cfg["ctx_encoder"]), ("DPRQuestionEncoder", cfg["question_encoder"])):
        d = snapshot_dir(repo)
        model, info = getattr(transformers, cls_name).from_pretrained(d, output_loading_info=True)
        tok = load_bert_tokenizer(d / "vocab.txt")
        enc = tok("Wie viele Menschen glauben an Gott?", return_tensors="pt")
        with torch.no_grad():
            vec = model.eval()(**enc).pooler_output
        print(f"{cls_name}  revision {d.name[:10]}")
        print(f"  khóa thiếu (phải rỗng): {sorted(info['missing_keys'])}   khóa thừa: {len(info['unexpected_keys'])} (pooler của BERT, DPR không dùng)")
        print(f"  từ vựng {len(tok.get_vocab())}, vector {tuple(vec.shape)}, chuẩn {vec.norm().item():.1f}, hữu hạn {bool(torch.isfinite(vec).all())}")


if __name__ == "__main__":
    main()
