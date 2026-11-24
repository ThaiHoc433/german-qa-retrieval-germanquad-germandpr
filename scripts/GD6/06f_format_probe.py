"""GĐ6.1: encoder DPR nhận đầu vào định dạng nào? Thử vài cách trên 300 cặp (câu hỏi, đoạn đúng) của GermanDPR train.
Mỗi câu hỏi được xếp hạng với đúng 300 đoạn đó; định dạng đúng với lúc huấn luyện cho top-1 cao.
Đây là phép thử ĐỊNH DẠNG, không phải phép đo chất lượng: encoder đã học trên chính các cặp này."""
import numpy as np
import torch
from datasets import load_from_disk

from germanqa.config import RAW_DIR, REPORT_DIR
from germanqa.data.io_utils import write_json
from germanqa.experiment import load_config
from germanqa.retrieval.dense import DPREncoder

N, BATCH = 300, 25


def main():
    cfg = load_config("dense")["dense"]
    train = load_from_disk(str(RAW_DIR / "germandpr"))["train"]
    rows = [train[int(i)] for i in np.random.default_rng(0).choice(len(train), N, replace=False)]
    questions = [r["question"] for r in rows]
    titles = [r["positive_ctxs"]["title"][0] for r in rows]
    texts = [r["positive_ctxs"]["text"][0] for r in rows]
    enc = DPREncoder(cfg["ctx_encoder"], cfg["question_encoder"], fp16=cfg["fp16"])
    q = enc.encode_questions(questions)
    model, tok, _ = enc.ctx

    def run(first, second, max_length, zero_token_types):
        vecs = []
        for s in range(0, N, BATCH):
            args = (first[s:s + BATCH],) if second is None else (first[s:s + BATCH], second[s:s + BATCH])
            e = tok(*args, truncation="longest_first", max_length=max_length, padding=True, return_tensors="pt").to(enc.device)
            if zero_token_types:
                e["token_type_ids"] = torch.zeros_like(e["token_type_ids"])
            with torch.no_grad(), torch.autocast(device_type=enc.device.type, dtype=torch.float16, enabled=enc.fp16):
                vecs.append(model(**e).pooler_output.float().cpu().numpy())
        scores = q @ np.concatenate(vecs).T
        rank = (scores > scores.diagonal()[:, None]).sum(1)          # 0 = đoạn đúng đứng đầu
        return {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean())}

    lower = [t.lower() for t in texts]
    cases = {
        "cặp (title, text), token_type_ids 0/1 của tokenizer, 256": run(titles, texts, 256, False),
        "cặp (title, text), token_type_ids toàn 0, 256": run(titles, texts, 256, True),
        "chỉ text, 256": run(texts, None, 256, True),
        "chỉ text, 512": run(texts, None, 512, True),
        "chỉ text viết thường, 256": run(lower, None, 256, True),
        "chỉ title, 256": run(titles, None, 256, True),
    }
    for k, v in cases.items():
        print(f"{k:58} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}")
    write_json(REPORT_DIR / "dense_format_probe.json", {"n_pairs": N, "cases": cases})


if __name__ == "__main__":
    main()
