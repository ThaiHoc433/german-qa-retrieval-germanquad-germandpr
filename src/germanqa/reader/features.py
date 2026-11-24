
"""Biến (câu hỏi, đoạn văn, đáp án) thành các cửa sổ token có nhãn (GĐ4.2)."""
from germanqa.data.io_utils import read_json, read_jsonl


def load_examples(data_dir, split):
    """Đọc qa_<split>.jsonl và gắn nguyên văn đoạn đúng (original_text) làm context."""
    corpus = {r["passage_id"]: r["original_text"] for r in read_jsonl(data_dir / "corpus.jsonl")}
    return [{"id": r["question_id"], "question": r["question"],
             "context": corpus[r["passage_id"]], "answers": r["answers"]}
            for r in read_jsonl(data_dir / f"qa_{split}.jsonl")]


def answer_chars(answer):
    """[start, end) của đáp án trên context, bỏ khoảng trắng thừa ở hai đầu.
    Nhiều đáp án gốc có dấu cách ở cuối (ví dụ 'Web ''World Wide Web'' ')."""
    text = answer["text"]
    lead = len(text) - len(text.lstrip())
    start = answer["start"] + lead
    return start, start + len(text.strip())


def _encode(examples, tokenizer, max_length, stride):
    return tokenizer([e["question"] for e in examples], [e["context"] for e in examples],
                     truncation="only_second", max_length=max_length, stride=stride,
                     return_overflowing_tokens=True, return_offsets_mapping=True)


def label_window(offsets, seq_ids, start_char, end_char):
    """Nhãn (start, end) theo chỉ số token trong MỘT cửa sổ.
    Chỉ gán khi toàn bộ đáp án nằm trong phần context của cửa sổ; ngược lại trả (0, 0) = [CLS]."""
    ctx = [i for i, s in enumerate(seq_ids) if s == 1]
    first, last = ctx[0], ctx[-1]
    if offsets[first][0] > start_char or offsets[last][1] < end_char:
        return 0, 0
    start = next(i for i in ctx if offsets[i][1] > start_char)       # token đầu tiên phủ ký tự bắt đầu
    end = next(i for i in reversed(ctx) if offsets[i][0] < end_char)  # token cuối cùng phủ ký tự kết thúc
    return start, end


def prepare_train_features(examples, tokenizer, max_length=384, stride=128, keep_offsets=False):
    """Mỗi ví dụ cho một hoặc nhiều cửa sổ; nhãn lấy đáp án đầu tiên (train chỉ có một).
    keep_offsets=True giữ thêm offsets để kiểm tra nhãn bằng mắt (không cần khi huấn luyện)."""
    enc = _encode(examples, tokenizer, max_length, stride)
    feats = []
    for i, ex_idx in enumerate(enc["overflow_to_sample_mapping"]):
        s_char, e_char = answer_chars(examples[ex_idx]["answers"][0])
        start, end = label_window(enc["offset_mapping"][i], enc.sequence_ids(i), s_char, e_char)
        feats.append({"input_ids": enc["input_ids"][i],
                      "token_type_ids": enc["token_type_ids"][i],
                      "start_positions": start, "end_positions": end,
                      "example_index": ex_idx})
        if keep_offsets:
            feats[-1]["offsets"] = enc["offset_mapping"][i]
    return feats


def prepare_eval_features(examples, tokenizer, max_length=384, stride=128):
    """Giống train nhưng không có nhãn; giữ offsets của token context (None cho phần còn lại)."""
    enc = _encode(examples, tokenizer, max_length, stride)
    feats = []
    for i, ex_idx in enumerate(enc["overflow_to_sample_mapping"]):
        seq_ids = enc.sequence_ids(i)
        feats.append({"input_ids": enc["input_ids"][i],
                      "token_type_ids": enc["token_type_ids"][i],
                      "offsets": [o if s == 1 else None for o, s in zip(enc["offset_mapping"][i], seq_ids)],
                      "example_index": ex_idx})
    return feats
