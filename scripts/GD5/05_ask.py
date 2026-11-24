"""GĐ5.5: hỏi một câu. In đáp án, đoạn chứa nó, điểm và thời gian từng bước."""
import argparse
import json

from germanqa.pipeline import build_pipeline


def main(question, checkpoint, trace):
    pipe = build_pipeline(checkpoint)
    pipe.answer("Wer war Konrad Adenauer?")                    # chạy khởi động: lần đầu luôn chậm hơn (CUDA, bộ nhớ đệm)
    r = pipe.answer(question, trace=trace)
    print(f"Hỏi     : {r.question}")
    print(f"Đáp án  : {r.answer!r}   (trạng thái: {r.status})")
    print(f"Đoạn    : {r.passage_id}  [{r.answer_start}:{r.answer_end}]")
    if r.passage_text:
        a, b = r.answer_start, r.answer_end
        print(f"Ngữ cảnh: ...{r.passage_text[max(0, a - 80):a]}[[{r.passage_text[a:b]}]]{r.passage_text[b:b + 80]}...")
    print(f"Điểm    : BM25 {r.retrieval_score:.2f}, reader {r.reader_score:.2f}")
    print(f"Thời gian: truy hồi {r.retrieval_time * 1000:.1f} ms, reader {r.reader_time * 1000:.1f} ms, "
          f"tổng {r.total_time * 1000:.1f} ms")
    print(f"Phiên bản: mô hình {r.model_version}, corpus {r.corpus_version}, thí nghiệm {r.experiment_id}")
    if trace:
        print(json.dumps(r.trace["candidates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    main(a.question, a.checkpoint, a.trace)
