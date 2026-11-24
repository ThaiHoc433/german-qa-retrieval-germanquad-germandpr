"""Hỏi reader với đoạn văn TỰ CHỌN (không truy hồi). Dùng đúng reader và cấu hình của pipeline
(configs/pipeline.yaml + configs/reader.yaml), rồi so với đáp án mong đợi bằng EM/F1 của dự án.

Ví dụ:
  python scripts/ask_context.py -q "Wann wurde die Berliner Mauer gebaut?" ^
      -c "Die Berliner Mauer wurde am 13. August 1961 errichtet und trennte West-Berlin von Ost-Berlin." ^
      -a "Am 13. August 1961"
  python scripts/ask_context.py --file cases.jsonl     # mỗi dòng {"question", "context", "answer"?}
"""
import argparse
import json
from pathlib import Path

from germanqa.config import ROOT
from germanqa.experiment import load_config
from germanqa.reader.qa_metrics import exact_match, f1_score


def build_reader(checkpoint=None):
    from germanqa.reader.inference import Reader          # torch chỉ nạp khi cần
    cfg, rcfg = load_config("pipeline"), load_config("reader")["reader"]
    ck = Path(checkpoint or cfg["pipeline"]["checkpoint"])
    ck = ck if ck.is_absolute() else ROOT / ck
    reader = Reader(ck, rcfg["max_length"], rcfg["stride"], rcfg["n_best"], cfg["reader"]["max_answer_tokens"],
                    cfg["reader"]["score_mode"], rcfg["eval_batch_size"])
    return reader, ck.name


def ask(reader, question, context):
    r = reader.read(question, [{"passage_id": "user", "text": context}])[0]
    return (r["text"] if r["start"] >= 0 else ""), r["start"], r["end"], r["score"]


def main(cases, checkpoint):
    reader, name = build_reader(checkpoint)
    print(f"Reader: {name}\n")
    ems, f1s = [], []
    for i, c in enumerate(cases, 1):
        pred, s, e, score = ask(reader, c["question"], c["context"])
        print(f"[{i}] Hỏi     : {c['question']}")
        if s >= 0:
            print(f"    Ngữ cảnh: {c['context'][:s]}[[{c['context'][s:e]}]]{c['context'][e:]}")
        print(f"    Đáp án  : {pred!r}   (điểm reader {score:.2f})")
        if c.get("answer"):
            em, f1 = exact_match(pred, c["answer"]), f1_score(pred, c["answer"])
            ems.append(em)
            f1s.append(f1)
            print(f"    Mong đợi: {c['answer']!r}   EM {float(em):.0f}  F1 {f1:.2f}")
        print()
    if len(ems) > 1:
        print(f"Tổng {len(ems)} câu: EM {sum(map(float, ems)) / len(ems):.3f}  F1 {sum(f1s) / len(f1s):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question")
    ap.add_argument("-c", "--context")
    ap.add_argument("-a", "--answer", default=None, help="đáp án mong đợi (tùy chọn) để tính EM/F1")
    ap.add_argument("--file", default=None, help="tệp .jsonl, mỗi dòng {question, context, answer?}")
    ap.add_argument("--checkpoint", default=None, help="mặc định: checkpoint trong pipeline.yaml")
    a = ap.parse_args()
    if a.file:
        cases = [json.loads(l) for l in Path(a.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif a.question and a.context:
        cases = [{"question": a.question, "context": a.context, "answer": a.answer}]
    else:
        ap.error("cần -q và -c, hoặc --file")
    main(cases, a.checkpoint)
