
"""GĐ7.4-7.5: phân loại câu trả lời sai từ file dự đoán. Rút mẫu để đọc thủ công CHỈ từ dev."""
import argparse
import random
from collections import Counter

from germanqa.config import PROCESSED_DIR, REPORT_DIR, SEED
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.evaluation.errors import RELATIONS, ROOT_CAUSES, STAGES, classify, coarse_stage
from germanqa.experiment import load_config
from germanqa.experiment_rules import guard_or_exit
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import exact_match


def snippet(text, start, end, width=160):
    return (text[max(0, start - width):start] + "[[" + text[start:end] + "]]" + text[end:end + width]).replace("\n", " ")


def main(split, confirm_test=False):
    if split == "test":
        guard_or_exit("07b_error_analysis.py", "phân loại lỗi trên dự đoán test; KHÔNG dùng để thiết kế (xem experiment_rules)", confirm_test,
                      kind="error_analysis", files=["pipeline_test_predictions.jsonl", "qrels_test.json"])
    ev = load_config("evaluation")["evaluation"]
    rows = {r["question_id"]: r for r in read_jsonl(REPORT_DIR / f"pipeline_{split}_predictions.jsonl")}
    examples = load_examples(PROCESSED_DIR, split)
    qrels = read_json(PROCESSED_DIR / f"qrels_{split}.json")
    passages = {r["passage_id"]: r["original_text"] for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}

    wrong = []
    for ex in examples:
        row = rows[ex["id"]]
        refs = [a["text"] for a in ex["answers"]]
        if max(exact_match(row["answer"], r) for r in refs) == 1.0:
            continue
        wrong.append({"id": ex["id"], "refs": ex["answers"], **classify(row, qrels[ex["id"]], refs, passages)})

    by_stage = Counter(w["stage"] for w in wrong)
    by_coarse = Counter(coarse_stage(w["stage"]) for w in wrong)      # 3 loại của plan (retrieval/selection/reader)
    table = {s: {rel: sum(1 for w in wrong if w["stage"] == s and w["relation"] == rel) for rel in RELATIONS} for s in STAGES}
    flags = {s: {"answer_in_selected_passage": sum(w["answer_in_selected_passage"] for w in wrong if w["stage"] == s),
                 "answer_in_other_retrieved": sum(w["answer_in_other_retrieved"] for w in wrong if w["stage"] == s),
                 "f1_at_least_0.5": sum(w["f1"] >= 0.5 for w in wrong if w["stage"] == s),
                 "article_only": sum(w["article_only"] for w in wrong if w["stage"] == s)} for s in STAGES}
    print(f"{split}: {len(examples)} câu, {len(wrong)} câu sai (EM = 0)")
    for s in STAGES:
        print(f"{s:24} {by_stage[s]:4}", table[s], flags[s])
    assert sum(by_stage.values()) == len(wrong)
    print("3 loại của plan:", dict(by_coarse))
    write_json(REPORT_DIR / f"error_analysis_{split}.json", {"wrong": len(wrong), "by_stage": dict(by_stage),
                                                             "by_coarse_stage": dict(by_coarse),
                                                             "stage_x_relation": table, "flags": flags})
    if split != "dev":
        print("Không rút mẫu đọc thủ công từ", split, "(chỉ đọc mẫu từ dev để không tinh chỉnh theo test).")
        return

    rng = random.Random(SEED)
    q_of = {ex["id"]: ex for ex in examples}
    lines = [f"# Mẫu lỗi dev (seed {SEED}, tối đa {ev['error_samples_per_class']} câu mỗi loại)\n"]
    samples = {}
    for s in STAGES:
        pool = [w for w in wrong if w["stage"] == s]
        pick = rng.sample(pool, min(ev["error_samples_per_class"], len(pool)))
        samples[s] = [w["id"] for w in pick]
        lines.append(f"\n## {s}: {len(pool)} câu sai, in {len(pick)}\n")
        for n, w in enumerate(pick, 1):
            ex, row = q_of[w["id"]], rows[w["id"]]
            gold_pid = qrels[w["id"]][0]
            ids = [h["passage_id"] for h in row["retrieved"]]
            g = w["refs"][0]
            rank = ids.index(gold_pid) + 1 if gold_pid in ids else None
            lines += [f"### {s} #{n}  ({w['id']})", f"- Câu hỏi: {ex['question']}",
                      f"- Đáp án đúng: {[a['text'] for a in w['refs']]}",
                      f"- Dự đoán: {row['answer']!r}  | quan hệ: {w['relation']}, F1 {w['f1']:.2f}",
                      f"- Hạng đoạn đúng trong truy hồi: {rank}; đoạn reader chọn ở hạng: "
                      f"{ids.index(row['passage_id']) + 1 if row['passage_id'] in ids else None}",
                      f"- Cờ: đoạn đã chọn có chứa đáp án = {w['answer_in_selected_passage']}, "
                      f"đoạn truy hồi ngoài qrels có chứa đáp án = {w['answer_in_other_retrieved']}",
                      f"- Ngữ cảnh đúng: {snippet(passages[gold_pid], g['start'], g['start'] + len(g['text']))}"]
            if row["passage_id"] and row["passage_id"] != gold_pid:
                lines.append(f"- Ngữ cảnh đã chọn: {snippet(passages[row['passage_id']], row['answer_start'], row['answer_end'])}")
            lines.append("")
    (REPORT_DIR / "error_samples_dev.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_json(REPORT_DIR / "error_samples_dev_ids.json", samples)

    # Mẫu nhãn nguyên nhân gốc (bước "nhóm"): người đọc chép thành error_labels_dev.json rồi điền primary/secondary/note.
    # Chỉ ghi file .template, không bao giờ ghi đè file nhãn đã điền.
    w_of = {w["id"]: w for w in wrong}
    template = [{"question_id": q, "stage": s, "coarse": coarse_stage(s), "question": q_of[q]["question"],
                 "refs": [a["text"] for a in w_of[q]["refs"]], "prediction": rows[q]["answer"],
                 "relation": w_of[q]["relation"], "f1": round(w_of[q]["f1"], 3),
                 "answer_in_selected_passage": w_of[q]["answer_in_selected_passage"],
                 "answer_in_other_retrieved": w_of[q]["answer_in_other_retrieved"],
                 "primary": None, "secondary": [], "note": ""}
                for s in STAGES for q in samples[s]]
    write_json(REPORT_DIR / "error_labels_dev.template.json", {"codebook": ROOT_CAUSES, "labels": template})
    print("đã ghi error_analysis_dev.json, error_samples_dev.md, error_labels_dev.template.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    main(a.split, a.confirm_test)