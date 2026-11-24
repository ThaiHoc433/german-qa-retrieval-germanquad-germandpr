"""GĐ14.7: gom các ca để ĐỌC TAY (chỉ dev-unseen, từ kết quả 14f). Chọn theo quy tắc cố định, không chọn theo cảm tính:
  A. EM: H0 đúng, H1 sai (reranker làm hỏng)      B. EM: H0 sai, H1 đúng (reranker cứu)
  C. đoạn đúng tụt từ top-5 xuống ngoài top-5 sau khi xếp lại (Hit@5: H0 có, H1 không)
  D. đoạn đúng lên top-5 từ ngoài top-5 (Hit@5: H0 không, H1 có)
  E. H1 có đoạn đúng trong top-5 nhưng đáp án sai (lỗi nằm ở reader, không phải ở xếp hạng)
Mỗi nhóm lấy tối đa 12 ca theo thứ tự question_id. Ghi reports/error_cases_H1_dev_unseen.json và .md (để đọc)."""
from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import score_answers

MAX_PER_GROUP = 12


def main():
    unseen = set(read_json(RESULTS_DIR / "select_question_ids.json")["dev"])
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    exs = [e for e in load_examples(PROCESSED_DIR, "dev") if e["id"] in unseen]
    corpus = {r["passage_id"]: r["original_text"] for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}
    runs = {n: {r["question_id"]: r for r in read_jsonl(REPORT_DIR / f"pipeline_dev_unseen_{n}_predictions.jsonl")} for n in ("H0", "H1")}
    em = {n: {e["id"]: s[0] for e, s in zip(exs, score_answers({q: {"text": r["answer"]} for q, r in runs[n].items()}, exs))} for n in runs}
    refs = {e["id"]: [a["text"] for a in e["answers"]] for e in exs}
    groups = {k: [] for k in "ABCDE"}
    for q in sorted(unseen):
        gold = set(qrels[q])
        t0 = [h["passage_id"] for h in runs["H0"][q]["retrieved"][:5]]
        t1 = [x["passage_id"] for x in runs["H1"][q]["reranked"][:5]]
        in0, in1 = bool(gold & set(t0)), bool(gold & set(t1))
        pos = {x["passage_id"]: x for x in runs["H1"][q]["reranked"]}
        gp = next((p for p in qrels[q] if p in pos), None)
        info = {"question_id": q, "question": runs["H1"][q]["question"], "refs": refs[q],
                "H0_answer": runs["H0"][q]["answer"], "H1_answer": runs["H1"][q]["answer"],
                "H0_em": em["H0"][q], "H1_em": em["H1"][q], "gold_in_top5": {"H0": in0, "H1": in1},
                "gold_rank_before": pos[gp]["rank_before"] if gp else None, "gold_rank_after": pos[gp]["rank"] if gp else None,
                "gold_rerank_score": pos[gp]["rerank_score"] if gp else None,
                "H1_top1_rerank_score": runs["H1"][q]["reranked"][0]["rerank_score"],
                "gold_text_head": corpus[qrels[q][0]][:300]}
        if em["H0"][q] and not em["H1"][q]:
            groups["A"].append(info)
        if not em["H0"][q] and em["H1"][q]:
            groups["B"].append(info)
        if in0 and not in1:
            groups["C"].append(info)
        if in1 and not in0:
            groups["D"].append(info)
        if in1 and not em["H1"][q]:
            groups["E"].append(info)
    full = {k: len(v) for k, v in groups.items()}
    out = {"group_sizes_full": full, "note": "mỗi nhóm cắt còn 12 ca đầu theo question_id", "cases": {k: v[:MAX_PER_GROUP] for k, v in groups.items()}}
    write_json(REPORT_DIR / "error_cases_H1_dev_unseen.json", out)
    names = {"A": "H0 đúng EM, H1 sai", "B": "H0 sai EM, H1 đúng", "C": "đoạn đúng rời top-5", "D": "đoạn đúng vào top-5",
             "E": "H1 có đoạn đúng trong top-5 nhưng sai đáp án"}
    md = ["# Ca đọc tay H1 (dev-unseen)", ""]
    for k, v in out["cases"].items():
        md += [f"## {k}. {names[k]} — {full[k]} ca (hiện {len(v)})", ""]
        for c in v:
            md += [f"- **{c['question_id']}** {c['question']}", f"  - đáp án chuẩn: {c['refs']}", f"  - H0: {c['H0_answer']!r} | H1: {c['H1_answer']!r}",
                   f"  - đoạn đúng: hạng {c['gold_rank_before']} -> {c['gold_rank_after']} (điểm {c['gold_rerank_score']})"]
        md.append("")
    (REPORT_DIR / "error_cases_H1_dev_unseen.md").write_text("\n".join(md), encoding="utf-8")
    print("kích thước nhóm:", full)


if __name__ == "__main__":
    main()
