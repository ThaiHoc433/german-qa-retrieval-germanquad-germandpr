"""GĐ14.7b: đo cơ chế sau khi đọc tay 14g (số liệu, không chỉ kể ca). Chỉ dev-unseen. Ghi results/14j_error_breakdown.json (không ghi đè).
Câu hỏi: Hit@5 tăng 7,3 điểm nhưng EM không đổi. Vì sao?
  1. reader chọn đoạn đúng thường xuyên đến mức nào khi đoạn đúng CÓ trong top-5 (H0 so với H1)?
  2. khi top-5 đổi thành phần, EM đổi theo hướng nào?
  3. trong số câu có đoạn đúng trong top-5 mà vẫn sai EM, lỗi thuộc loại nào (chọn nhầm đoạn / cùng đoạn nhưng lệch ranh giới / khác hẳn)?"""
import numpy as np

from germanqa.config import PROCESSED_DIR, REPORT_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import score_answers


def main():
    out_path = RESULTS_DIR / "14j_error_breakdown.json"
    if out_path.exists():
        raise SystemExit(f"{out_path.name} đã tồn tại")
    unseen = set(read_json(RESULTS_DIR / "select_question_ids.json")["dev"])
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    exs = [e for e in load_examples(PROCESSED_DIR, "dev") if e["id"] in unseen]
    ids = [e["id"] for e in exs]
    run = {n: {r["question_id"]: r for r in read_jsonl(REPORT_DIR / f"pipeline_dev_unseen_{n}_predictions.jsonl")} for n in ("H0", "H1")}
    sc = {n: dict(zip(ids, score_answers({q: {"text": run[n][q]["answer"]} for q in ids}, exs))) for n in run}
    def top5(n, q):
        key = "reranked" if n == "H1" else "retrieved"
        return [x["passage_id"] for x in run[n][q][key][:5]]
    res = {"n": len(ids)}
    for n in ("H0", "H1"):
        gold_in = [q for q in ids if set(qrels[q]) & set(top5(n, q))]
        picked_gold = [q for q in gold_in if run[n][q]["passage_id"] in qrels[q]]
        wrong = [q for q in gold_in if not sc[n][q][0]]
        from_other = [q for q in wrong if run[n][q]["passage_id"] not in qrels[q]]
        same_passage = [q for q in wrong if run[n][q]["passage_id"] in qrels[q]]
        res[n] = {"gold_in_top5": len(gold_in), "reader_picks_gold_passage": len(picked_gold),
                  "pct_picks_gold_given_gold_in_top5": round(100 * len(picked_gold) / len(gold_in), 1),
                  "em": round(float(np.mean([sc[n][q][0] for q in ids])), 4), "f1": round(float(np.mean([sc[n][q][1] for q in ids])), 4),
                  "gold_in_top5_but_em_wrong": len(wrong), "of_which_answer_from_other_passage": len(from_other),
                  "of_which_same_gold_passage_wrong_span": len(same_passage),
                  "same_passage_wrong_span_f1_ge_0.5": sum(sc[n][q][1] >= 0.5 for q in same_passage),
                  "em_when_gold_in_top5": round(float(np.mean([sc[n][q][0] for q in gold_in])), 4)}
    changed = [q for q in ids if set(top5("H0", q)) != set(top5("H1", q))]
    same = [q for q in ids if q not in set(changed)]
    def em_of(n, qs):
        return round(float(np.mean([sc[n][q][0] for q in qs])), 4) if qs else None
    res["top5_composition"] = {"changed": len(changed), "unchanged": len(same),
                               "em_H0_on_changed": em_of("H0", changed), "em_H1_on_changed": em_of("H1", changed),
                               "em_H0_on_unchanged": em_of("H0", same), "em_H1_on_unchanged": em_of("H1", same),
                               "em_flips_on_changed": {"H0_right_H1_wrong": sum(sc["H0"][q][0] > sc["H1"][q][0] for q in changed),
                                                       "H0_wrong_H1_right": sum(sc["H1"][q][0] > sc["H0"][q][0] for q in changed)},
                               "em_flips_on_unchanged": sum(sc["H0"][q][0] != sc["H1"][q][0] for q in same)}
    gained = [q for q in ids if not (set(qrels[q]) & set(top5("H0", q))) and (set(qrels[q]) & set(top5("H1", q)))]
    res["questions_where_h1_adds_gold_to_top5"] = {"n": len(gained), "em_H0": em_of("H0", gained), "em_H1": em_of("H1", gained),
                                                    "h1_picks_gold_passage": sum(run["H1"][q]["passage_id"] in qrels[q] for q in gained)}
    write_json(out_path, res)
    import json
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
