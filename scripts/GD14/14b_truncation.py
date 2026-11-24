"""GĐ14.3: kiểm cắt token. Với mỗi họ tokenizer của ứng viên, đo trên pool top-20 của tập chọn:
  * bao nhiêu cặp (câu hỏi, đoạn) dài hơn giới hạn (nghĩa là bị cắt nếu để mặc định 'truncate');
  * với đoạn ĐÚNG: bao nhiêu đoạn bị cắt, và trong đó bao nhiêu đáp án (token kết thúc của đáp án) nằm NGOÀI phần giữ lại.
Chỉ dùng độ dài và vị trí đáp án; không dùng điểm của bất kỳ reranker nào. Ghi results/14b_truncation.json."""
import json
from collections import defaultdict

from transformers import AutoTokenizer

from germanqa.config import PROCESSED_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.rerank.cross_encoder import MAX_QUERY_TOKENS
from germanqa.rerank.hub import local_snapshot
from germanqa.rerank.pool import hybrid_pool

REPOS = {"mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "bge_m3": "BAAI/bge-reranker-v2-m3",
         "gbert_dpr": "deepset/gbert-base-germandpr-reranking"}
MAX_LEN = {"mmarco": 512, "bge_m3": 512, "gbert_dpr": 510}


def main():
    out_path = RESULTS_DIR / "14b_truncation.json"
    if out_path.exists():
        raise SystemExit(f"{out_path.name} đã tồn tại: không ghi đè")
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    qids = list(sel["train"]) + list(sel["dev"])
    qa = {}
    for split in ("train", "dev"):
        for r in read_jsonl(PROCESSED_DIR / f"qa_{split}.jsonl"):
            qa[r["question_id"]] = r
    corpus = {r["passage_id"]: r["original_text"] for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}
    pool = hybrid_pool(qids)
    res = {"n_questions": len(qids), "pairs_per_question": 20, "text_seen_by_reranker": "original_text", "models": {}}
    for key, repo in REPOS.items():
        tok = AutoTokenizer.from_pretrained(local_snapshot(repo), use_fast=True)
        add = tok.num_special_tokens_to_add(pair=True)
        cap = MAX_LEN[key]
        st = defaultdict(int)
        lens = []
        gold_lost, gold_over = [], []
        for q in qids:
            r = qa[q]
            enc_q = tok(r["question"], add_special_tokens=False, return_offsets_mapping=True)
            qq = r["question"] if len(enc_q["offset_mapping"]) <= MAX_QUERY_TOKENS else r["question"][:enc_q["offset_mapping"][MAX_QUERY_TOKENS - 1][1]]
            q_len = min(len(enc_q["input_ids"]), MAX_QUERY_TOKENS)
            texts = [corpus[p] for p in pool[q]]
            L = [len(x) for x in tok([qq] * len(texts), texts, truncation=False, add_special_tokens=True)["input_ids"]]
            lens += L
            over_q = [l > cap for l in L]
            st["pairs"] += len(L)
            st["pairs_over"] += sum(over_q)
            st["questions_with_overflow"] += int(any(over_q))
            if r["passage_id"] in pool[q]:                                       # đoạn đúng nằm trong pool
                i = pool[q].index(r["passage_id"])
                st["gold_in_pool"] += 1
                gold_over.append(over_q[i])
                if over_q[i]:
                    st["gold_over"] += 1
                    keep = cap - q_len - add                                       # số token của đoạn còn lại sau khi cắt
                    offs = tok(texts[i], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                    a = r["answers"][0]
                    a_end = a["start"] + len(a["text"])
                    end_tok = next((k for k, (s, e) in enumerate(offs) if e >= a_end), len(offs) - 1)
                    lost = end_tok >= keep
                    st["gold_answer_lost"] += int(lost)
        lens.sort()
        res["models"][key] = {
            "max_length": cap, "pairs": st["pairs"], "pairs_over_cap": st["pairs_over"],
            "pct_pairs_over_cap": round(100 * st["pairs_over"] / st["pairs"], 3),
            "questions_with_any_overflow": st["questions_with_overflow"],
            "gold_in_pool": st["gold_in_pool"], "gold_over_cap": st["gold_over"],
            "gold_answer_lost_by_truncation": st["gold_answer_lost"],
            "pct_gold_answer_lost": round(100 * st["gold_answer_lost"] / st["gold_in_pool"], 3),
            "length_percentiles": {p: lens[int(len(lens) * p / 100) - 1] for p in (50, 90, 99, 100)},
        }
        print(key, json.dumps(res["models"][key], ensure_ascii=False), flush=True)
    write_json(out_path, res)


if __name__ == "__main__":
    main()
