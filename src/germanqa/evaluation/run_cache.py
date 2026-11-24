"""File kết quả truy hồi top-k (GĐ12.2): ghi một lần, quét RRF/rerank nhiều lần trên file."""
from germanqa.data.io_utils import read_jsonl, write_jsonl
from germanqa.evaluation.retrieval_metrics import hit_at_k, mrr_at_k, recall_at_k


def validate_rows(rows, k, corpus_ids, allowed_question_ids=None):
    """rows: [{'question_id', 'hits': [[passage_id, score], ...]}]. Trả về danh sách lỗi (rỗng = hợp lệ)."""
    problems, seen_q = [], set()
    for r in rows:
        qid, hits = r["question_id"], r["hits"]
        if qid in seen_q:
            problems.append(f"{qid}: câu hỏi xuất hiện hai lần")
        seen_q.add(qid)
        if allowed_question_ids is not None and qid not in allowed_question_ids:
            problems.append(f"{qid}: không thuộc tập được phép (có thể là câu hỏi test)")
        if len(hits) > k:
            problems.append(f"{qid}: {len(hits)} kết quả > k={k}")
        pids = [h[0] for h in hits]
        if len(set(pids)) != len(pids):
            problems.append(f"{qid}: đoạn trùng trong danh sách")
        missing = [p for p in pids if p not in corpus_ids]
        if missing:
            problems.append(f"{qid}: {len(missing)} đoạn không có trong corpus")
        scores = [h[1] for h in hits]
        if any(a < b for a, b in zip(scores, scores[1:])):
            problems.append(f"{qid}: điểm tăng dần theo hạng (sai thứ tự)")
    return problems


def to_run(rows):
    """{question_id: [passage_id theo hạng]} từ các dòng của file top-k."""
    return {r["question_id"]: [h[0] for h in r["hits"]] for r in rows}


def per_question(run, qrels, qids, ks=(1, 5, 20, 100)):
    """{'hit@5': [0/1 theo từng câu], ...} theo ĐÚNG thứ tự qids (cần cho bootstrap ghép cặp)."""
    out = {f"hit@{k}": [hit_at_k(run[q], set(qrels[q]), k) for q in qids] for k in ks}
    out["recall@20"] = [recall_at_k(run[q], set(qrels[q]), 20) for q in qids]
    out["mrr@10"] = [mrr_at_k(run[q], set(qrels[q]), 10) for q in qids]
    return out

def write_run(path, rows):
    write_jsonl(path, rows)


def read_run(path):
    return read_jsonl(path)


def as_ranked(rows):
    """{question_id: [passage_id theo hạng]} (cùng nghĩa với to_run)."""
    return to_run(rows)
