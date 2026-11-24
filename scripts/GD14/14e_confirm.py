"""GĐ14.5b: xác nhận MỘT lần cấu hình đã chọn ở 14d trên 234 câu dev-unseen (Hit@5, ghép cặp với H0).
Chỉ chạy khi results/14_selection.json đã tồn tại; chỉ đọc điểm của model đã chọn; ghi results/14_confirmation.json một lần.
Không mở test. Kết luận theo prereg: cận dưới Hit@5 > 0 -> 'H1 hơn H0 về Hit@5', ngược lại 'chưa đủ bằng chứng'."""
import time

from germanqa.config import PROCESSED_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.evaluation.fusion_select import require_prereg
from germanqa.rerank import analysis as A
from germanqa.rerank.exposure import GROUPS, question_group
from germanqa.rerank.pool import hybrid_pool


def main():
    pre = require_prereg(RESULTS_DIR / "14_prereg.json", ROOT)
    out_path = RESULTS_DIR / "14_confirmation.json"
    if out_path.exists():
        print("14_confirmation.json đã có: đọc lại, KHÔNG tính lại (xác nhận chỉ một lần)")
        return read_json(out_path)
    sel_path = RESULTS_DIR / "14_selection.json"
    if not sel_path.exists():
        raise SystemExit("chưa có 14_selection.json: phải chọn cấu hình ở 14d trước khi xem dev-unseen")
    chosen = read_json(sel_path)["chosen"]
    boot = {k: pre["bootstrap"][k] for k in ("n_boot", "confidence", "seed")}
    qids = list(read_json(RESULTS_DIR / "select_question_ids.json")["dev"])
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    corpus = {r["passage_id"]: r for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}
    pool = hybrid_pool(qids)
    rows = A.load_scores(RESULTS_DIR / f"14c_scores_{chosen['model']}.jsonl", qids)
    bad = A.integrity_problems(rows, pool, qids)
    if bad:
        raise SystemExit(f"dữ liệu điểm không toàn vẹn: {bad[:3]}")
    pq0, m0 = A.metrics(A.h0_run(pool, qids), qrels, qids)
    pq1, m1 = A.metrics(A.reranked_run(rows, qids, chosen["overflow"]), qrels, qids)
    groups = {q: question_group(qrels[q], corpus) for q in qids}
    idx = {g: [i for i, q in enumerate(qids) if groups[q] == g] for g in GROUPS}
    gains = {g: dict(zip(("diff", "lo", "hi"), A.gain_ci(pq1, pq0, "hit@5", idx=idx[g], **boot))) for g in GROUPS}
    paired = {m: dict(zip(("diff", "lo", "hi"), A.gain_ci(pq1, pq0, m, **boot))) for m in ("hit@1", "hit@5", "hit@20", "mrr@10")}
    h5 = paired["hit@5"]
    out = {"split": "dev-unseen", "n": len(qids), "config": chosen, "h0": m0, "h1": m1, "paired_h1_minus_h0": paired,
           "hit5_gain_by_group": gains, "group_sizes": {g: len(v) for g, v in idx.items()},
           "verdict": "H1 hơn H0 về Hit@5 (cận dưới > 0)" if h5["lo"] > 0 else "chưa đủ bằng chứng H1 hơn H0 về Hit@5",
           "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(out_path, out)
    return out


if __name__ == "__main__":
    o = main()
    print("cấu hình:", o["config"]["model"], o["config"]["overflow"], "| n =", o["n"])
    print("H0 hit@5 %.4f -> H1 %.4f | chênh ghép cặp %+.4f [%+.4f; %+.4f]" % (
        o["h0"]["hit@5"], o["h1"]["hit@5"], *(o["paired_h1_minus_h0"]["hit@5"][k] for k in ("diff", "lo", "hi"))))
    print("theo nhóm:", {g: (round(v["diff"], 4), round(v["lo"], 4), round(v["hi"], 4)) for g, v in o["hit5_gain_by_group"].items()})
    print("KẾT LUẬN:", o["verdict"])
