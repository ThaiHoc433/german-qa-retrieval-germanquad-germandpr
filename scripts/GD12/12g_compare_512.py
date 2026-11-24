"""GĐ12.3 bổ sung: so text512 với text384 trên tập chọn (2.223 câu) bằng bootstrap GHÉP CẶP và ĐÚNG quy tắc nhận của GĐ12
(decision.adopt_alternative): Hit@5 chênh ghép cặp > 0 với cận dưới > 0, và Hit@20 cận dưới >= -0,01.
Quy tắc được ghi ở đây TRƯỚC khi chạy. Đây là THÍ NGHIỆM PHỤ có nhãn: kể cả khi 512 được 'nhận' theo quy tắc, script KHÔNG đổi configs/dense.yaml.
Vì sao không tự đổi: nếu dense đổi thì B1, H0, H1, quét RRF và mọi điểm reranker đã chấm đều phải làm lại; đó là quyết định của mày, không của script.
Đọc results/B1_select_top100_384.jsonl và results/B1_select_top100_512.jsonl (12a_dense_select.py --artifact text512 --tag 512).
Ghi results/12g_512_vs_384.json (không ghi đè). Chỉ train + dev, không test."""
import time

from germanqa.config import PROCESSED_DIR, RESULTS_DIR
from germanqa.data.io_utils import file_sha1, read_json, read_jsonl, write_json
from germanqa.evaluation.bootstrap import paired_bootstrap
from germanqa.evaluation.decision import adopt_alternative
from germanqa.evaluation.run_cache import per_question, to_run

METRICS = ("hit@1", "hit@5", "hit@20", "hit@100", "mrr@10", "recall@20")
REDO_IF_ADOPTED = ["artifacts/dense (đổi text384 -> text512 trong configs/dense.yaml)", "results/B1_select_top100_384.jsonl -> bản 512",
                   "13_prereg + quét RRF (13a) + xác nhận (13b) vì hybrid dựa trên dense", "baselines/B1, H0, H1 (manifest)",
                   "14c: điểm reranker của H1 (pool top-20 của H0 đổi)", "14d, 14e, 14f"]


def main():
    out_path = RESULTS_DIR / "12g_512_vs_384.json"
    if out_path.exists():
        raise SystemExit(f"{out_path.name} đã tồn tại: không ghi đè")
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    ids = sel["train"] + sel["dev"]
    qrels = {}
    for split in ("train", "dev"):
        qrels.update(read_json(PROCESSED_DIR / f"qrels_{split}.json"))
    f384, f512 = RESULTS_DIR / "B1_select_top100_384.jsonl", RESULTS_DIR / "B1_select_top100_512.jsonl"
    r384, r512 = to_run(read_jsonl(f384)), to_run(read_jsonl(f512))
    p384, p512 = per_question(r384, qrels, ids), per_question(r512, qrels, ids)
    paired = {m: dict(zip(("diff", "lo", "hi"), paired_bootstrap(p512[m], p384[m]))) for m in METRICS}
    means = {m: {"text384": sum(p384[m]) / len(ids), "text512": sum(p512[m]) / len(ids)} for m in METRICS}
    ci = lambda m: (paired[m]["diff"], paired[m]["lo"], paired[m]["hi"])
    ok, reason = adopt_alternative(ci("hit@5"), ci("hit@20"), margin=0.01)
    flips = {"hit@5_512_right_384_wrong": sum(a > b for a, b in zip(p512["hit@5"], p384["hit@5"])),
             "hit@5_384_right_512_wrong": sum(b > a for a, b in zip(p512["hit@5"], p384["hit@5"]))}
    res = {"purpose": "thí nghiệm phụ: có nên dùng cắt 512 thay 384?", "n": len(ids), "means": means, "paired_512_minus_384": paired, "flips": flips,
           "rule": {"function": "decision.adopt_alternative", "margin": 0.01, "primary": "hit@5 cận dưới > 0", "secondary": "hit@20 cận dưới >= -0.01"},
           "adopt_by_rule": ok, "reason": reason, "config_changed": False,
           "if_adopted_must_redo": REDO_IF_ADOPTED, "inputs_sha1": {"B1_select_top100_384.jsonl": file_sha1(f384), "B1_select_top100_512.jsonl": file_sha1(f512)},
           "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(out_path, res)
    for m in METRICS:
        d = paired[m]
        print(f"{m:9} 384 {means[m]['text384']:.4f}  512 {means[m]['text512']:.4f}  chênh {d['diff']*100:+.2f} [{d['lo']*100:+.2f}; {d['hi']*100:+.2f}]")
    print("lật Hit@5:", flips)
    print("THEO QUY TẮC GĐ12:", "NHẬN" if ok else "KHÔNG NHẬN", "|", reason, "| configs/dense.yaml KHÔNG bị đổi")


if __name__ == "__main__":
    main()
