"""GĐ12.3: so text384 với text256 bằng bootstrap GHÉP CẶP trên tập chọn; đếm câu lật kết quả; bảng 2x2 BM25 vs dense.
Ghi results/12b_analysis.json (12e đọc khóa C_paired_384_minus_256)."""
from germanqa.config import PROCESSED_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.evaluation.bootstrap import paired_bootstrap
from germanqa.evaluation.retrieval_metrics import hit_at_k
from germanqa.evaluation.run_cache import per_question, to_run


def main():
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    ids = sel["train"] + sel["dev"]
    qrels = {}
    for split in ("train", "dev"):
        qrels.update(read_json(PROCESSED_DIR / f"qrels_{split}.json"))
    b0 = to_run(read_jsonl(RESULTS_DIR / "B0_select_top100.jsonl"))
    r256 = to_run(read_jsonl(RESULTS_DIR / "B1_select_top100.jsonl"))
    r384 = to_run(read_jsonl(RESULTS_DIR / "B1_select_top100_384.jsonl"))
    p256, p384 = per_question(r256, qrels, ids), per_question(r384, qrels, ids)

    out = {"n": len(ids), "C_paired_384_minus_256": {}, "flips_hit@5": {}, "A_overlap": {}}
    for m in ("hit@1", "hit@5", "hit@20", "hit@100"):
        out["C_paired_384_minus_256"][m] = dict(zip(("diff", "lo", "hi"), paired_bootstrap(p384[m], p256[m])))
        d = out["C_paired_384_minus_256"][m]
        print(f"{m:8} 384 - 256 = {d['diff']*100:+.2f} điểm  [{d['lo']*100:+.2f}; {d['hi']*100:+.2f}]")
    a, b = p256["hit@5"], p384["hit@5"]
    out["flips_hit@5"] = {"miss256_hit384": sum(1 for x, y in zip(a, b) if x == 0 and y == 1),
                          "hit256_miss384": sum(1 for x, y in zip(a, b) if x == 1 and y == 0)}
    for k in (5, 20):                                          # bảng 2x2: nhánh nào trúng
        bm = [hit_at_k(b0[q], set(qrels[q]), k) for q in ids]
        dn = [hit_at_k(r256[q], set(qrels[q]), k) for q in ids]
        out["A_overlap"][f"hit@{k}"] = {"both": sum(x and y for x, y in zip(bm, dn)),
                                        "dense_only": sum((not x) and y for x, y in zip(bm, dn)),
                                        "bm25_only": sum(x and (not y) for x, y in zip(bm, dn)),
                                        "neither": sum((not x) and (not y) for x, y in zip(bm, dn))}
    print("lật Hit@5:", out["flips_hit@5"], "\n2x2:", out["A_overlap"])
    write_json(RESULTS_DIR / "12b_analysis.json", out)


if __name__ == "__main__":
    main()
