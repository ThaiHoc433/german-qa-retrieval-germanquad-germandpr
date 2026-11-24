"""GĐ13.3: quét RRF OFFLINE trên danh sách top-100 đã lưu (không cần reader, GPU hay encoder).
CHỈ dùng 1.989 câu train trong tập chọn để chọn; dev-unseen KHÔNG được đọc ở đây (xác nhận ở 13b).
Ghi: results/rrf_sweep.json, results/13_selection.json, và cập nhật configs/hybrid.yaml."""
import argparse
import time

from germanqa.config import CONFIG_DIR, PROCESSED_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, write_json
from germanqa.evaluation.bootstrap import paired_bootstrap
from germanqa.evaluation.decision import adopt_alternative
from germanqa.evaluation.fusion_select import choose_config, require_prereg
from germanqa.evaluation.run_cache import as_ranked, per_question, read_run
from germanqa.retrieval.rrf import rrf

METRICS = ("hit@1", "hit@5", "hit@20", "recall@20", "mrr@10")


def fuse(bm, dn, ids, c, depth, w_dense=1.0, top_k=20):
    return {q: [r["passage_id"] for r in rrf({"bm25": bm[q][:depth], "dense": dn[q][:depth]}, c=c, top_k=top_k,
                                             weights={"dense": w_dense})] for q in ids}


def evaluate(run, qrels, ids):
    pq = per_question(run, qrels, ids, ks=(1, 5, 20))
    return pq, {m: sum(pq[m]) / len(ids) for m in METRICS}


def paired(a, b, boot):
    out = {}
    for m in ("hit@5", "hit@20"):
        d, lo, hi = paired_bootstrap(a[m], b[m], **boot)
        out[m] = {"diff": d, "lo": lo, "hi": hi}
    return out


def write_hybrid_yaml(sel):
    text = f"""# Cấu hình hybrid retriever V1 (GĐ13): BM25 + dense hợp nhất bằng RRF.
# Giá trị do 13a_sweep.py chọn trên 1.989 câu train theo quy tắc ghi trước (results/13_prereg.json);
# lý do và số liệu: results/rrf_sweep.json. Chưa xác nhận trên dev-unseen khi file này được ghi.
hybrid:
  bm25_top_k: {sel['depth']}        # số đoạn lấy từ BM25 đưa vào RRF (độ sâu của nhánh)
  dense_top_k: {sel['depth']}       # số đoạn lấy từ dense đưa vào RRF
  fusion_top_k: 20      # số đoạn trả về sau RRF (khớp pipeline.retrieval_top_k; reader chỉ đọc reader_top_k = 5 đầu)
  rrf_c: {sel['rrf_c']}             # hằng số làm trơn của RRF; KHÔNG phải top-k
  w_dense: {float(sel['w_dense'])}          # trọng số nhánh dense (BM25 luôn 1); 1.0 = RRF thường
"""
    (CONFIG_DIR / "hybrid.yaml").write_text(text, encoding="utf-8", newline="\n")


def main(force):
    pre = require_prereg(RESULTS_DIR / "13_prereg.json", ROOT)
    if (RESULTS_DIR / "13_selection.json").exists() and not force:
        raise SystemExit("13_selection.json đã tồn tại: cấu hình đã được chọn. Quét lại cần --force và phải báo lý do.")
    t0 = time.perf_counter()
    boot = dict(n_boot=pre["bootstrap"]["n_boot"], confidence=pre["bootstrap"]["confidence"], seed=pre["bootstrap"]["seed"])
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    ids = list(sel["train"])                                        # CHỈ train: dev-unseen không được chạm ở đây
    qrels = {**read_json(PROCESSED_DIR / "qrels_train.json")}
    bm = as_ranked(read_run(RESULTS_DIR / "B0_select_top100.jsonl"))
    dn = as_ranked(read_run(RESULTS_DIR / "B1_select_top100_384.jsonl"))
    base = {}
    pq_base = {}
    for name, run in (("bm25", bm), ("dense384", dn)):
        pq_base[name], base[name] = evaluate({q: run[q] for q in ids}, qrels, ids)

    default = {"rrf_c": pre["default"]["rrf_c"], "depth": pre["default"]["depth"]}
    grid, pq_grid = [], {}
    for depth in pre["grid"]["depth"]:
        for c in pre["grid"]["rrf_c"]:
            pq, mean = evaluate(fuse(bm, dn, ids, c, depth), qrels, ids)
            pq_grid[(c, depth)] = pq
            grid.append({"rrf_c": c, "depth": depth, **mean})
    assert len(grid) == pre["grid"]["n_configs"]
    pq_def = pq_grid[(default["rrf_c"], default["depth"])]
    for g in grid:
        pq = pq_grid[(g["rrf_c"], g["depth"])]
        g["vs_dense"] = paired(pq, pq_base["dense384"], boot)
        g["vs_default"] = paired(pq, pq_def, boot)

    best = choose_config(grid, default)
    notes = []
    if (best["rrf_c"], best["depth"]) == (default["rrf_c"], default["depth"]):
        main_cfg, adopted = dict(default), None
        notes.append("cấu hình tốt nhất chính là mặc định: không cần thách đấu")
    else:
        pq_best = pq_grid[(best["rrf_c"], best["depth"])]
        ok, why = adopt_alternative(tuple(best["vs_default"]["hit@5"].values()), tuple(best["vs_default"]["hit@20"].values()),
                                    pre["adoption_rule"]["margin"])
        adopted = {"adopt": ok, "reason": why}
        main_cfg = {"rrf_c": best["rrf_c"], "depth": best["depth"]} if ok else dict(default)
        notes.append(f"cấu hình tốt nhất ({best['rrf_c']}, {best['depth']}) {'được nhận' if ok else 'không được nhận'} so với mặc định: {why}")
    main_key = (main_cfg["rrf_c"], main_cfg["depth"])
    pq_main = pq_grid[main_key]

    expl, final_w = [], 1.0
    for w in pre["exploratory"]["w_dense"]:
        if w == 1.0:
            continue
        pq, mean = evaluate(fuse(bm, dn, ids, main_cfg["rrf_c"], main_cfg["depth"], w), qrels, ids)
        cmp_ = paired(pq, pq_main, boot)
        ok, why = adopt_alternative(tuple(cmp_["hit@5"].values()), tuple(cmp_["hit@20"].values()), pre["adoption_rule"]["margin"])
        expl.append({"w_dense": w, **mean, "vs_main": cmp_, "adopt": ok, "reason": why, "_pq": pq})
    passing = [e for e in expl if e["adopt"]]
    if passing:
        win = sorted(passing, key=lambda e: (-round(e["hit@5"], 12), -round(e["recall@20"], 12), e["w_dense"]))[0]
        final_w = win["w_dense"]
        notes.append(f"nhóm thăm dò: w_dense = {final_w} qua quy tắc nhận, thay cấu hình chính")
    else:
        notes.append("nhóm thăm dò: không w_dense nào qua quy tắc nhận, giữ w_dense = 1.0")
    for e in expl:
        e.pop("_pq")

    final = {"rrf_c": main_cfg["rrf_c"], "depth": main_cfg["depth"], "w_dense": final_w, "fusion_top_k": 20}
    pq_final = pq_main if final_w == 1.0 else evaluate(fuse(bm, dn, ids, final["rrf_c"], final["depth"], final_w), qrels, ids)[0]
    final_metrics = {m: sum(pq_final[m]) / len(ids) for m in METRICS}
    final_vs = {"dense384": paired(pq_final, pq_base["dense384"], boot), "bm25": paired(pq_final, pq_base["bm25"], boot),
                "default": paired(pq_final, pq_def, boot)}
    out = {"split": "train-select", "n": len(ids), "prereg_created_at": pre["created_at"], "baselines": base, "grid": grid,
           "best_in_grid": {k: best[k] for k in ("rrf_c", "depth")}, "main_vs_default": adopted,
           "exploratory_weighted": expl, "final": final, "final_metrics": final_metrics, "final_vs": final_vs,
           "n_configs_tried": len(grid) + len(expl), "notes": notes, "seconds": round(time.perf_counter() - t0, 1)}
    write_json(RESULTS_DIR / "rrf_sweep.json", out)
    write_json(RESULTS_DIR / "13_selection.json", {"final": final, "chosen_on": "train-select", "n": len(ids),
                                                  "prereg_created_at": pre["created_at"], "notes": notes})
    write_hybrid_yaml(final)
    print("train-select n =", len(ids))
    for n, m in base.items():
        print(f"  {n:9s}", {k: round(v, 4) for k, v in m.items()})
    for g in sorted(grid, key=lambda g: -g["hit@5"]):
        print(f"  c={g['rrf_c']:3d} depth={g['depth']:3d}  hit@5 {g['hit@5']:.4f}  hit@20 {g['hit@20']:.4f}  rec@20 {g['recall@20']:.4f}  mrr@10 {g['mrr@10']:.4f}  vs dense {g['vs_dense']['hit@5']['diff']:+.4f}")
    for e in expl:
        print(f"  w_dense={e['w_dense']}  hit@5 {e['hit@5']:.4f}  vs main {e['vs_main']['hit@5']['diff']:+.4f} [{e['vs_main']['hit@5']['lo']:+.4f}; {e['vs_main']['hit@5']['hi']:+.4f}]  adopt={e['adopt']}")
    print("notes:", *notes, sep="\n  ")
    print("FINAL:", final, {k: round(v, 4) for k, v in final_metrics.items()})
    print("vs dense hit@5", final_vs["dense384"]["hit@5"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
