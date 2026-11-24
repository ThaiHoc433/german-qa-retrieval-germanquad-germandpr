"""Liệt kê kết quả thí nghiệm đã lưu (GĐ8). CHỈ ĐỌC results/*.json: API không chạy và không ghi gì.

Mỗi file chỉ được rút ra các trường nêu tên dưới đây (danh sách cho phép), không bao giờ trả nguyên file, vì file có
thể chứa đường dẫn máy ('checkpoint' trong results/reader_*.json là đường dẫn tuyệt đối)."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ID_RE = re.compile(r"^(?P<kind>.+?)_(?P<stamp>\d{8}T\d{6}Z)_(?P<digest>[0-9a-f]{8})$")
RETRIEVAL_KEYS = ("hit@1", "hit@5", "hit@20", "mrr@10")
MATRIX_CODES = ("B0", "B1", "H0", "H1")
MATRIX_DESC = {"B0": "BM25 → reader", "B1": "Dense → reader", "H0": "BM25 + dense → RRF → reader", "H1": "H0 + cross-encoder → reader"}
HYBRID_CAVEAT = ("Hybrid và dense dùng encoder đã học GermanDPR train; trên toàn dev, chúng có thể được lợi từ đoạn mà encoder đã thấy "
                 "(nhóm 'đã thấy đoạn': Hit@5 của H0 0,939 so với 0,857 ở nhóm chưa thấy). Đọc cột dev_unseen khi so sánh.")
MATRIX_CAVEATS = [
    "B0, B1, H0 đo trên toàn dev (1.176 câu) và dev_unseen (234 câu). H1 chỉ đo trên dev_unseen (234 câu): so H1 với H0 ở cột dev_unseen.",
    "Toàn dev của dense và hybrid có thể cao hơn thực tế vì encoder dense đã học GermanDPR train. Số đáng tin hơn để so các cấu hình là dev_unseen.",
    "EM/F1 ở đây là đáp án tham chiếu đầu tiên (first_ref). Dev chỉ có một đáp án mỗi câu nên bằng all_refs; test có nhiều đáp án nên hai loại khác nhau.",
    "Khoảng trong ngoặc vuông là khoảng tin cậy bootstrap 95% của EM. Trang này chỉ đọc số đã lưu, không xếp hạng 'tốt nhất'.",
]
DENSE_DEV_CAVEAT = ("Tập dev bị nhiễm với encoder dense: ~80% câu hỏi dev nằm trong GermanDPR train mà encoder đã học. "
                    "Đọc cột dev_unseen khi so sánh dense với BM25.")


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _pick(d, keys):
    if not isinstance(d, dict):
        return {}
    return {k: v for k in keys if (v := _num(d.get(k))) is not None}


def _layer(block, ref="first_ref"):
    """em/f1 của một tầng ({'all_refs': {...}, 'first_ref': {...}})."""
    cell = block.get(ref) if isinstance(block, dict) else None
    return {f"{k}_{ref}" if ref != "first_ref" else k: v for k in ("em", "f1") if (v := _num((cell or {}).get(k))) is not None}


def _ci(block, ref="first_ref"):
    """Khoảng tin cậy bootstrap {'em': [lo, hi], 'f1': [lo, hi]} của một tầng, nếu file có lưu."""
    cell = block.get(ref) if isinstance(block, dict) else None
    out = {}
    for k in ("em", "f1"):
        v = (cell or {}).get(f"{k}_ci")
        if isinstance(v, list) and len(v) == 2 and all(_num(x) is not None for x in v):
            out[k] = [float(v[0]), float(v[1])]
    return out


def _intervals(kind, config, m):
    """{tập: {em|f1: [lo, hi]}} cho những file có lưu khoảng tin cậy (eval_*); loại khác -> rỗng."""
    if not isinstance(m, dict):
        return {}
    if kind == "eval_dev_dense":
        out = {s: _ci((m.get(s) or {}).get("dense")) for s in ("dev", "dev_unseen") if isinstance(m.get(s), dict)}
    elif kind == "eval_dev_hybrid":
        out = {s: _ci((m.get(s) or {}).get("hybrid")) for s in ("dev", "dev_unseen") if isinstance(m.get(s), dict)}
    elif kind.startswith("eval_"):
        out = {(config or {}).get("split") or kind.split("_", 1)[1]: _ci(m.get("end_to_end"))}
    else:
        return {}
    return {s: v for s, v in out.items() if v}


def _basename(s):
    return re.split(r"[\\/]", s)[-1] if isinstance(s, str) and s else None


def _both_refs(block):
    """em/f1 theo first_ref; thêm *_all_refs CHỈ khi khác first_ref (test có nhiều đáp án tham chiếu, dev thì bằng nhau)."""
    first, allr = _layer(block), _layer(block, "all_refs")
    extra = {k: v for k, v in allr.items() if k.replace("_all_refs", "") in first and abs(v - first[k.replace("_all_refs", "")]) > 1e-12}
    return {**first, **extra}


def _end_to_end(m, retrieval):
    out = {**_both_refs(m.get("end_to_end")), **_pick(retrieval, RETRIEVAL_KEYS)}
    total = ((m.get("timing") or {}).get("total_time") or {}).get("p50_ms")
    if _num(total) is not None:
        out["total_p50_ms"] = float(total)
    return out


def _metrics(kind, config, m):
    """{tập: {tên: số}} theo hình dạng từng loại file. Loại lạ -> chỉ lấy số đo phẳng nếu có."""
    if not isinstance(m, dict):
        return {}
    if kind == "eval_dev_dense":
        out = {}
        for split in ("dev", "dev_unseen"):
            blk = m.get(split)
            if isinstance(blk, dict):
                d = {**_layer((blk.get("dense") or {})), **_pick(blk.get("retrieval_dense"), RETRIEVAL_KEYS)}
                b = {**_layer((blk.get("bm25") or {})), **_pick(blk.get("retrieval_bm25"), RETRIEVAL_KEYS)}
                out[split] = {**d, **{f"bm25_{k}": v for k, v in b.items()}}
        return out
    if kind == "eval_dev_hybrid":
        out = {}
        for split in ("dev", "dev_unseen"):
            blk = m.get(split)
            if isinstance(blk, dict):
                h = {**_layer(blk.get("hybrid") or {}), **_pick(blk.get("retrieval_hybrid"), RETRIEVAL_KEYS)}
                b = {**_layer(blk.get("B0_bm25") or {}), **_pick(blk.get("retrieval_B0_bm25"), RETRIEVAL_KEYS)}
                d = {**_layer(blk.get("B1_dense384") or {}), **_pick(blk.get("retrieval_B1_dense384"), RETRIEVAL_KEYS)}
                out[split] = {**h, **{f"bm25_{k}": v for k, v in b.items()}, **{f"dense_{k}": v for k, v in d.items()}}
        total = ((m.get("timing") or {}).get("total_time") or {}).get("p50_ms")
        if _num(total) is not None and "dev" in out:
            out["dev"]["total_p50_ms"] = float(total)
        return out
    if kind.startswith("eval_"):
        split = (config or {}).get("split") or kind.split("_", 1)[1]
        return {split: _end_to_end(m, m.get("retrieval"))}
    if kind == "bm25":
        return {s: _pick((m.get(s) or {}).get("metrics"), RETRIEVAL_KEYS) for s in ("dev", "test") if s in m}
    if kind == "pipeline":
        return {(config or {}).get("split") or "dev": _end_to_end(m, m.get("retrieval"))}
    if kind == "reader":
        hist = m.get("history")
        last = hist[-1] if isinstance(hist, list) and hist and isinstance(hist[-1], dict) else {}
        return {"dev_last_epoch": _pick(last.get("dev"), ("em", "f1"))}      # EM/F1 trên dev sau epoch cuối của lần huấn luyện
    flat = {k: float(v) for k, v in m.items() if _num(v) is not None}
    return {"all": flat} if flat else {}


def summarize_file(path):
    """Một file results/*.json -> một mục, hoặc None nếu hỏng / không đúng hình dạng."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not isinstance(d.get("experiment_id"), str):
        return None
    match = ID_RE.match(d["experiment_id"])
    kind = match["kind"] if match else "unknown"
    config = d.get("config") if isinstance(d.get("config"), dict) else {}
    # Mốc trong mã thí nghiệm luôn là UTC và cùng định dạng; 'finished_at' (nếu có) là giờ địa phương nên không dùng để sắp xếp.
    created = d.get("finished_at")
    if match:
        try:
            created = datetime.strptime(match["stamp"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
        except ValueError:          # đúng dạng nhưng không phải ngày hợp lệ: không được làm sập cả danh sách
            pass
    metrics = _metrics(kind, config, d.get("metrics"))
    retriever = config.get("retriever") or ("dense" if "dense" in kind else "bm25" if kind.startswith(("bm25", "eval", "pipeline")) else None)
    split = config.get("split") or (kind.split("_")[1] if kind.startswith("eval_") else None)
    return {"experiment_id": d["experiment_id"], "kind": kind, "created_at": created, "split": split,
            "retriever": retriever, "checkpoint": _basename(config.get("checkpoint") or d.get("checkpoint")),
            "corpus_version": config.get("corpus_version") if isinstance(config.get("corpus_version"), str) else None,
            "metrics": metrics, "intervals": _intervals(kind, config, d.get("metrics")), "caveats": [DENSE_DEV_CAVEAT] if kind == "eval_dev_dense" else [HYBRID_CAVEAT] if kind == "eval_dev_hybrid" else []}


def _read_json(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _manifest_for(baselines_dir, code):
    """Manifest của một mã: ưu tiên bản <mã>.v2.json (đầy đủ hơn) rồi mới tới <mã>.json."""
    for name in (f"{code}.v2.json", f"{code}.json"):
        d = _read_json(Path(baselines_dir) / name)
        if d is not None and d.get("code") == code:
            return d
    return None


def _reference_codes(baselines_dir):
    """{experiment_id: mã} cho các lần chạy được manifest chọn làm số đo tham chiếu (reference_dev_run.file)."""
    out = {}
    if not baselines_dir or not Path(baselines_dir).is_dir():
        return out
    for code in MATRIX_CODES:
        man = _manifest_for(baselines_dir, code)
        ref = ((man or {}).get("reference_dev_run") or {}).get("file")
        if isinstance(ref, str) and ref:
            out[Path(ref).stem] = code
    return out


def _mark_duplicates(items):
    """Các lần chạy cùng loại và cùng đuôi băm cấu hình: giữ một đại diện (lần có mã, không thì mới nhất; items đã sắp mới nhất trước)."""
    groups = {}
    for it in items:
        m = ID_RE.match(it["experiment_id"])
        if m:
            groups.setdefault((it["kind"], m["digest"]), []).append(it)
    for g in groups.values():
        if len(g) > 1:
            keep = next((x for x in g if x.get("code")), g[0])
            for x in g:
                if x is not keep:
                    x["duplicate_of"] = keep["experiment_id"]


def list_experiments(results_dir, kind=None, baselines_dir=None):
    """(các mục mới nhất trước, số file bị bỏ qua). Có baselines_dir thì gắn mã B0/B1/H0/H1 và đánh dấu lần chạy trùng cấu hình."""
    items, skipped = [], 0
    codes = _reference_codes(baselines_dir)
    for p in sorted(Path(results_dir).glob("*.json")) if Path(results_dir).is_dir() else []:
        item = summarize_file(p)
        if item is None:
            skipped += 1
        else:
            item["code"] = codes.get(item["experiment_id"])
            item["duplicate_of"] = None
            items.append(item)
    items.sort(key=lambda it: str(it["created_at"] or ""), reverse=True)
    _mark_duplicates(items)
    return [it for it in items if kind is None or it["kind"] == kind], skipped


_MATRIX_SOURCE = {"B0": ("B0_bm25", "retrieval_B0_bm25"), "B1": ("B1_dense384", "retrieval_B1_dense384"), "H0": ("hybrid", "retrieval_hybrid")}


def _split_numbers(layer_block, retrieval_block):
    return {**_layer(layer_block or {}), **_pick(retrieval_block, RETRIEVAL_KEYS)}


def _row(code, man, results_dir, h0_metrics):
    ref = man.get("reference_dev_run") if isinstance(man.get("reference_dev_run"), dict) else {}
    fname = Path(ref["file"]).name if isinstance(ref.get("file"), str) else None
    splits, intervals, queries = {}, {}, {}
    if code in _MATRIX_SOURCE and h0_metrics is not None:                 # cả dev và dev_unseen từ tệp so sánh GĐ13
        layer_key, retr_key = _MATRIX_SOURCE[code]
        for split in ("dev", "dev_unseen"):
            blk = h0_metrics.get(split)
            if isinstance(blk, dict) and isinstance(blk.get(layer_key), dict):
                splits[split] = _split_numbers(blk[layer_key], blk.get(retr_key))
                ci = _ci(blk[layer_key])
                if ci:
                    intervals[split] = ci
                q = ((blk[layer_key].get("first_ref") or {}).get("queries"))
                if isinstance(q, int):
                    queries[split] = q
    elif code == "H1" and fname:                                            # H1 chỉ có dev_unseen (234 câu)
        res = _read_json(Path(results_dir) / fname) or {}
        h1 = res.get("H1") if isinstance(res.get("H1"), dict) else None
        if h1:
            splits["dev_unseen"] = {**_layer(h1), **_pick(ref.get("retrieval"), ("hit@5", "hit@20"))}
            ci = _ci(h1)
            if ci:
                intervals["dev_unseen"] = ci
            q = (h1.get("first_ref") or {}).get("queries")
            if isinstance(q, int):
                queries["dev_unseen"] = q
    if not splits:                                                           # dự phòng: số tham chiếu nằm ngay trong manifest
        e = ref.get("end_to_end_all_refs") or ref.get("end_to_end_first_ref") or {}
        sp = str(ref.get("split") or "dev").replace("-", "_")
        sp = sp if sp in ("dev", "dev_unseen", "test") else "dev"
        nums = {**{k: float(e[k]) for k in ("em", "f1") if _num(e.get(k)) is not None}, **_pick(ref.get("retrieval"), RETRIEVAL_KEYS)}
        if nums:
            splits[sp] = nums
            if isinstance(e.get("queries"), int):
                queries[sp] = e["queries"]
    status = man.get("status") if isinstance(man.get("status"), str) else None
    return {"code": code, "description": MATRIX_DESC[code], "config_id": man.get("config_id") if isinstance(man.get("config_id"), str) else None,
            "frozen_on": man.get("frozen_on") if isinstance(man.get("frozen_on"), str) else None,
            "status": status[:300] if status else None, "reference_file": fname, "splits": splits, "intervals": intervals, "queries": queries}


def list_matrix(baselines_dir, results_dir):
    """Hàng ma trận B0/B1/H0/H1 đọc từ baselines/*.json (đã SHA-1 xác minh ở GĐ11) và tệp kết quả mà chúng tham chiếu. Chỉ đọc; thiếu thì bỏ hàng."""
    if not baselines_dir or not Path(baselines_dir).is_dir():
        return []
    h0 = _manifest_for(baselines_dir, "H0")
    h0_ref = ((h0 or {}).get("reference_dev_run") or {}).get("file")
    h0_res = _read_json(Path(results_dir) / Path(h0_ref).name) if isinstance(h0_ref, str) else None
    h0_metrics = h0_res.get("metrics") if isinstance((h0_res or {}).get("metrics"), dict) else None
    rows = []
    for code in MATRIX_CODES:
        man = _manifest_for(baselines_dir, code)
        if man is not None:
            row = _row(code, man, results_dir, h0_metrics)
            if row["splits"]:
                rows.append(row)
    return rows
