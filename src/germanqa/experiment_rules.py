"""GĐ11.1: quy tắc thí nghiệm bằng MÃ, không chỉ bằng lời.

1. Sổ phơi bày test (results/test_exposure_ledger.jsonl): chỉ THÊM dòng, không sửa. Mỗi lần một script mở dữ liệu
   hay kết quả của test là một dòng: script nào, loại truy cập, để làm gì, tệp nào, lúc nào.
2. Rào chắn `require_test_access`: script chạy trên test phải được truyền cờ --confirm-test, nếu không thì dừng
   TRƯỚC khi mở bất cứ thứ gì. Đi qua rào chắn thì tự ghi sổ.
3. Mã thí nghiệm B0/B1/H0/H1 cộng băm cấu hình, và ghi tệp mốc KHÔNG ghi đè."""
import hashlib
import json
import time
from pathlib import Path

from germanqa.config import RESULTS_DIR

LEDGER_PATH = RESULTS_DIR / "test_exposure_ledger.jsonl"
CODES = ("B0", "B1", "H0", "H1")
KINDS = ("metric_run", "error_analysis", "metadata_only")


class TestAccessError(RuntimeError):
    """Gọi vào test mà không có xác nhận. (Đặt tên Test... nhưng không phải lớp test: pytest bỏ qua nhờ __test__.)"""
    __test__ = False


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_ledger(path=None):
    p = Path(path or LEDGER_PATH)
    if not p.exists():
        return []
    with open(p, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_ledger(entry, path=None):
    """Chỉ thêm dòng. Không có hàm sửa hay xóa dòng nào trong module này, và đó là chủ ý."""
    p = Path(path or LEDGER_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": _now(), "source": "live", **entry}
    if entry.get("kind") not in KINDS:
        raise ValueError(f"kind phải thuộc {KINDS}")
    for k in ("script", "purpose"):
        if not entry.get(k):
            raise ValueError(f"thiếu trường bắt buộc: {k}")
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry


def require_test_access(script, purpose, confirm, kind="metric_run", files=(), codes=(), ledger_path=None):
    """Rào chắn. Không có `confirm` thì ném TestAccessError và KHÔNG ghi sổ (chưa phơi bày gì).
    Có `confirm` thì ghi một dòng vào sổ rồi cho đi tiếp."""
    if kind == "metadata_only":
        raise ValueError("truy cập chỉ đọc siêu dữ liệu dùng log_test_metadata, không cần cờ")
    if not confirm:
        raise TestAccessError(f"{script}: chạm test cần cờ --confirm-test. Test chỉ chạy MỘT lần sau khi đã chốt cấu hình "
                              f"trên dev; mỗi lần mở đều ghi vào {LEDGER_PATH.name}.")
    bad = [c for c in codes if c not in CODES]
    if bad:
        raise ValueError(f"mã thí nghiệm lạ {bad}; hợp lệ: {CODES}")
    return append_ledger({"script": script, "purpose": purpose, "kind": kind, "files": list(files),
                          "codes": list(codes), "confirmed": True}, ledger_path)


def guard_or_exit(*args, **kwargs):
    """Dùng trong script: thiếu cờ thì thoát với thông báo, không in traceback."""
    try:
        return require_test_access(*args, **kwargs)
    except TestAccessError as exc:
        raise SystemExit(str(exc)) from None


def log_test_metadata(script, purpose, files=(), ledger_path=None):
    """Script chỉ đọc câu hỏi/nhãn của test để kiểm tra rò rỉ, không tính chỉ số chất lượng. Vẫn ghi sổ."""
    return append_ledger({"script": script, "purpose": purpose, "kind": "metadata_only", "files": list(files),
                          "codes": [], "confirmed": False}, ledger_path)


def exposure_summary(path=None):
    rows = read_ledger(path)
    by_kind = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    return {"entries": len(rows), "by_kind": by_kind,
            "metric_runs": [r for r in rows if r["kind"] == "metric_run"],
            "reconstructed": sum(1 for r in rows if r.get("source") == "reconstructed")}


def disclosure_text(path=None):
    """Đoạn công khai cho báo cáo cuối, sinh thẳng từ sổ để không viết tay sai."""
    s = exposure_summary(path)
    if not s["entries"]:
        return "Dữ liệu test chưa bị mở."
    runs = "; ".join(f"{r['ts'][:10]} {r['script']} ({r['purpose']})" for r in s["metric_runs"])
    return (f"Sổ phơi bày test ghi {s['entries']} lần truy cập ({', '.join(f'{k}: {v}' for k, v in s['by_kind'].items())}); "
            f"{s['reconstructed']} dòng được dựng lại từ thời gian sửa tệp vì xảy ra trước khi có sổ. "
            f"Các lần tính chỉ số trên test: {runs or 'chưa có'}.")


def config_sha1(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def baseline_id(code, cfg):
    """B0_3f2a91bc: mã + 8 ký tự băm của cấu hình hiệu lực. Đổi cấu hình thì đuôi đổi."""
    if code not in CODES:
        raise ValueError(f"mã phải thuộc {CODES}")
    return f"{code}_{config_sha1(cfg)[:8]}"


def write_once(path, obj):
    """Ghi tệp mốc. Đã tồn tại thì từ chối: muốn đổi mốc phải tạo mã mới, không sửa mốc cũ."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"{p.name} đã tồn tại: mốc không bị ghi đè. Đổi mốc = tạo mã mới và ghi lý do.")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    return p
