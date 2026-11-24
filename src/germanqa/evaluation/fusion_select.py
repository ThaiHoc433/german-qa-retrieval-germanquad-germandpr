"""Ghi trước và chọn cấu hình RRF (GĐ13.0, 13.3).

Quy tắc chọn được viết thành hàm để không bị diễn giải lại sau khi thấy kết quả; file ghi trước
(results/13_prereg.json) khóa SHA-1 của dữ liệu đầu vào và của mã quy tắc."""
from pathlib import Path

from germanqa.data.io_utils import file_sha1, read_json


class PreregError(RuntimeError):
    pass


def inputs_sha1(base_dir, relative_paths):
    base = Path(base_dir)
    return {p: file_sha1(base / p) for p in relative_paths}


def require_prereg(prereg_path, base_dir):
    """Đọc file ghi trước và kiểm tra mọi SHA-1 đầu vào/mã còn khớp. Thiếu file hoặc lệch -> PreregError."""
    prereg_path = Path(prereg_path)
    if not prereg_path.exists():
        raise PreregError(f"chưa có {prereg_path.name}: phải ghi trước (13_prereg.py) rồi mới được quét")
    pre = read_json(prereg_path)
    bad = []
    for rel, want in {**pre["inputs"], **pre["code"]}.items():
        p = Path(base_dir) / rel
        got = file_sha1(p) if p.exists() else None
        if got != want:
            bad.append(f"{rel}: ghi trước {want}, hiện tại {got}")
    if bad:
        raise PreregError("đầu vào/mã đã đổi sau khi ghi trước: " + "; ".join(bad))
    return pre


def _key(cfg):
    return (cfg["rrf_c"], cfg["depth"])


def choose_config(results, default, eps=1e-12):
    """results: [{'rrf_c', 'depth', 'hit@5', 'recall@20', 'mrr@10'}]. Chọn Hit@5 cao nhất; hòa thì Recall@20, rồi MRR@10,
    rồi cấu hình giống mặc định nhất (chính nó, rồi độ sâu gần, rồi c gần). Trả về phần tử được chọn."""
    if not results:
        raise ValueError("không có cấu hình nào")
    r = lambda x: round(x / eps) if eps else x
    def order(row):
        near = (_key(row) != _key(default), abs(row["depth"] - default["depth"]), abs(row["rrf_c"] - default["rrf_c"]))
        return (-r(row["hit@5"] * 1.0), -r(row["recall@20"]), -r(row["mrr@10"])) + near
    return sorted(results, key=order)[0]
