"""Checklist dense (docs/dense_checklist.md) chỉ có giá trị khi mọi bằng chứng nó nêu còn tồn tại: kiểm đường dẫn và tên test."""
import ast
import re

import pytest

from germanqa.config import ROOT

DOC = ROOT / "docs" / "dense_checklist.md"
REF = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:py|json|md|yaml))(?:::([A-Za-z0-9_]+))?`")


def _refs():
    return sorted(set(REF.findall(DOC.read_text(encoding="utf-8"))))


def test_checklist_co_du_muc_va_bang_chung():
    text = DOC.read_text(encoding="utf-8")
    rows = [l for l in text.splitlines() if re.match(r"\| \d+ \|", l)]
    assert len(rows) == 12
    assert all(re.search(r"`[^`]+\.(py|json)", r) for r in rows), "mỗi mục phải dẫn bằng chứng là tệp thật"


@pytest.mark.parametrize("path,fn", _refs())
def test_bang_chung_ton_tai(path, fn):
    p = ROOT / path
    assert p.exists(), f"{path} không tồn tại"
    if fn:
        names = {n.name for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))) if isinstance(n, ast.FunctionDef)}
        assert fn in names, f"{path} không có hàm {fn}"
