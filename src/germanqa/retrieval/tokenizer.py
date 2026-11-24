
"""Tách từ cho BM25 (GĐ3). Đây là tokenizer của BM25, khác tokenizer của reader (GĐ4)."""
import re

# \w trong chuỗi Unicode khớp chữ cái có dấu: ä ö ü ß é... nên không làm mất umlaut
_TOKEN = re.compile(r"\w+", re.UNICODE)

# Danh sách stopword nhỏ, dùng cho biến thể "use_stopwords" (đo riêng trên dev)
GERMAN_STOPWORDS = frozenset("""
der die das den dem des ein eine einen einem einer eines und oder aber
ist sind war waren wird werden wurde wurden hat haben hatte hatten
im in an auf aus bei mit nach von vor zu zum zur über unter für durch
als wie wo wer was wann warum welche welcher welches
nicht auch sich es er sie wir ihr ich du man dass
""".split())


def tokenize(text, stopwords=None):
    """Chữ thường, tách theo chữ-số; bỏ stopword nếu được yêu cầu."""
    tokens = _TOKEN.findall(text.lower())
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens