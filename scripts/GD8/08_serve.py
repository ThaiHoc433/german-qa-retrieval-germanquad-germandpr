"""GĐ8.9: chạy API. Một worker: mỗi tiến trình nạp riêng reader và encoder lên GPU (~1,2 GB VRAM đã đo trên máy 4 GB), nên hai worker không vừa."""
import argparse
import logging

import uvicorn

from germanqa.api.app import create_app
from germanqa.api.settings import load_settings


def main(host, port, checkpoint, retrievers, reranker, log_level):
    logging.basicConfig(level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings(checkpoint=checkpoint, retrievers=retrievers.split(",") if retrievers else None, reranker=reranker)
    uvicorn.run(create_app(settings), host=host, port=port, workers=1, log_level="warning")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint", default=None, help="mặc định: pipeline.yaml")
    ap.add_argument("--retrievers", default=None, help="ví dụ bm25 hoặc bm25,dense (mặc định: api.yaml)")
    ap.add_argument("--reranker", default=None, help="bge_m3 hoặc mmarco; chỉ dùng sau hybrid (mặc định: api.yaml, tức tắt)")
    ap.add_argument("--log-level", default="info")
    a = ap.parse_args()
    main(a.host, a.port, a.checkpoint, a.retrievers, a.reranker, a.log_level)
