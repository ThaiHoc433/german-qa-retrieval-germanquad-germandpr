"""GĐ8.9: hỏi API, in kết quả giống 05_ask.py. Dùng httpx (không dùng Invoke-RestMethod của PowerShell 5.1: nó đọc UTF-8 sai)."""
import argparse
import json

import httpx


def main(question, retriever, url, trace):
    body = {"question": question, "trace": trace}
    if retriever:
        body["retriever"] = retriever
    r = httpx.post(f"{url}/api/v1/qa", json=body, timeout=60)
    d = r.json()
    if r.status_code != 200:
        e = d["error"]
        print(f"Lỗi {r.status_code}: {e['code']} - {e['message']}  (request_id {e['request_id']})")
        return
    a, b = d["answer_start"], d["answer_end"]
    t = d["passage_text"]
    print(f"Hỏi     : {d['question']}")
    print(f"Đáp án  : {d['answer']!r}   (trạng thái: {d['status']}, retriever: {d['retriever']})")
    print(f"Đoạn    : {d['passage_id']}  [{a}:{b}]  UTF-16 [{d['answer_start_utf16']}:{d['answer_end_utf16']}]")
    if t:
        print(f"Ngữ cảnh: ...{t[max(0, a - 80):a]}[[{t[a:b]}]]{t[b:b + 80]}...")
    print(f"Điểm    : truy hồi {d['retrieval_score']:.2f}, reader {d['reader_score']:.2f}")
    tm = d["timings"]
    print(f"Thời gian: chờ {tm['queue_ms']:.1f} ms, truy hồi {tm['retrieval_ms']:.1f} ms, reader {tm['reader_ms']:.1f} ms, "
          f"tổng {tm['total_ms']:.1f} ms")
    v = d["versions"]
    print(f"Phiên bản: mô hình {v['model']}, corpus {v['corpus']}, thí nghiệm {v['experiment_id']}, request {d['request_id']}")
    if trace:
        print(json.dumps(d["trace"]["candidates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--retriever", choices=["bm25", "dense"], default=None)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    main(a.question, a.retriever, a.url, a.trace)
