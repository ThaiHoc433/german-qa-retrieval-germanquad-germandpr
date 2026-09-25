---
buoi: 18
chang: 6
ten_chang: Run infra, Evaluation và BM25
tuan_goi_y: 6
thu_goi_y: Thứ 6 (lịch gợi ý, có thể đổi)
tasks: [T6.2, T6.3]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 150 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 6 — Hạ tầng run, Evaluation và BM25
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §3.3 Quy tắc xử lý, §3.5 Trường hợp biên, §3.6 Kiểm thử và §3.7 Artifact
  - docs/learning/00-project-analysis.md §6. Hợp đồng dữ liệu và giao thức cần giữ
  - results/metrics_bm25_base_dev.json §recall@1, recall@5, recall@10, recall@20
  - data/processed/qa_germanquad_dev.jsonl §query_id và gold_passage_ids
  - data/processed/retrieval_bm25_base_dev.jsonl §query_id, rank và passage_id
trang_thai: chưa làm
---

# Buổi 18 — Metric, mẫu số và bucket Evaluation

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- tính tay Recall@k, GoldCoverage@k, MRR@10, EM và F1 trên fixture;
- phân biệt `N_eval` với `n_latency` và không tự bỏ query lỗi khỏi mẫu số;
- phân loại năm bucket tầng 1 và giữ `final_correct` độc lập;
- kiểm tra mode/manifest trước khi tính metric và nhận ra lệch V1.1 trong code cũ.

## 2. Vị trí trong bức tranh lớn

Buổi 18 thực hiện T6.2–T6.3 sau run identity. Evaluation chỉ đọc artifact đã lưu, không chạy model; nó là giao thức chung để so BM25, Dense, Reader và Reranker ở các chặng sau. Kết luận “retrieval là bottleneck” vẫn là giả thuyết cho tới khi bucket và metric từ run hợp lệ cho thấy điều đó.

## 3. Tiền đề

- Buổi 17: manifest, fingerprint, `retrieval_k=20`, `reader_k=5`.
- Buổi 6/10: D2, D3, Recall@k và các bucket.
- Biết list rank bắt đầu từ 1 và EM/F1 lấy max qua mọi `reference_answers`.
- Có thể làm T6.1–T6.3 trên fixture khi nhánh T5 chưa xong; điều này không mở gate chạy BM25 thật T6.4.

## 4. Khái niệm cốt lõi

| Khái niệm | Fixture minh họa | Sai lầm hay gặp |
|---|---|---|
| `N_eval` | 3 QARecord, 1 có `gold_evidences=[]` → `N_eval=2`; query gold rỗng báo riêng. | Dùng tổng số dòng làm mẫu số chất lượng. |
| `n_latency` | Chỉ record `status="ok"` thuộc `N_eval` có latency hợp lệ; file predictions rỗng có `n_latency=0`. | Dùng `N_eval` cho latency. |
| Recall@k | `retrieved=[p3,p1,p2]`, `gold=[p2,p4]`: Recall@1=0, Recall@3=1. | Dùng phân số gold tìm được rồi gọi là Recall. |
| GoldCoverage@k/MRR | Cùng fixture: Coverage@3=1/2, MRR@10=1/3. | Đổi cutoff hoặc lấy rank của passage không phải gold. |
| Bucket/final_correct | Gold ngoài pool → `retrieval_miss`; EM=1 vẫn lưu `final_correct=true` nếu trả lời đúng từ passage khác. | Suy `final_correct` từ bucket. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–25 | Giải thích bằng ví dụ | N_samples, metric và bucket. |
| 25–55 | Xem input | Đọc spec §3.3–3.7 và `src/evaluation.py`. |
| 55–95 | Fixture CPU | Tính metric tay, gọi hàm, đối chiếu artifact cũ bằng lệnh có sẵn. |
| 95–125 | Kiểm tra | Manifest/mode/empty prediction và bucket matrix. |
| 125–150 | Tự giải thích + nhật ký | Nêu điều kiện chuyển T6.4. |

## 6. Input

- `src/evaluation.py` và `tests/test_evaluation.py` chỉ đọc.
- Spec §3.3–3.7: công thức, edge cases, validator và artifact.
- Fixture 3 QARecord (một gold rỗng), retrieval list `p3,p1,p2`, gold `p2,p4`, predictions có EM/F1 khác nhau và một record `status="error"`.
- Base V1.1/manifest thật chỉ được dùng sau gate T6.1 và Chặng 5.
- Ba file cũ để học cách phát hiện nhãn sai: `results/metrics_bm25_base_dev.json`, `data/processed/qa_germanquad_dev.jsonl`, `data/processed/retrieval_bm25_base_dev.jsonl`. Gold cũ dùng `gold_passage_ids`; không chuyển nó thành QARecord V1.1 chỉ bằng đổi nhãn.

## 7. Thao tác từng bước

1. Đọc hàm metric và validator:

   ```powershell
   Select-String -LiteralPath 'src/evaluation.py' -Pattern 'def recall_at_k|def gold_coverage_at_k|def reciprocal_rank|def exact_match_score|def f1_score|def classify_query|def evaluate_retrieval_results'
   ```

   Kỳ vọng: thấy các hàm lẻ và hàm tổng hợp; chưa coi code hiện tại tự động khớp toàn bộ V1.1.
2. Tính tay fixture trước khi chạy: với `retrieved=[p3,p1,p2]`, `gold=[p2,p4]`, ghi Recall@1=0, Recall@3=1, GoldCoverage@3=0,5, MRR@10=1/3. Sau đó đối chiếu bằng lệnh đã kiểm tra:

   ```powershell
   & '.venv/Scripts/python.exe' -B -c "from src.evaluation import recall_at_k,gold_coverage_at_k,reciprocal_rank,exact_match_score,f1_score; r=['p3','p1','p2']; g=['p2','p4']; print(recall_at_k(r,g,1), recall_at_k(r,g,3), gold_coverage_at_k(r,g,3), reciprocal_rank(r,g,10)); print(exact_match_score('Einstein','Albert Einstein'), f1_score('Einstein','Albert Einstein'))"
   ```

   Kỳ vọng: `0.0 1.0 0.5 0.3333333333333333` rồi `0.0 0.6666666666666666` (có thể khác cách in số thập phân nhưng không khác giá trị).
3. **Ví dụ thật từ artifact cũ:** dự đoán cột nào sẽ khớp field `recall@k` trong file, rồi tính lại cả hai công thức. Lệnh đọc gold cũ và sắp kết quả theo `rank`, giữ query thiếu retrieval trong mẫu số bằng danh sách rỗng; không chạy BM25 hay ghi artifact. Khi copy Markdown nguồn, đưa dấu `@'` và `'@` về cột 0.

   ```powershell
   @'
   import json
   from collections import defaultdict
   from math import isclose
   from pathlib import Path
   def read_rows(path):
       return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
   qa = read_rows("data/processed/qa_germanquad_dev.jsonl")
   assert len({r["query_id"] for r in qa}) == len(qa)
   gold = {r["query_id"]: set(r["gold_passage_ids"]) for r in qa if r["gold_passage_ids"]}
   ranking = defaultdict(list)
   for row in read_rows("data/processed/retrieval_bm25_base_dev.jsonl"):
       assert row["query_id"] in gold
       ranking[row["query_id"]].append(row)
   for rows in ranking.values():
       rows.sort(key=lambda r: r["rank"])
       assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
       assert len({r["passage_id"] for r in rows}) == len(rows)
   saved = json.loads(Path("results/metrics_bm25_base_dev.json").read_text(encoding="utf-8"))
   print("rows", len(qa), "nonempty_gold", len(gold), "retrieval_rows", sum(map(len, ranking.values())))
   print("k saved_recall GoldCoverage Hit match")
   for k in (1, 5, 10, 20):
       hits, coverage_sum = 0, 0.0
       for qid, ids in gold.items():
           top = {r["passage_id"] for r in ranking.get(qid, [])[:k]}
           overlap = len(top & ids)
           hits += bool(overlap)
           coverage_sum += overlap / len(ids)
       coverage, hit = coverage_sum / len(gold), hits / len(gold)
       match = isclose(saved[f"recall@{k}"], coverage, rel_tol=0, abs_tol=1e-12)
       print(k, f'{saved[f"recall@{k}"]:.4f}', f"{coverage:.4f}", f"{hit:.4f}", match)
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: `rows 1031 nonempty_gold 1031 retrieval_rows 51550`; bốn dòng `match=True` và bảng sau (làm tròn bốn chữ số):

   | k | Field `recall@k` đã lưu | GoldCoverage tính lại | Hit@k tính lại |
   |---:|---:|---:|---:|
   | 1 | 0,1896 | 0,1896 | 0,1979 |
   | 5 | 0,4481 | 0,4481 | 0,4617 |
   | 10 | 0,5708 | 0,5708 | 0,5839 |
   | 20 | 0,6639 | 0,6639 | 0,6780 |

   **Kết luận giới hạn:** cả bốn field cũ mang tên `recall@k` chứa GoldCoverage theo phép đối chiếu này; Hit@k là công thức Recall của V1.1 nhưng tính trên **artifact cũ**, chưa phải kết quả V1.1 được nghiệm thu. Không suy lỗi này còn nằm trong hàm `recall_at_k` hiện tại (bước 2 đã kiểm hàm lẻ), không sửa file cũ. [results/metrics_bm25_base_dev.json §recall@1, recall@5, recall@10, recall@20] [data/processed/qa_germanquad_dev.jsonl §gold_passage_ids] [data/processed/retrieval_bm25_base_dev.jsonl §query_id, rank và passage_id] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §3.3 Quy tắc xử lý]

   Nếu số khác: kiểm ba path/hash, số query và gold rỗng, cách sort rank, mẫu số; không đổi công thức để ép khớp.
4. Lập bảng mẫu số: 3 record tổng, 1 gold rỗng → `N_eval=2`; nếu chỉ 1 record `status="ok"` có latency thì `n_latency=1`; file predictions rỗng vẫn có `N_eval=2`, `n_latency=0` và mọi query đủ gold là `execution_failure`.
5. Đối chiếu bucket theo thứ tự: execution_failure → retrieval_miss → selection_failure → reader_failure → correct. Ghi riêng `final_correct`, `gold_in_retrieval_pool`, `gold_in_reader_input`, `first_gold_rank`.
6. Kiểm tra lệch tên bucket trong code cũ:

   ```powershell
   Select-String -LiteralPath 'src/evaluation.py' -Pattern 'BUCKETS|reranking_failure|selection_failure'
   ```

   Kỳ vọng: code hiện tại còn chuỗi `reranking_failure`; spec V1.1 yêu cầu `selection_failure`. Ghi đây là việc T6.3 cần sửa/test, không đổi spec.
7. Đọc CLI mà không ghi artifact:

   ```powershell
   & '.venv/Scripts/python.exe' -B -m src.evaluation --help
   ```

   Kỳ vọng: có các mode/đường input; lệnh có `--out` sẽ ghi file nên không chạy trong giờ học liệu.
8. Viết bảng validator: query_id không trùng, prediction query thuộc gold, passage_id thuộc corpus manifest, cùng run/config/corpus/k, manifest fingerprint khớp; vi phạm thì dừng, không tính số.
9. **Nhánh thực thi sau gate (không tính vào 150 phút):** cập nhật test/logic T6.2–T6.3 trong `src/evaluation.py` và `tests/` theo spec; test metric lẻ, empty prediction, năm bucket và manifest mismatch. Có thể làm trên fixture trong khi T5 chưa xong. T6.4 chạy thật phải chờ cả T6.1–T6.3 và base T5.8 pass; không dùng score để sửa giao thức.

## 8. Dự đoán trước, kiểm tra sau

1. Query gold rỗng có vào `N_eval` không? **Không; báo riêng `n_excluded_empty_gold`.**
2. Gold ở rank 8 với `reader_k=5`, `retrieval_k=20` thuộc bucket nào nếu có trong pool? **`selection_failure`.**
3. File predictions rỗng có tính được không? **Có nếu manifest hợp lệ; mọi query đủ gold là `execution_failure`, `n_latency=0`.**
4. `GoldCoverage@k=0,5` có được gọi là Recall@k không? **Không; Recall@k là chỉ báo hit nhị phân.**

## 9. Output cần có

- Bảng tính tay metric và mẫu số `N_eval`/`n_latency`.
- Bảng bốn hàng đối chiếu nhãn `recall@k` cũ với GoldCoverage/Hit và ghi rõ chưa phải benchmark V1.1.
- Bucket matrix gồm execution, retrieval, selection, reader, correct; cờ `final_correct` tách riêng.
- Bảng validation manifest và ghi nhận lệch `reranking_failure` của code cũ.
- Sau nhánh thực thi: test/logic T6.2–T6.3 và log; chưa có thì ghi chưa làm.

## 10. Kiểm tra hiểu bài

1. Vì sao gold rỗng không vào mẫu số? **Không có evidence để đánh giá retrieval/QA theo D2.** Chấm: nêu D2 và số báo riêng.
2. Vì sao Coverage khác Recall? **Coverage đếm tỉ lệ gold giao top-k; Recall chỉ hỏi có ít nhất một gold.** Chấm: dùng đúng tên.
3. `status="error"` có bị xóa khỏi N_eval không? **Không; thuộc mẫu số chất lượng và vào execution_failure.** Chấm: không âm thầm loại.
4. Reranker loại gold ở rank 8 có phải retrieval_miss không? **Không; gold đã ở retrieval pool, nên là selection_failure.** Chấm: phân biệt pool và reader input.
5. Vì sao manifest sai corpus phải dừng? **Passage ID/metric có thể thuộc tier khác, không còn truy vết hợp lệ.** Chấm: nêu fingerprint/corpus_id.
6. File cũ ghi `recall@20=0,6639` đã đủ bằng chứng Recall V1.1 chưa? **Chưa: nó khớp GoldCoverage; Hit tính lại là 0,6780 trên bộ cũ.** Chấm: 1 điểm phân biệt công thức, 1 điểm nêu giới hạn artifact cũ.

## 11. Bài tự giải thích

Giải thích bằng sơ đồ cách một query đi từ gold → retrieval pool → reader input → bucket, đồng thời chỉ ra `N_eval`, `n_latency` và `final_correct` nằm ở đâu.

Rubric 4 điểm: đúng mẫu số; đúng thứ tự bucket; tách final_correct; có điều kiện manifest.

## 12. Gate của buổi

Qua giờ hướng dẫn khi tính đúng fixture metric/mẫu số, phân loại đúng năm bucket và giải thích được nhãn sai của artifact cũ. Chỉ sau nhánh thực thi T6.1–T6.3 có test pass, gồm manifest mismatch/file predictions rỗng, và base V1.1 T5.8 đạt gate mới chạy T6.4 thật.

## 13. Lỗi thường gặp & cách gỡ

- **Metric:** gọi GoldCoverage là Recall → dùng đúng định nghĩa §3.3.
- **Mẫu số:** bỏ query error → giữ trong `N_eval`, gán execution_failure.
- **Bucket:** dùng tên `reranking_failure` của code cũ → ghi lệch và sửa theo `selection_failure` trong task.
- **Manifest:** suy config từ dòng đầu → đọc entry theo `run_id`.
- **Scope:** chạy CLI có `--out` → chỉ dùng `--help`/fixture trong giờ học liệu.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T6.2, T6.3
Lệnh chỉ-đọc đã chạy:
Output path:
Fixture metric/mẫu số:
Bảng nhãn metric cũ ↔ công thức tính lại; ba input path:
Bucket matrix:
Manifest validation:
Nhánh thực thi đã chạy/chưa chạy:
Hiểu/chưa hiểu:
Lỗi:
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc §4 BM25, tokenizer và runner; kiểm tra base V1.1 đã qua gate trước khi cân nhắc retrieval-only thật.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: signature manifest validator/CLI và nơi đặt test mới.
- CẦN XÁC MINH: bản sửa code cũ từ `reranking_failure` sang `selection_failure` theo frozen spec.
- CẦN XÁC MINH: base V1.1 và corpus manifest revision trước T6.4.
