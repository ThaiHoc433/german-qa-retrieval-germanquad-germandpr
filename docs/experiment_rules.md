# Quy tắc thí nghiệm (GĐ11)

Áp dụng cho mọi thí nghiệm từ GĐ12 trở đi. Phần lớn quy tắc được thực thi bằng mã (`src/germanqa/experiment_rules.py`), phần còn lại là cam kết của người làm.

## 1. Năm cam kết về dữ liệu test

1. **Đóng băng cấu hình trên dev trước.** Mọi lựa chọn (tham số BM25, `max_answer_tokens`, độ dài đoạn dense, `rrf_c`, độ sâu, ...) chốt bằng dev hoặc tập chọn, không bằng test.
2. **Test chạy một lần cho cả ma trận B0/B1/H0/H1**, sau khi mọi cấu hình đã đóng băng, với cờ `--confirm-test`.
3. **Không thiết kế từ lỗi trên test.** Không đọc `reports/error_analysis_test.json` hay `reports/pipeline_test_predictions.jsonl` để quyết định thiết kế; lý do cho hybrid hay rerank chỉ dựa trên số dev.
4. **Sổ phơi bày test** (`results/test_exposure_ledger.jsonl`): mỗi lần script mở dữ liệu hoặc kết quả test là một dòng, chỉ thêm, không sửa.
5. **Công khai trong báo cáo cuối**: đoạn sinh bằng `experiment_rules.disclosure_text()` từ chính sổ này.

## 2. Rào chắn bằng mã

| Thành phần | Việc làm |
|---|---|
| `require_test_access` / `guard_or_exit` | Không có `--confirm-test` thì dừng trước khi mở gì; có thì ghi sổ rồi cho đi |
| `log_test_metadata` | Dành cho script chỉ đọc siêu dữ liệu (kiểm rò rỉ, băm toàn vẹn); không cần cờ nhưng vẫn ghi sổ |
| Test quét tĩnh | `tests/GD11/test_exposure_rules.py` thất bại nếu có script nhắc tới tệp test mà không qua rào chắn |

Các script đã nối rào chắn: `03_bm25_eval.py --test`, `04d_eval_reader.py --test`, `07a_run_split.py --split test`, `07b_error_analysis.py --split test`, `06a_leakage_check.py`, `11d_manifests.py`.

**Giới hạn đã biết.** Rào chắn chỉ thấy script trong `scripts/`. Notebook, lệnh Python gõ tay và đọc trực tiếp tệp test không đi qua nó. Sổ ghi việc *tạo* kết quả test, không ghi được việc *đọc* tệp kết quả; điều đó vẫn dựa vào cam kết số 3.

## 3. Mã thí nghiệm và mốc

* **B0** BM25 + reader. **B1** dense + reader. **H0** hybrid RRF + reader. **H1** hybrid + reranker (GĐ14).
* Mốc ở `baselines/<mã>.json`, mã cấu hình dạng `B0_8037d4f8` = mã + 8 ký tự SHA-1 của cấu hình hiệu lực.
* **Mốc không bị ghi đè** (`write_once`). Muốn đổi mốc phải tạo mã mới và ghi lý do vào manifest mới, không sửa tệp cũ.
* Mỗi manifest ghi SHA-1 của checkpoint, corpus, qrels, mọi split, chỉ mục BM25 (kèm dấu vân tay), artifact dense; cấu hình hiệu lực kể cả `max_answer_tokens = 80`; phiên bản thư viện; seed; tên GPU, torch/CUDA, precision.
* `verify_manifest` tính lại SHA-1 và báo chỗ lệch; test chạy nó với B0, B1, H0.

## 4. Không ghi đè kết quả

Mỗi lần chạy sinh `experiment_id` mới (`thẻ_thời-điểm_băm-cấu-hình`). Kết quả cũ giữ nguyên. Script ghi mốc hoặc xác nhận (`13_prereg`, `13_confirmation`, `11e_B0_rerun_*`) từ chối chạy lại nếu tệp đã có.

## 5. Tập chọn

`results/select_question_ids.json`: 1.989 câu train và 234 câu dev mà encoder dense chưa thấy (tổng 2.223). Chỉ số truy hồi dùng cả hai; EM/F1 chỉ dùng 234 câu dev. Không được đưa các ID này vào huấn luyện ở GĐ16.

## 6. BM25 lưu sẵn

`artifacts/bm25/` giữ `index.npz` (không pickle) và `meta.json` (dấu vân tay). Nạp sai dấu vân tay thì dựng lại (mặc định) hoặc báo lỗi (`strict=True`, dùng cho dịch vụ và Docker). Đổi corpus, `k1`, `b`, `use_stopwords`, danh sách stopword, tokenizer hoặc công thức điểm đều làm dấu vân tay đổi.

## 7. Một evaluator cho mọi mã

Mọi con số EM/F1 của B0, B1, H0, H1 đều đi qua `germanqa.reader.qa_metrics` (`normalize`, `exact_match`, `f1_score`, `score_answers`, `evaluate_answers`). Quy tắc chuẩn hóa: chữ thường, bỏ dấu câu kể cả ngoặc kép kiểu Đức (`„“”‚‘’»«–—…`), gộp khoảng trắng. **Không** bỏ mạo từ (`der`, `die`, `das`), **không** gộp `ß` với `ss`, **không** bỏ dấu. Nhiều đáp án tham chiếu: lấy điểm cao nhất. Thiếu dự đoán: tính như chuỗi rỗng.

`tests/GD11/test_shared_evaluator.py` chặn ba cách làm sai:

| Lớp | Chặn gì |
|---|---|
| Tái lập | chấm lại dự đoán đã lưu của B0, B1, H0 (1.176 câu) phải ra đúng EM/F1 trong tệp kết quả, sai số 1e-12; đổi evaluator hay quy tắc chuẩn hóa là vỡ |
| Không có bản sao | không file nào trong `src/` hoặc `scripts/` được định nghĩa lại các hàm trên, hoặc import chúng từ nơi khác `qa_metrics` |
| Quy tắc chuẩn hóa | 9 ca chuẩn hóa biết trước, trường hợp rỗng, nhiều tham chiếu |

Hai hàm `normalize_text` (`data/text_utils.py`) và `normalize_question` (`evaluation/leakage.py`) là việc khác (làm sạch văn bản, so khớp câu hỏi) và không liên quan tới chấm điểm.

## 8. Giả thuyết ghi trước cho GĐ15

`results/15_hypotheses.json` (ghi bằng `scripts/GD15/15_hypotheses.py`, một lần) ghi ba giả thuyết **trước** khi chạy test cho B1, H0, H1, cùng hàm phán định trong `src/germanqa/evaluation/gd15_hypotheses.py`:

* **HA.** Chênh EM so với BM25 (B1−B0, H0−B0) trên test gần chênh trên dev-unseen hơn chênh trên toàn dev. Lý do: trên dev, dense và hybrid được lợi từ đoạn mà encoder đã học; test không có đoạn đúng nào như vậy. Dùng B0 làm đối chứng độ khó của tập.
* **HB.** Trên test, H0 không kém B1 về EM (chênh ghép cặp ≥ 0). Trên toàn dev H0 kém B1 0,010; trên dev-unseen H0 hơn B1 0,013.
* **HC.** Trên test, H1 hơn H0 về Hit@5 (cận dưới > 0) nhưng chênh EM có khoảng tin cậy chứa 0.

Script từ chối chạy nếu sổ phơi bày đã có lần tính chỉ số test cho B1, H0 hoặc H1. Kết quả test chỉ dùng để phán định và báo cáo, không dùng để chọn lại cấu hình, và phải báo cả khi giả thuyết bị bác bỏ.

## 9. Trang Số đo: mã ma trận và lần chạy trùng

* **Mã B0/B1/H0/H1 không nằm trong `experiment_id`.** Định dạng `thẻ_thời-điểm_băm` giữ nguyên để không đổi mã của các tệp đã lưu. Mã được gán qua manifest: lần chạy mà `reference_dev_run.file` của `baselines/<mã>.json` (hoặc `<mã>.v2.json`, ưu tiên hơn) trỏ tới thì mang mã đó (`code` trong `/api/v1/experiments`).
* **Bảng ma trận** (`GET /api/v1/matrix`) đọc thẳng `baselines/*.json` và tệp kết quả chúng tham chiếu; thiếu tệp thì bỏ hàng, không lỗi. Chỉ lấy *tên* tệp rồi tìm trong thư mục kết quả, nên manifest không thể trỏ API ra ngoài thư mục đó.
* B0, B1, H0 có cả `dev` (1.176 câu) và `dev_unseen` (234 câu); **H1 chỉ có `dev_unseen`**. Chỉ so các hàng trong cùng một bảng.
* **Lần chạy trùng cấu hình** (cùng loại và cùng đuôi băm, ví dụ hai tệp `eval_dev_hybrid_…_598ddf6f`) được gộp: giữ lần có mã (không có thì lần mới nhất); các lần còn lại có `duplicate_of` và nằm trong mục gập trên trang. Tệp không bị xóa hay đổi.
* EM/F1 `first_ref` là số chính. `em_all_refs`/`f1_all_refs` chỉ hiện khi khác `first_ref` (test có nhiều đáp án, dev thì bằng nhau). Giả thuyết GĐ15 phán định theo `first_ref` (xem `results/15_hypotheses_addendum.json`).
* Trang chỉ đọc; không xếp hạng "tốt nhất".

## 10. Chi phí VRAM của H1 (GĐ14.5, mô phỏng)

`scripts/GD14/14l_vram_real.py` → `results/14l_vram_real.json`. **Mô phỏng**, không phải đo trên máy bị chương trình khác chiếm thật: script chiếm sẵn phần VRAM bằng mức "chương trình khác" (1.377 MB đo lúc GĐ8, và 1.700 MB là cận trên nêu ra), rồi dựng pipeline hybrid + reader + reranker trong một tiến trình riêng và trả lời 30 câu dev-unseen. Cột "còn" là phần card 4.096 MB chưa bị dùng ở đỉnh.

| Chương trình khác | Biến thể | Đỉnh cả card (MB) | Còn (MB) | Xếp lại (ms) | Tổng/câu (ms) |
|---|---|---|---|---|---|
| 1.377 | H0, không reranker | 2.711 | 1.385 | 0 | 138 |
| 1.377 | bge-m3, batch 16 | 3.844 | **252** | 567 | 734 |
| 1.377 | bge-m3, batch 4 | 3.846 | 250 | 532 | 671 |
| 1.377 | bge-m3, nạp tuần tự | 2.730 | 1.366 | 4.457 | **4.643** |
| 1.377 | mmarco, batch 16 | 2.895 | 1.201 | 99 | 227 |
| 1.700 | bge-m3, batch 16 | 3.896 | **200** | 974 | 1.089 |
| 1.700 | mmarco, batch 16 | 3.217 | 879 | 94 | 224 |

* Cả 12 lần chạy đều **vừa** trong mô phỏng. bge-m3 vừa nhưng chỉ còn khoảng 200–250 MB; giảm batch không giải phóng đáng kể bộ nhớ (3.844 → 3.846 MB).
* Nạp tuần tự (đưa reranker lên GPU chỉ lúc xếp lại) giải phóng gần 1,1 GB nhưng làm mỗi câu chậm khoảng 6 lần.
* **Giới hạn:** đây là H1 đứng một mình, không phải dịch vụ API nạp đồng thời nhiều pipeline. Ở GĐ8, riêng dịch vụ (bm25 + dense) đã tăng card từ 1.377 lên 2.891 MB; cộng thêm bge-m3 (khoảng 1,3 GB) rất có thể vượt 4.096 MB. Chưa đo, cần đo lại ở GĐ18.
* Chênh độ trễ của bge-m3 giữa hai mức nền (734 so với 1.089 ms) chưa được giải thích; không rút kết luận từ đó.

## 11. Kết quả GĐ15 trên test (mở một lần, 2026-11-09)

`scripts/GD15/15f_run_test.py --confirm-test` → `results/final_test_matrix.json`, `reports/pipeline_test_{B1,H0,H1}_predictions.jsonl`. B0 dùng lại số ngày 2026-11-04 (cấu hình khớp manifest B0, tính lại ra đúng từng chữ số). Bản đóng băng (`15_freeze.json`, 40 tệp) còn khớp sau khi chạy. 2.203 câu.

| | Hit@20 | Hit@5 | MRR@10 | EM (đáp án đầu) | EM (mọi đáp án) | ms/câu (p50) |
|---|---|---|---|---|---|---|
| B0 | 0,887 | 0,793 | 0,686 | 0,222 | 0,330 | 64 |
| B1 | 0,874 | 0,709 | 0,516 | 0,220 | 0,316 | 116 |
| H0 | 0,977 | 0,919 | 0,777 | 0,261 | 0,383 | 142 |
| H1 | 0,977 | 0,969 | 0,914 | 0,250 | 0,361 | 744 |

Phán định ba giả thuyết ghi trước (EM `first_ref` là số chính): **HA ủng hộ** (cả B1 và H0), **HB ủng hộ**, **HC hỗn hợp** (Hit@5 tăng có ý nghĩa nhưng EM giảm có ý nghĩa, không phải "không đổi").

Quy tắc: kết quả test chỉ để báo cáo, không dùng để chọn lại cấu hình.

### Đoạn công bố (từ sổ phơi bày)

Sổ phơi bày test ghi 10 lần truy cập (metric_run: 4, error_analysis: 1, metadata_only: 5); 5 dòng được dựng lại từ thời gian sửa tệp vì xảy ra trước khi có sổ. Các lần tính chỉ số trên test: 2026-10-31 03_bm25_eval.py --test (BM25 trên test với cấu hình chốt trên dev (k1=0.7, b=0.2, stopwords)); 2026-11-02 04d_eval_reader.py --test (reader gold-context trên test (EM/F1 khi đưa đúng đoạn)); 2026-11-04 07a_run_split.py --split test (pipeline end-to-end B0 trên test); 2026-11-09 15f_run_test.py (test MỘT lần cho B1, H0, H1 với cấu hình đã đóng băng (B0 dùng lại số đã chạy)).
