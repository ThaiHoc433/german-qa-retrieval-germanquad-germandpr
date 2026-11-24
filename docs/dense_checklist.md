# Checklist dense retriever (GĐ12.1)

Rà soát phần dense đã làm ở GĐ6 theo danh sách "Lỗi cần phòng tránh" của `plan/01b_v1_phan_b_cai_tien.md` (giai đoạn 12).
Mỗi mục dẫn tới **bằng chứng có thể kiểm lại** (test hoặc tệp kết quả). Một test trong `tests/GD12/test_checklist_refs.py`
kiểm mọi đường dẫn nêu ở đây còn tồn tại, nên tài liệu này không lỗi thời âm thầm.

Phạm vi: artifact đang dùng cho B1 và H0 là `artifacts/dense/text384` (cấu hình `configs/dense.yaml`). `text256` là phương án đã bị thay.

| # | Lỗi cần phòng | Cách đã phòng | Bằng chứng | Trạng thái |
|---|---|---|---|---|
| 1 | Encoder không tương thích hoặc sai pooling | Dùng cặp DPR gốc (context + question encoder). Vector lấy từ `pooler_output` (vị trí [CLS]) đúng như DPR. Encoder được huấn luyện với segment id toàn 0 nên ép `token_type_ids = 0` kể cả với cặp (title, text) | `src/germanqa/retrieval/dense.py` (`_embed`); `tests/GD6/test_dense.py::test_token_type_ids_luon_bang_0_khi_goi_encoder` | Đạt |
| 2 | Sai định dạng đầu vào (title–text) | Plan bảo theo định dạng của model; **phép thử 300 cặp** chọn `passage_input: text` trên `original_text`: chỉ-text 256 đạt top-1 0,84; cặp (title, text) theo `token_type_ids` của tokenizer chỉ 0,487; chữ thường chỉ 0,52. Cấu hình lưu trong manifest và bị từ chối nếu lệch | `reports/dense_format_probe.json`; `tests/GD12/test_dense_compat.py::test_doi_dinh_dang_thi_bi_tu_choi` | Đạt (lệch có lý do so với plan, đã ghi) |
| 3 | Sai similarity / normalize một bên | DPR huấn luyện bằng tích vô hướng nên **không chuẩn hóa** cả hai phía; FAISS dùng `IndexFlatIP` (tìm chính xác). Chuẩn vector 25,07–25,84, manifest ghi `normalized: false` | `tests/GD6/test_dense.py::test_diem_la_tich_vo_huong_khong_chuan_hoa`, `::test_faiss_khop_voi_tinh_truc_tiep`; `reports/dense_index_check_text384.json` | Đạt |
| 4 | Index vector đúng nhưng mapping passage sai thứ tự | Số vector = số ID = 26.864, ID duy nhất, thứ tự đúng bằng thứ tự corpus; nạp lại từ đĩa từ chối khi corpus đã đổi (SHA-1 corpus trong manifest) | `reports/dense_index_check_text384.json` (`ids_match_corpus_order`, `ids_unique`, `n_vectors`); `tests/GD6/test_dense.py::test_thu_tu_mapping_khong_lech_khi_tron_id`, `::test_luu_doc_lai_va_tu_choi_corpus_da_doi`, `::test_check_embeddings_bat_loi_so_luong_trung_id_nan` | Đạt |
| 5 | Vector chứa NaN/Inf hoặc sai chiều | `check_embeddings` khi ghi; kiểm lại toàn bộ trên artifact đang dùng: chiều 768, hữu hạn | `reports/dense_index_check_text384.json` (`finite`, `dim`) | Đạt (kiểm toàn bộ cho `text384` ở GĐ12.1) |
| 6 | FAISS tìm khác tính trực tiếp | So top-20 của FAISS (dựng lại), FAISS (đọc từ đĩa) và numpy trên 200 câu: 196/200 hàng giống hệt; 4 hàng khác nhau cùng tập ID, chênh điểm tối đa 7,6·10⁻⁵ (hòa điểm); chỉ mục trong đĩa = chỉ mục dựng lại | `reports/dense_index_check_text384.json` (`faiss_vs_numpy`) | Đạt (không phải “giống hệt”: 4 hàng hòa điểm, có giải thích) |
| 7 | Vector fp16 lệch so với fp32 | So 300 đoạn: cosine nhỏ nhất 0,9999995; top-1 giống 100%, top-5 giống 100% | `reports/dense_index_check_text384.json` (`fp16_vs_fp32`) | Đạt |
| 8 | Truncation làm mất nội dung mà không ghi nhận | `passage_max_length: 384`; manifest đếm đoạn bị cắt: 2.923/26.864 (10,9%), token trung bình 236, p95 454, tối đa 2.204. Chọn 384 thay 256 bằng quy tắc ghi trước (Hit@5 +0,0148, cận dưới > 0) | `artifacts/dense/text384/manifest.json` (`truncation`); `baselines/B1.json` (`length_decision`); `results/12g_512_vs_384.json` | Đạt: đã thử 512 ở quy mô corpus, **kém hơn 384** (Hit@5 −4,5 điểm, cận trên −3,6), nên giữ 384 |
| 9 | Dùng index cũ sau khi đổi encoder hoặc tiền xử lý | Manifest ghi repo + revision của **cả hai** encoder, `passage_field`, `passage_input`, `similarity`; `DenseRetriever.from_dir(expected_cfg=…)` từ chối khi lệch | `tests/GD12/test_dense_compat.py::test_doi_encoder_hoac_revision_thi_bi_tu_choi`, `::test_tuong_thich_thi_khong_loi` | Đạt |
| 10 | Câu hỏi dài hơn giới hạn | `question_max_length: 64` cắt câu hỏi, có test | `tests/GD6/test_dense.py::test_cau_hoi_duoc_cat_theo_question_max_length` | Đạt |
| 11 | Giao diện khác BM25 làm hỏng pipeline | `retrieve(question, top_k)` → `[{passage_id, score, rank}]`, dùng trong `QAPipeline` không sửa pipeline | `tests/GD6/test_dense.py::test_dung_trong_qapipeline_khong_sua_pipeline`, `::test_retrieve_tra_dung_doan_thu_hang_bat_dau_tu_1` | Đạt |
| 12 | Đo thiếu chi phí | Mã hóa câu hỏi (GPU) trung bình 23,9 ms, tìm kiếm 6,1 ms, VRAM đỉnh 610 MB, RAM thêm 763 MB; CPU: 35,1 ms và 7,4 ms | `results/12d_resources_cuda.json`, `results/12d_resources_cpu.json` | Đạt |

## Còn mở

* **512 token (mục 12.3): đã thử, không nhận.** Mã hóa lại 26.864 đoạn ở 512 token (`artifacts/dense/text512`, 165 giây, 2,0% đoạn bị cắt so với 10,9% ở 384), truy hồi trên tập chọn 2.223 câu, so ghép cặp với 384 bằng đúng quy tắc nhận của GĐ12 (`scripts/GD12/12g_compare_512.py` → `results/12g_512_vs_384.json`): Hit@5 −4,50 điểm [−5,49; −3,55], Hit@20 −0,81 [−1,26; −0,40], 116 câu Hit@5 mất so với 16 câu được. `configs/dense.yaml` không đổi. Card của context encoder ghi nó huấn luyện đoạn văn với tối đa 300 token; việc cắt 256 → 384 → 512 không cải thiện đơn điệu ăn khớp với việc encoder kém ở độ dài xa 300, nhưng đây là **giả thuyết chưa kiểm**.
* **Báo cáo kiểm cho `text256`** (`reports/dense_index_check.json`) giữ nguyên làm tư liệu lịch sử; số liệu của B1/H0 dùng báo cáo `text384`.
