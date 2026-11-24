# Bộ 10 infographic — kế hoạch QA tiếng Đức

Ảnh mô tả kế hoạch đề xuất cho GermanQuAD/GermanDPR, theo phong cách của `../plan-1-infographic-v2.png`. Các sơ đồ không khẳng định project đã được triển khai hoặc đã đạt kết quả benchmark.

Đọc theo thứ tự 01 → 10. Mỗi ảnh giải thích mục tiêu, luồng xử lý, phần cần code, kiến thức và đầu ra.

| Bước | Ảnh | Trọng tâm |
|---|---|---|
| 01 | [CẤU TRÚC PROJECT & CẤU HÌNH](01_cau_truc_project.png) | Kiến trúc dùng chung giữa notebook, đánh giá và API |
| 02 | [CHUẨN BỊ DỮ LIỆU & CORPUS](02_du_lieu_va_corpus.png) | Dữ liệu sạch, chia tập và nhãn passage liên quan |
| 03 | [BASELINE BM25 & BỘ ĐÁNH GIÁ](03_bm25_va_metrics.png) | Baseline lexical và kiểm tra các metrics |
| 04 | [EXTRACTIVE READER TIẾNG ĐỨC](04_extractive_reader.png) | Fine-tune, ánh xạ span và gold-context QA |
| 05 | [GHÉP BM25 + READER](05_pipeline_bm25_reader.png) | Kết nối retrieval với reader thành pipeline |
| 06 | [DENSE RETRIEVER & FAISS](06_dense_retriever_faiss.png) | Hai encoder, embeddings và FAISS index |
| 07 | [ĐÁNH GIÁ & RERANKING](07_danh_gia_va_reranking.png) | Đánh giá từng bước và so sánh reranking |
| 08 | [ĐÓNG GÓI BACKEND FASTAPI](08_fastapi_backend.png) | HTTP API, lifecycle và quản lý tài nguyên |
| 09 | [GIAO DIỆN REACT](09_react_frontend.png) | Giao diện hỏi–đáp và kết quả benchmark |
| 10 | [DOCKER, DEPLOY & KIỂM TRA](10_docker_deploy.png) | Docker, artifacts, triển khai và kiểm tra |

## Cách hiểu trạng thái

- CẦN CODE / KIỂM TRA: việc sẽ thực hiện và tiêu chí kiểm tra, không phải xác nhận đã hoàn thành.
- MỞ RỘNG: hạng mục tùy chọn như tự fine-tune dense retriever.
- Ví dụ Berlin chỉ minh họa đầu vào/đầu ra, không phải kết quả đo.
- Các mẫu huấn luyện GermanDPR không thay thế corpus chung khi đánh giá retrieval.
- Chỉ các câu hỏi có đáp án chuẩn mới được chấm EM/F1 và phân loại lỗi dựa trên bằng chứng chuẩn.

## Nguồn nội dung

- Kế hoạch 10 bước trong cuộc trao đổi.
- [GermanQuAD và GermanDPR — MRQA 2021](https://aclanthology.org/2021.mrqa-1.4/)
- [GermanQuAD](https://huggingface.co/datasets/deepset/germanquad)
- [GermanDPR](https://huggingface.co/datasets/deepset/germandpr)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
- [Vite deployment](https://vite.dev/guide/static-deploy.html)

## Phương thức tạo

Dùng công cụ image_gen tích hợp. Ảnh gốc chỉ làm tham chiếu phong cách; nội dung được chia thành 10 chủ đề riêng. Bộ prompt nằm trong [prompts.json](prompts.json).

