# German QA — V1.2 Amendment Draft

- Status: **DRAFT — CHƯA CÓ HIỆU LỰC**. Ngày: 2026-09-11.
- Đây là phụ lục đề xuất, không phải bản module spec hợp nhất V1.2.
- Baseline đang hiệu lực: [V1.1 frozen](2026-09-10-german-qa-retrieval-module-specs.md).
- Phạm vi: làm rõ phụ thuộc triển khai và provenance đánh giá; không đổi model,
  corpus tiers, split, metric, checkpoint criterion hay span scoring của V1.1.
- Chủ project đã duyệt việc soạn draft; chưa có quyết định phát hành thay thế V1.1.

## A1 — Hạ tầng run có trước lần đánh giá đầu tiên

Hạ tầng định danh run, fingerprint và manifest (§8.4, §8.7) được triển khai cùng
Evaluation, trước run BM25 đầu tiên. Runner reader-only triển khai cùng Reader.
Orchestration end-to-end, lỗi liên giai đoạn và latency toàn pipeline hoàn thiện ở Integration.

Đây là làm rõ đồ thị phụ thuộc, không đổi thứ tự xử lý query của pipeline.
Schema run đầy đủ được thiết kế từ đầu; không đợi Integration mới bổ sung checkpoint
fingerprint, checkpoint_bundle_hash hay điều kiện đo cần thiết cho oracle.

## A2 — Một nguồn cấu hình cho retrieval-only

V1.1 §3.2 cho phép retrieval-only có metadata, §8.7 mô tả run manifest.
Đề xuất dùng entry run_manifest làm nguồn chuẩn cho mọi run đánh giá/chọn checkpoint;
metadata kèm artifact chỉ tham chiếu run_id, không giữ cấu hình thứ hai độc lập.

- BM25/dev: mode=retrieval_only; checkpoint không dùng ghi null; điều kiện đo/warm-up
  null nếu không đo latency. Full identity/config/fingerprints lưu trước chạy.
- Dense checkpoint selection: cùng cấu trúc, fingerprint hai tower và bundle hash.
- Oracle: mode=reader_only, retriever=oracle, reranker_enabled=false, retrieved=[],
  reranked=null, latency.retrieval=0. Fingerprint reader và điều kiện đo có ngay ở runner này.
- Fingerprint checkpoint/bundle theo V1.1 §8.4/§8.7; không tự sáng tạo hash mới.
- Đổi một byte bản sao gold/corpus mà giữ manifest cũ phải bị từ chối.

## A3 — Phân biệt điểm retriever và điểm sau rerank

Criterion P2 cho reranker là Recall@5 trên danh sách sau rerank; không gán metric đó
cho danh sách retrieved trước rerank. Đề xuất thêm metadata sau cho run retrieval-only:

- evaluation_stage: retrieval hoặc post_rerank.
- candidate_run_id: null cho retrieval; run_id tạo top-20 cố định cho post_rerank.
- candidate_results_path, candidate_results_sha256: null nếu không dùng; còn lại
  định danh byte artifact candidate nguồn, đường dẫn resolve theo thư mục manifest.
- retriever vẫn là bm25/dense tạo pool; checkpoint bundle gồm retriever và reranker
  thực dùng; rows sau rerank có retriever_name=cross_encoder.

Run post_rerank chỉ báo Recall@5/Hit@5 để chọn checkpoint và số query hợp lệ;
không suy Recall@20 của retriever từ danh sách đã bị cắt còn 5. Kiểm tra subset,
số lượng, candidate fingerprint và cùng gold/corpus trước chấm.
Artifact/metrics post_rerank không ghi đè kết quả retrieval.

Trước khi A3 được phê chuẩn, task có thể dùng fixture để hoàn thiện thiết kế adapter;
không gắn schema mở rộng này nhãn V1.1 hay chạy selection production theo một schema draft.

## A4 — Chính sách kết luận hiện tại

RQ nằm trong [design](2026-09-10-german-qa-retrieval-design.md).
Đợt hiện tại báo cáo kết luận mô tả; không bổ sung khoảng tin cậy hoặc kiểm định vào draft này.
Không giả định reader_failure ổn định khi phần query đến reader thay đổi.
Nếu mở rộng thống kê: lập amendment riêng, ghi đơn vị lấy mẫu, đại lượng so sánh,
quy tắc xử lý lỗi/ghép query và phạm vi diễn giải trước khi chọn cách phân tích theo kết quả.

## Ảnh hưởng và cổng phê chuẩn

| Phần | Tác động dự kiến | Kiểm tra trước phát hành |
|---|---|---|
| Shared run infra | Viết cùng Evaluation | Identity, file/checkpoint hash, manifest rỗng, sai fingerprint |
| Retrieval-only | Dùng một manifest chuẩn | Cấu hình/gold/corpus khớp, null đúng khi không đo |
| Oracle runner | Có trước chọn reader checkpoint | Gold-only input, answer fields, latency/warm-up |
| Post-rerank adapter | Metadata mới A3 | Không lẫn stage, candidate cố định, subset và checkpoint trace |
| Lộ trình | Run infra ở chặng 6 | Các runner dùng chung module, không chờ chặng 9 |

Để phát hành: chủ project chốt A1–A4; ghi ngày hiệu lực; tạo module spec hợp nhất
phiên bản mới; cập nhật tham chiếu constitution/plan; giữ nguyên V1.1.
Không có thao tác phát hành tự động chỉ vì draft được lưu.
