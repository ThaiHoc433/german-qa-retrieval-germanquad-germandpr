<!-- Sync Impact Report
Version: unratified scaffold -> 1.0.0 (initial adoption).
Principles: five generic slots replaced by project principles I–VII.
Added: Phạm vi và nguồn chuẩn; Quy trình phát triển và nghiệm thu.
Removed: generic examples only. Deferred placeholders: none.
Templates/commands unchanged; constitution read at runtime.
-->
# German Open-Domain Extractive QA — Constitution

## Core Principles

### I. Spec là nguồn chuẩn kỹ thuật, đóng băng theo phiên bản

Code MUST khớp module spec đang hiệu lực: V1.1, frozen 2026-09-11.
Khi lệch, MUST sửa code hoặc đề xuất phiên bản spec mới có decision log;
MUST NOT sửa bản frozen. V1.2 draft chưa thay thế V1.1.
Constitution quản trị quy trình; module spec định nghĩa hợp đồng kỹ thuật;
plan/task triển khai hợp đồng. Mâu thuẫn MUST giải quyết bằng amendment tường minh.

### II. Kỷ luật tập test

MUST NOT dùng GermanQuAD/GermanDPR test để chọn checkpoint, hyperparameter, R hay ngưỡng.
Checkpoint/hyperparameter dùng dev với metric P2; R chọn theo số lượng chunk hợp lệ
ở §2.3.d, không theo metric dev/test. Cấu hình/checkpoint MUST khóa trước test.
Được resume, kiểm tra tái lập hoặc chạy lại sau sửa lỗi triển khai có ghi nhận;
MUST giữ lịch sử run và công bố thay đổi ảnh hưởng kết quả. MUST NOT chọn cấu hình
hoặc lần chạy để báo cáo vì đã thấy kết quả test thuận lợi hơn.

### III. Chống leakage và giữ đúng nguồn corpus

Câu hỏi của bất kỳ test nào MUST NOT xuất hiện trong train/dev của bất kỳ thành phần nào.
Test mỗi dataset MUST giữ đúng test gốc; dòng train trùng test bị loại, không chuyển sang test
(§2.3.e). Corpus MUST xây từ nguồn được phép D6/§2.3.d, không thêm passage chỉ vì nó được
tuyển theo từng query làm hard negative. Chunk trùng một nguồn corpus hợp lệ vẫn được giữ
theo D6; MUST NOT dùng danh sách hard negative làm blacklist corpus.

### IV. Tái lập và truy vết được

Run đánh giá/chọn checkpoint MUST lưu cấu hình và fingerprint trước khi chạy.
Prediction runs dùng run_manifest (§8.4/§8.7); retrieval-only có metadata (§3.2),
được triển khai bằng hạ tầng manifest chung để tránh hai nguồn cấu hình.
Mọi lựa chọn ngẫu nhiên MUST tái lập và ghi seed; dùng SHA-256 ở nơi spec quy định,
MUST NOT dùng Python hash() để dẫn xuất seed. Model/tokenizer revision MUST được ghim.
Mỗi con số MUST truy ngược tới metrics, run_id, cấu hình, checkpoint được dùng
(null nếu không dùng) và corpus tương ứng.

### V. Không âm thầm bỏ hay sửa dữ liệu

Record/evidence bị loại MUST được báo cáo theo schema và mức chi tiết spec yêu cầu.
Gold rỗng vẫn lưu và đếm ngoài N_eval. Validator/Evaluation MUST dừng khi vi phạm
nhất quán (§3.3/§3.5); không tự sửa dữ liệu để tính tiếp. Lỗi inference một query
được ghi status=error và pipeline tiếp tục (§8.5), giữ kết quả giai đoạn đã hoàn thành.
Remap/dedup được phép ở §2.3 vẫn áp dụng cùng báo cáo tương ứng.

### VI. Kiểm thử hành vi, nghiệm thu bằng bằng chứng

Thay đổi hành vi code MUST có kiểm thử phù hợp; metric MUST khớp ví dụ tính tay.
Tài liệu/cấu hình MUST được kiểm chứng bằng đối chiếu nguồn, liên kết hoặc lệnh phù hợp;
không yêu cầu unit test cho sửa câu chữ. Cổng nghiệm thu theo §9.
MUST NOT thay điều kiện đúng giao thức bằng "Recall phải cao" hoặc "Dense phải thắng".
Test cũ trái V1.1 MUST cập nhật; test xanh theo kỳ vọng cũ chưa chứng minh tuân thủ.

### VII. Kết luận theo bằng chứng

Giả thuyết chỉ thành kết luận khi có bảng số liệu tương ứng. Hiện tại dùng kết luận
mô tả trong cấu hình/tập đánh giá đã chạy; MUST NOT khẳng định ưu thế ổn định hoặc
có ý nghĩa thống kê khi chưa có giao thức và bằng chứng hỗ trợ.
Báo cáo MUST nêu giới hạn một seed, gold theo nhãn, distractor lọc P4 và tài nguyên.
Số dự kiến/audit MUST được phân biệt với benchmark đã nghiệm thu.

## Phạm vi và nguồn chuẩn

Project root là nlp. Constitution ở thư mục cha không phải bản quản trị của project NLP.
Chỉ duy trì bản chính tại `.specify/memory/constitution.md`; tài liệu khác dẫn liên kết.

- [Spec hiệu lực V1.1](../../docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md).
- [Design và RQ](../../docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md).
- [V1.2 draft chưa hiệu lực](../../docs/superpowers/specs/2026-09-11-german-qa-retrieval-v1.2-amendments-draft.md).
- [Lộ trình học và triển khai](../../docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md).

## Quy trình phát triển và nghiệm thu

Trước chặng học: đọc spec liên quan, xác định input/output và task phụ thuộc.
Trước sửa code: viết thẻ task có file, test, acceptance và tài nguyên.
Sau task: lưu bằng chứng, phân biệt có code, qua test và nghiệm thu artifact.
Artifact mới MUST không ghi đè bộ cũ dùng làm audit; V1.1 dùng namespace riêng.
Website cập nhật từ spec/kết quả đã nghiệm thu, không định nghĩa giao thức.

## Governance

Constitution ưu tiên hơn plan/task về quản trị; module spec là hợp đồng kỹ thuật.
Amendment MUST ghi lý do, phạm vi và ảnh hưởng tới dữ liệu/code/test/kết quả, được chủ
project chấp thuận trước khi có hiệu lực. Draft không tự động có hiệu lực.
Semantic versioning: MAJOR cho thay đổi nguyên tắc không tương thích; MINOR cho bổ sung
nguyên tắc; PATCH cho làm rõ không đổi nghĩa. Plan/task MUST đối chiếu constitution
và spec hiệu lực trước triển khai. Không tự commit/push hoặc chạy training đầy đủ
chỉ vì tài liệu đã được duyệt.

**Version**: 1.0.0 | **Ratified**: 2026-09-11 | **Last Amended**: 2026-09-11
