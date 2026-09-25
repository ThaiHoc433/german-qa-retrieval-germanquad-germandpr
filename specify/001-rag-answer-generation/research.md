# Research — RAG answer generation

Ngày 2026-09-12. Nguồn chuẩn: [spec](spec.md), năm quyết định Clarifications và
[constitution](../../.specify/memory/constitution.md). Đây là quyết định thiết kế,
không phải kết quả benchmark hoặc xác nhận môi trường GPU đã chạy được.

## R1 — Hiện trạng và phạm vi reuse

Decision: sửa/reuse src/citation.py và tests/test_citation.py hiện có; đối chiếu parser,
prompt loader, clients, answerer, schema RAG, evaluation RAG, runner và web theo milestone.
Rationale: citation hiện chỉ có hàm matching/offset cơ bản. is_short_answer_supported
đang dùng substring nên Ein khớp Einstein, chưa đạt FR-008. Hàm exact hiện chưa chặn
quote rỗng. Test hiện chưa phủ NFC offset, parser hay sáu trạng thái citation.
Alternatives: tạo module citation thứ hai gây hai nguồn chuẩn; copy nguyên plan
Superpowers cũ bỏ sót năm clarification. Không dùng hai cách này.

src/evaluation.py có normalize_answer, exact_match_score, f1_score,
max_over_ground_truths và percentile để reuse. BUCKETS hiện còn reranking_failure;
không lấy test xanh cũ làm bằng chứng V1.1 đã nghiệm thu. Không sửa Evaluation V1
như một việc ngầm của V2; checkpoint/data/integration V1 là gate riêng.

Cập nhật inventory cuối bước plan: answer_parser.py, prompt_loader.py, llm_clients.py,
answerer.py, ba prompt và tests tương ứng đã xuất hiện trong repo. Các nhận xét code
ở trên là snapshot kiểm tra đầu phiên; không dùng chúng làm lệnh ghi đè file. Task M0
đọc bản mới nhất và kiểm xem các gap còn tồn tại; tài liệu này chưa nghiệm thu code.

## R2 — Citation và parser

Decision: parser theo answerer/backend, cùng một verifier sau adapter. Quote match
exact trước, rồi NFC + collapse whitespace trên toàn chuỗi có bản đồ về text gốc;
khác normalization chấm đáp án. Chỉ bỏ khoảng trắng và cặp dấu nháy ngoài quote,
không sửa nội dung bên trong. Token-support giữ trạng thái provisional theo FR-008.
Rationale: cần bảo toàn offset và không chấp nhận các phần của tên riêng.
Alternatives: fuzzy matching, repair bằng lần gọi LLM thứ hai, hoặc lấy API citation
làm sự thật đều trái quyết định đã chốt.

Chuỗi rỗng sau trim không phải citation hợp lệ. Nhiều lần khớp: chọn offset nhỏ nhất
trong mức matching thành công, đếm cả occurrence chồng lấp; không cộng exact và
normalized occurrences. Đây là quy ước implementation cho audit, không đổi metric QA.
Offset short answer được tìm trong đoạn text gốc tương ứng quote; không cộng độ dài
quote đã normalize vào offset gốc. V2 dùng Python Unicode code-point [start,end);
frontend phải chuyển code points sang đoạn DOM, không dùng trực tiếp UTF-16 slice.

## R3 — Claude adapter

Decision: mỗi passage là một plain-text document; gửi title riêng và text gốc nguyên
vẹn. document_index bắt đầu từ 0, map về P1..Pn. Giữ thứ tự text blocks khi ghép output,
đếm tất cả citation objects trên mọi block. Không dùng structured output cùng citations.
Citation chỉ hỗ trợ text document, không dùng PDF/page offsets ở feature này.
Rationale: API trả citations theo block, title không phải nội dung có thể trích dẫn.
Alternatives: đọc duy nhất content[0] mất citation; nối tự ý thêm newline có thể
biến output sai thành hợp lệ. [Tài liệu citations](https://platform.claude.com/docs/en/build-with-claude/citations).

Decision: đặt max_retries=0 và timeout tường minh cho cả đếm token/generation;
mock transport kiểm số lần gọi. SDK mặc định retry hai lần với một số lỗi.
[Python SDK](https://platform.claude.com/docs/en/api/sdks/python).

Decision: dùng canonical model ID được cấu hình trước dev, ghi cả requested và
response model ID, API/SDK version/request ID; không hard-code model alias latest.
Không yêu cầu ID phải có ngày trong tên vì canonical ID mới có thể không có ngày.
Không hứa bit-for-bit API reproducibility.
[Model versioning](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions).

Token counting của API là ước lượng, có thể khác usage thực tế. Đếm cả request đã
đóng gói, dành output và safety margin cấu hình; nếu provider vẫn từ chối vì context
thì map context_overflow, không retry/cắt. Đếm token không phải generation attempt;
ghi count request riêng và bao gồm trong prepare_context/total latency.
[Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting).

## R4 — Model mở và tài nguyên

Decision: ứng viên smoke ban đầu là Qwen/Qwen2.5-3B-Instruct; model card có tiếng Đức,
3.09B tham số và context model 32,768 token. License trên card là qwen-research;
ghi license/revision trước khi tải/bàn giao, không mặc định Apache.
Đây là ứng viên kỹ thuật, chưa phải model thắng dev.
[Model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).

Decision: CPU local dùng fake clients và Python 3.13 hiện có. Thử GPU trong môi trường
Colab riêng, đóng băng phiên bản thực tế sau smoke. Cấu hình smoke dự kiến batch=1,
greedy do_sample=false, output budget=256, operational context limit=4096; thử 4-bit
NF4 compute float16 trên T4. Đây là giá trị thử nghiệm cần khóa trên dev, không phải
bảo đảm VRAM/latency hoặc benchmark. Không ép cài lại môi trường V1.
Transformers có tích hợp bitsandbytes 4/8-bit; chọn float16 để tránh giả định phần cứng
T4 hỗ trợ bf16 như ví dụ của GPU khác.
[Bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

Alternatives: fine-tune LLM làm rộng phạm vi; chọn checkpoint cuối dựa trí nhớ hoặc
test làm sai kỷ luật dữ liệu. Dùng model pretrained instruction và dev-selection.
Tên revision/lockfile được sinh ở task environment smoke thay vì bịa hash trong plan.
Chỉ tối đa ba bộ prompt, mỗi bộ có bản with-context và closed-book/backend tương ứng.

## R5 — Một pipeline, hai lối vào

Decision: Answerer trả kết quả có schema; runner giữ timing và serialization.
Batch runner ghi manifest trước query đầu; web gọi pipeline không ghi artifact.
Khi V1 Integration xuất hiện, tách/reuse lớp tính toán và chứng minh output Reader
không đổi bằng frozen fixture/checkpoint; không sửa spec frozen để hợp thức code.
Alternatives: cho web gọi script batch sẽ ghi metrics/gold vào chỗ nghiên cứu;
ép schema RAG vào PredictionRecord V1 làm hỏng validator hiện hành.

Web có thể đọc gold dev để hiển thị/chấm tạm thời nhưng không phải run nghiên cứu
chính thức. Tất cả số đưa vào báo cáo phải chạy lại qua batch runner có manifest.
Quy tắc này làm rõ phạm vi tái lập của Constitution IV, không miễn manifest cho research.

## R6 — Web service và capability

Decision: FastAPI cùng origin, lifespan nạp tài nguyên khả dụng một lần, chia sẻ
checkpoint giữa các tier, index/cache theo fingerprint. Availability theo config,
không bắt BM25 phải có FAISS hoặc rerank-off phải có reranker. Startup health chỉ là
snapshot; request vẫn có thể lỗi sau đó. Cache/log không giữ API key hoặc header auth.
TestClient chạy trong context manager để thực thi lifespan; fake service được inject.
[FastAPI testing lifespan](https://fastapi.tiangolo.com/advanced/testing-events/).

Không tự mở tunnel Colab, deploy hay gọi paid API khi lập plan. Endpoint web phải
được cấu hình sẵn; server không nhận URL endpoint tùy ý từ browser. App chỉ bind
127.0.0.1, validate Host/Origin và render question/model/passage bằng textContent,
không innerHTML. Prompt coi passage là dữ liệu, không làm theo chỉ dẫn trong passage.

## R7 — Metric và ma trận

Decision: đánh giá short_answer với hàm V1; tỷ lệ citation chỉ là grounding proxy,
không kiểm claim. Mẫu số, lỗi và N/A đã chép đầy đủ vào FR-010. Oracle giới hạn năm
khác reader oracle toàn gold. API và open thay cả model lẫn citation mechanism;
không diễn giải một run API là thí nghiệm cô lập cơ chế citation.
Alternatives: nhân baseline theo tier làm tăng giả số run; chỉ tính query thành công
làm tăng giả EM/F1. Sáu cấu hình open + một API là số cấu hình test, không bao gồm
các run chọn prompt/model trên dev.

Thứ tự tie mặc định ghi trước dev: dense/base/off, dense/base/on,
dense/large/off, dense/large/on. Chọn F1 rồi EM rồi index trong danh sách.

## R8 — Gate còn cần bằng chứng, không phải câu hỏi thiết kế chưa trả lời

- FR-008: provisional, có test/dev review riêng trước khóa protocol; không tự nâng
  thành approved chỉ vì tạo plan.
- Model revision/SDK lock/hardware/context budget: lấy từ smoke thực tế, ghi manifest.
- Adapter API: test fixtures ngoại tuyến trước, live smoke riêng khi có quyền gọi.
- V1 data/checkpoint/candidate fingerprint: nghiệm thu ở V1, kiểm lại khi tích hợp.
- Không còn NEEDS CLARIFICATION trong quyết định thiết kế; các gate trên là output
  của task thực thi cần có trước benchmark.
