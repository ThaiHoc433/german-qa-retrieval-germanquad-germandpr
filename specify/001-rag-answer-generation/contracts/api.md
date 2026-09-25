# Local web API contract

Base origin: http://127.0.0.1:8000. FastAPI cùng origin với static HTML/CSS/JS.
JSON HTTP response khác text output của model; ràng buộc không dùng JSON chỉ dành
cho output model mở. Data fields theo [data-model](../data-model.md).

## Request config

Config gồm answerer, model_backend, retriever, rerank, tier. answerer/backend enum
theo spec; backend null khi extractive. rag/extractive cần bm25|dense và tier
base|mid|large (web hỗ trợ các tier thực có, RQ4 chỉ base/large).

Với closed_book/oracle_rag, trước validation availability và khóa inflight, chuẩn hóa
retriever=null, rerank=false, tier=null; trả effective_config trong response. Không
ghi đè raw request trong error audit. Không nhận model ID, prompt text, endpoint URL,
đường dẫn corpus hay secret từ browser. Cấu hình đó do startup config quản lý.

Input question phải là string không rỗng sau trim. Unknown field split hoặc file/path
bị từ chối422. query_id chỉ lookup trong map QA dev đã load; ID train/test/không tồn
tại trả404 đồng nhất. Không suy split từ prefix vì ID gốc train có thể thuộc dev.

## Routes

| Method/path | Input | Output |
|---|---|---|
| GET /api/config | Không | public_answerers, dev_only_answerers, backends, tiers, config capabilities + reason, model/prompt metadata không secret |
| POST /api/ask | question + config | Một PredictionRagRecord web, run_mode=ask, request_id; không gold/EM/F1/bucket |
| GET /api/dev-questions | tier tùy chọn | {questions:[{query_id,question_preview}], split:dev}; không answer/context/gold |
| GET /api/dev-questions/{query_id} | answerer/backend/retriever/rerank/tier query params | Prediction + gold references, EM/F1/bucket; run_mode=dev_evaluation |
| POST /api/compare | question HOẶC query_id, config_a, config_b | {comparison_id, input, a, b}; cùng một câu hỏi cho cả hai |

/ask từ chối oracle_rag422. Dev detail không có default answerer ngầm: thiếu trường
bắt buộc trả422. tier tùy chọn khi closed_book/oracle; khi oracle cần gold corpus đã
validate ở startup dù tier=null. Danh sách dev với tier chỉ kiểm tier khả dụng, không
lọc N_eval theo retrieval hits. Cùng dataset/split giữ cùng QA population mọi tier.

/compare query_id xác định duy nhất dev question, không chấp nhận thêm question khác
hoặc query_id trong từng config. Nếu có question tự nhập, cả hai config theo /ask.
Oracle chỉ hợp lệ với query_id dev. Chạy A rồi B để giới hạn GPU memory, không ngụ ý
latency cạnh nhau là benchmark được kiểm soát. Tất cả config được validate trước khi
chạy; syntax error trả422. Runtime mỗi bên giữ status/http_status/body riêng, một bên
lỗi không làm mất bên còn lại; response envelope200. Hai bên cùng config được chạy
hai request tuần tự, không cache câu trả lời.

## HTTP và trạng thái

| HTTP | Điều kiện | Body tối thiểu |
|---|---|---|
| 200 | Pipeline đã xử lý, gồm generation timeout/refusal/format/context_overflow | prediction với error_type đúng tầng |
| 422 | Input sai enum/rỗng, oracle ở ask, thiếu backend, nhận split/path | error_type=invalid_request, message |
| 503 | Trước pipeline: artifact/endpoint/checkpoint cần thiết chưa sẵn sàng | error_type=resource_unavailable, reason_code |
| 429 | Cùng query_id và effective config đang chạy | error_type=duplicate_inflight_request |
| 404 | query_id không thuộc map dev | error_type=dev_question_not_found |

Known generation lỗi sau pipeline bắt đầu là generation_status=error; lỗi runtime
retrieval/selection/read/verify là execution_status=error, giữ kết quả stage đã xong.
Không bắt mọi Exception rồi gán format_violation. Bug HTTP chưa tạo được record có
thể trả500 kèm request_id và thông báo đã bỏ secrets.

Availability trong /config là snapshot lúc startup; endpoint chết sau đó vẫn có thể
tạo lỗi request. Reload config bằng restart, không thay prompt/model giữa run.
Missing .env không làm app crash; extractive chỉ available nếu Reader/index thật có.

## Inflight, timeout, dữ liệu read-only

Một process/worker. Khóa theo query_id + canonical hash effective config (gồm fingerprint
model/prompt/corpus), tạo trước inference và release trong finally. Compare dev cũng
dùng cùng khóa; hai request độc lập trùng bị429. Timeout/backend deadline cấu hình ở
server, không do browser quyết định. SDK retry=0, endpoint wrapper retry=0.

Web không tạo run_manifest/research artifact, không sửa dataset/checkpoint; cache model
và query embedding, log lỗi đã bỏ secrets được phép. In-memory map dev không bao gồm
test. Số EM/F1 trên web chỉ để quan sát, không đưa vào báo cáo nghiên cứu.

## Response và UI

Response có answer_result, retrieved top20, selected≤5, passage_texts và offsets,
latency_ms theo stage, effective_config, prompt/model identity, request_id/run_mode.
Run_id=null trên web, không bịa research manifest. Extractive prompt/model_backend
metadata không áp dụng là null; Reader checkpoint identity vẫn được trả.

Mỗi passage có title/text và nhãn Pk. Tô quote vàng, short span có viền khi có offset
chính xác; normalized support không có offset thì hiển thị nhãn thay vì highlight giả.
Backend offset là Unicode code point; frontend dùng Array.from(text) để dựng span
text an toàn, không nhét HTML từ passage/model. Màu luôn kèm chữ, focus rõ, form/tab
dùng bàn phím, aria-live thông báo đang chạy/lỗi. Kiểm tra375/768/1440px và overflow.

Public hiển thị answer/citation status; dev thêm short_answer/EM/F1/gold/bucket.
Closed-book hiển thị “Không dùng nguồn — citation không áp dụng”, không hiển thị
verified hoặc0%. Không có web history/export/streaming/public deployment trong V2.

## Contract tests

TestClient với fake lifespan/service/client: startup thiếu từng dependency, BM25 vẫn
dùng được khi thiếu FAISS; oracle public422; tier typo422/tier chưa build503; lookup
dev allowlist404; closed-book effective null; duplicate429/release sau error; compare
cùng query; HTTP200 format/timeout/context error; không có filesystem research writes.
Kiểm tra no retry bằng call count, quote offsets có Unicode ngoài BMP và HTML payload
được render thành text. GET inference giữ route đã chốt và dùng Cache-Control:no-store;
frontend chỉ gọi khi người dùng bấm, không prefetch link để tránh phát sinh lượt LLM.
