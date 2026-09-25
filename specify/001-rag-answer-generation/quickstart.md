# Quickstart — validation guide

Đây là hướng dẫn nghiệm thu theo [plan](plan.md). Các lệnh “sau triển khai” chỉ chạy
sau khi task tương ứng tạo file; không phải bằng chứng implementation đã có.
Working directory: D:\Project\2026-2027-HK1\nlp.

## 1. Kiểm tra phần hiện có

PowerShell, dùng interpreter đã cài thay alias Windows Store:

```powershell
& 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -m pytest tests/test_citation.py tests/test_evaluation.py tests/test_schemas.py tests/test_bm25_retriever.py -q -p no:cacheprovider
```

Ghi kết quả thật khi chạy. Test xanh cũ không xác nhận FR-008 hoặc toàn spec V1.1;
code hiện vẫn dùng substring support, cần test Ein/Einstein khi implement M1.

## 2. Môi trường sau task M0

M0 tạo requirements-rag-core.txt (pytest + test/web deps đã pin),
requirements-rag-api.txt và requirements-rag-gpu.txt riêng. Những file này chưa được
tạo bởi plan. API/GPU không là dependency của unit test core.

```powershell
py -3.13 -m venv .venv-rag
.\.venv-rag\Scripts\python.exe -m pip install -r requirements-rag-core.txt
```

Không dùng pip freeze của toàn máy làm lock V1. Ghi environment version trong artifact
smoke. GPU setup Colab có lock riêng theo runtime thực tế; không hứa clone venv Windows.

## 3. Core/contract tests sau M1–M4

```powershell
.\.venv-rag\Scripts\python.exe -m pytest tests/test_schemas_rag.py tests/test_citation.py tests/test_answer_parser.py tests/test_prompt_loader.py tests/test_context_budget.py tests/test_llm_clients.py tests/test_answerer.py tests/test_evaluation_rag.py tests/test_rag_run.py -q
```

Expected behavior, không đặt trước số test pass:

- Exact/NFC/whitespace quote match trả substring offset đúng source; empty quote bị loại.
- Einstein/Ein và empty-normalized answer theo rule provisional; multiple source giữ raw.
- Four-line open vs two-line closed/Claude; parser no-answer đúng mode, lỗi có null fields.
- Context đúng giới hạn gọi client1 lần; vượt1 token không gọi generate; N_eval vẫn giữ.
- Fake SDK timeout/rate-limit/refusal không retry; citation từ mọi text block được đọc.
- Fixture arithmetic ở [pipeline contract](contracts/pipeline.md) khớp;7/5/4 buckets
  phủ N_eval, error vẫn0EM/F1, N/A không thành0.
- Gold bản sao đổi1 byte nhưng manifest cũ bị từ chối; prediction rỗng có manifest
  vẫn chấm; resume không gọi lại query đã có record/error.

Tất cả network trong suite bị fake/blocked. Live tests không được tự skip rồi tính là
đã nghiệm thu; báo riêng “chưa chạy” nếu chưa có GPU/credentials/checkpoint.

## 4. Reader/pipeline sau M5

```powershell
.\.venv-rag\Scripts\python.exe -m pytest tests/test_rag_pipeline.py -q
```

Fixture adapter luôn chạy; checkpoint-dependent smoke riêng kiểm cùng question/context
Reader trực tiếp và qua adapter cho cùng span, passage_id, offsets và status. Không so
bit-for-bit raw logits giữa hai hardware khác nhau. Chỉ đạt gate thật khi có checkpoint
V1 và các artifact V1.1 được xác minh, không chỉ fake reader.

## 5. Web fake mode sau M6

M6 tạo app factory và CLI/config fake mode explicit. Các lệnh dự kiến:

```powershell
.\.venv-rag\Scripts\python.exe -m pytest tests/test_rag_api.py -q
$env:RAG_RUNTIME_MODE = 'fake'
.\.venv-rag\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Mở http://127.0.0.1:8000. Fake mode hiển thị banner “Dữ liệu mô phỏng”, không fallback
từ real mode sang fake. Thử hỏi, chọn cấu hình, dev question, compare, lỗi citation.
Kiểm tra375/768/1440px, keyboard/tab/focus, quote Unicode/highlight và request trùng.
Kiểm tra thư mục research không bị ghi. Dừng server và bỏ env fake trước real mode.

## 6. Live smoke và khóa benchmark

M3/M7 tạo CLI, chưa chạy từ tài liệu này:

```powershell
.\.venv-rag\Scripts\python.exe scripts/smoke_rag.py --config configs/rag/dev-smoke.json
.\.venv-rag\Scripts\python.exe scripts/run_rag.py --config configs/rag/open-base-off-dev.json
.\.venv-rag\Scripts\python.exe scripts/evaluate_rag.py --manifest results/rag/<run_id>/run_manifest.json --run-id <run_id>
```

Trong lệnh cuối thay <run_id> bằng ID thật do runner trả; đây là tham số runtime,
không phải placeholder thiết kế. Không cung cấp khóa thật trong command history.

Gate trước test: V1 data/checkpoint/candidates đúng hash; smoke GPU/API có metadata;
support rule provisional đã được review và ghi quyết định; tối đa3 bộ prompt dev;
chọn F1→EM→fixed-order; khóa model/tokenizer/quantization/budget/prompt. Một config đã
khóa dùng nhất quán mọi tier/baseline; không chọn lại theo test.

Batch tạo manifest rồi mới chạy, tách namespace RAG; đo warmup và timing đúng scope.
RAG6+API1 là số cấu hình test, không phải tổng số dev calls. User cấp quyền live API
riêng trước khi phát sinh lượt trả phí; plan không tự tải model/train/gọi API.

## 7. Evidence cần bàn giao

Log tests, environment locks, fixture/dev rule review, corpus/candidate/checkpoint hashes,
prompt/model configs, per-query prediction/metrics, bảng RQ4 và run manifest. So sánh
API/open có giới hạn khác model/citation; oracle5 khác Reader oracle toàn gold;
không dùng UI screenshot làm bằng chứng chất lượng hay lấy số giả điền bảng.
