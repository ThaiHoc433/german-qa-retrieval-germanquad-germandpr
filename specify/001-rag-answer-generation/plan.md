# Implementation Plan: RAG Answer Generation with Verifiable Citations

**Branch thực tế lúc lập plan**: `test1` | **Feature ID**: `001-rag-answer-generation`
**Date**: 2026-09-12 | **Spec**: [spec.md](spec.md)
**Input**: specify/001-rag-answer-generation/spec.md
**Status**: Kế hoạch triển khai; chưa nghiệm thu V2 hoặc chạy benchmark.

## Summary

Giữ Reader V1 qua adapter, thêm LLM answerer (open/Claude), bộ kiểm chứng citation,
đánh giá RQ4 và web local dùng chung pipeline. Làm phần thuần hàm/fake-client trước;
tích hợp và thí nghiệm thật đi qua gate dữ liệu/checkpoint V1. Mọi artifact do Spec Kit
tạo nằm trong feature directory này.

[Research](research.md) giải quyết lựa chọn kỹ thuật;
[data-model](data-model.md), [contracts](contracts/api.md) và
[quickstart](quickstart.md) là hợp đồng cho giai đoạn tạo tasks.

## Technical Context

**Language/Version**: Python 3.13 local cho fixture hiện có; môi trường Colab riêng,
ghi phiên bản thực sau smoke. Không giả định mọi gói GPU hỗ trợ interpreter local.
**Primary Dependencies**: stdlib/dataclasses, pytest; reuse metric V1; FastAPI/Pydantic,
Uvicorn, HTTPX cho web; anthropic cho API; Transformers/PyTorch/Accelerate/bitsandbytes
cho open model. SDK/GPU imports lazy để test core không cần network/GPU.
**Storage**: JSONL/JSON và file prompt; không DB. results/rag/<run_id>/ dự kiến cho
batch; web chỉ có cache/runtime logs. Secrets chỉ đọc phía server.
**Testing**: unit/contract bằng fake clients, fake clocks, tokenizer và pipeline;
integration Reader với frozen fixture/checkpoint khi sẵn sàng; smoke GPU/API tách riêng.
**Target Platform**: Windows local CPU; Colab T4 batch1; web 127.0.0.1 một worker.
**Project Type**: Python library + batch CLI + local web.
**Performance Goals**: đo mean/p50/p95 theo từng stage, có n_latency và environment;
không đặt ngưỡng EM/F1 hay latency chưa có bằng chứng làm gate.
**Constraints**: tối đa5 context passages; không truncate/retry/fallback; output không
claim-level verified; research manifest trước run; FR-008 provisional; test không tune.
**Scale/Scope**: Dense × rerank on/off × base/large =4 RAG; closed_book + oracle5 =2;
API thêm1 RAG; đối chiếu extractive V1 tương ứng. Dev selection runs không nằm trong6+1.
Web vẫn hỗ trợ BM25 khi artifact hợp lệ, dù BM25 không ở ma trận RQ4.

## Constitution Check

| Nguyên tắc | Trước research / sau design | Bằng chứng trong plan |
|---|---|---|
| I. V1.1 frozen | Pass / Pass | Schema RAG riêng; adapter hồi quy; không sửa source spec |
| II. Test discipline | Pass / Pass | Dev-only selection, config lock; research runner kiểm split/manifest |
| III. Leakage và corpus | Pass / Pass | Reuse QA split V1; oracle dùng gold; web chỉ lookup dev |
| IV. Tái lập | Pass / Pass | research identity/fingerprints trước inference; web output không dùng làm số báo cáo |
| V. Không sửa âm thầm | Pass / Pass | Raw generation/citations giữ; fingerprint mismatch dừng; query runtime lỗi được lưu |
| VI. Kiểm thử | Pass / Pass | Contract fixtures, Unicode spans, zero denominators, Reader equivalence |
| VII. Bằng chứng | Pass / Pass | Model smoke/config là dự kiến; citation proxy có giới hạn; không bịa score |

Pass ở đây là tuân thủ của thiết kế, không chứng nhận implementation. FR-008 được
người dùng tạm chấp nhận để làm fixture; khóa benchmark yêu cầu ghi quyết định xác nhận
sau dev review. V1 chưa đạt gate không ngăn viết parser/schema/route giả lập.

## Project Structure

### Documentation (this feature)

```text
specify/001-rag-answer-generation/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── api.md
│   └── pipeline.md
└── checklists/requirements.md
```

tasks.md sẽ do speckit-tasks tạo sau bước plan; chưa phải output của bước này.

### Source Code (repository root, planned)

```text
src/
  citation.py             # đã có, sửa/reuse
  evaluation.py           # V1, chỉ reuse hàm metric
  schemas_rag.py
  answer_parser.py         # đã có, review/reuse
  prompt_loader.py         # đã có, review/reuse
  context_budget.py
  llm_clients.py           # đã có, review/reuse
  answerer.py              # đã có, review/reuse
  evaluation_rag.py
  rag_pipeline.py
  rag_run.py
  prompts/
    answer_de_v1.txt
    answer_de_closed_book_v1.txt
    answer_de_claude_v1.txt
app/
  main.py
  routes.py
  service.py
  static/index.html
  static/styles.css
  static/script.js
scripts/
  run_rag.py
  evaluate_rag.py
  smoke_rag.py
tests/
  test_citation.py        # đã có, mở rộng
  test_schemas_rag.py
  test_answer_parser.py
  test_prompt_loader.py
  test_context_budget.py
  test_llm_clients.py
  test_answerer.py
  test_evaluation_rag.py
  test_rag_run.py
  test_rag_pipeline.py
  test_rag_api.py
```

**Structure Decision**: Dùng các folder chuẩn hiện có. rag_pipeline.py là lớp V2 gọi
retrieval/selection lõi V1 khi sẵn sàng; không tạo retriever thứ hai hoặc thay tên schema
V1. rag_run.py dùng lại shared run helpers V1 khi có, thêm metadata RAG tại boundary.
Nếu integration V1 đã cung cấp extension point phù hợp khi thực thi, reuse thay vì
sao chép. Quyết định interface đã ghi trong contracts/pipeline.md.

Plan Superpowers cũ tại future_RAG/superpowers/plans/ chỉ bao phủ phần độc lập V1,
có code mẫu trước clarification. Nó là tài liệu tham khảo, không chạy nguyên xi:
token-support, baseline buckets, oracle5 và selection tie lấy theo spec feature này.
Code citation hiện có thuộc người dùng; khi implement phải sửa nhỏ có test thay vì ghi đè.

Inventory cuối bước plan: repo đã có thêm answer_parser.py, prompt_loader.py,
llm_clients.py, answerer.py, ba prompt và test tương ứng. Các file này thay đổi trong
lúc lập tài liệu; M0 phải đối chiếu lại trước khi giao tasks. M1–M3 là review/reuse/fill
gaps cho file đã có, không lệnh tạo đè. Chưa tuyên bố chúng đạt contract mới hoặc có
benchmark chỉ dựa vào sự tồn tại của file. Các thay đổi source/tests không thuộc bước
speckit-plan này và được giữ nguyên.

## Phases và dependency gates

| Mốc | Việc / file chính | Đầu vào và phụ thuộc | Gate kiểm chứng |
|---|---|---|---|
| M0 | Inventory, dependency profiles, traceability | Code hiện có, spec, design cũ | Ghi phần reuse/gap, không giả định task cũ hoàn tất |
| M1 | schemas_rag, citation, parser | Fixture thuần; reuse normalize_answer | Toàn trạng thái, token/NFC/offset, raw refs; FR-008 provisional |
| M2 | prompt_loader, context_budget, fake LLM, Answerer | M1 | Prompt byte hash, no overflow call, parser theo mode/backend, không gọi mạng |
| M3 | Open/Claude adapters và dev smoke | M2; GPU/endpoint/API khi sẵn sàng | Disable SDK retries; pinned identities; không hứa bitwise API; record smoke |
| M4 | evaluation_rag + rag_run | M1/M2; V1 Evaluation/run infra chặng6 cho tích hợp thật | Metric tính tay,7/5/4 buckets, fingerprint/resume, manifest rỗng |
| M5 | Extractive adapter + pipeline reuse | Reader checkpoint chặng8, Integration chặng9 | Same span/ID/offset/status; 20→5 pool; no V1 artifact overwrite |
| M6 | FastAPI/static | Route fake làm sau M2; nối thật sau M5 | Public/dev isolation, config availability,429,HTTP errors,read-only,375/768/1440 |
| M7 | Dev selection → lock → RQ4 | M3–M5 và V1 data/base/large/candidates đã nghiệm thu | 6+1 configs, fixed prompt/model across tiers, manifest, report/table provenance |

M4 fixture không cần đợi GPU; M6 khung route không cần đợi corpus thật. Không cần web
hoàn tất để chạy batch nghiên cứu; nghiên cứu vẫn là mục đích chính. M3 live calls và
M7 không được chạy tự động từ lệnh unit test hoặc chỉ vì plan được tạo.

## Traceability và nghiệm thu

| Spec | Mốc / Contract / Kiểm tra |
|---|---|
| FR-001/002 | M1,M2,M5; pipeline; oracle7→5, closed_book zero context |
| FR-003/004/005 | M2,M3; parser và config; four-line vs two-line, refusal/overflow |
| FR-006/007/008 | M1; data-model; quote exact/NFC, no partial token, multiple source |
| FR-009 | M1,M5; PredictionRagRecord; Reader unchanged |
| FR-010/011 | M4; pipeline evaluation; arithmetic fixture, complete buckets, N/A |
| FR-012 | M7; selection registry; F1→EM→fixed order |
| FR-013/014 | M6; api contract; unavailable dependency, same-query compare |
| FR-015/016 | M2–M6; fake call counters, duplicate lock, lazy imports |
| FR-017 | M2,M3; context budget; boundary equality, no truncation, estimate failure |
| SC-001..008 | Quickstart validation stages; evidence linked to each milestone |

Các bước nhỏ với tên test và lệnh cụ thể sẽ nằm trong tasks.md; không nhét code mẫu
hoàn chỉnh vào plan và không đánh dấu pass bằng số test dự kiến.

## Complexity Tracking

Không có ngoại lệ constitution cần hợp thức. Thêm RAG schema và runner vì lỗi generation
không đồng nghĩa execution error của V1; dùng schema V1 trực tiếp sẽ nuốt hết các bucket
generation/grounding vào execution_failure. Ranh giới này có regression test.
