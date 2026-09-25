# Pipeline và evaluation contracts

[Spec](../spec.md) / [Data model](../data-model.md). Tên interface dưới đây là mục tiêu
implementation; không khẳng định các hàm đã tồn tại.

## Boundaries

| Interface | Input → Output | Side effect được phép |
|---|---|---|
| parse_llm_output | raw text, answerer, backend → ParsedAnswer/error | Không |
| verify_citation | selected passages, parsed answer, raw citation metadata → CitationOutcome | Không |
| build_context | question, selected, prompt config → payload + ID map + hash | Không ghi file |
| check_context_budget | payload, backend token counter, locked budget → budget audit/error | Count API nếu backend Claude; không generate |
| LLMClient.generate | prepared payload, locked config → raw text/blocks + response metadata | Đúng một generation attempt, không retry |
| Answerer.answer | question, selected, effective config → AnswerResult | Inject client; giữ raw errors |
| RagPipeline.run | RequestContext, config → PredictionRagRecord | Đo timing, gọi module, không ghi manifest |
| BatchRunner.run | manifest/config, QA records → predictions artifact | Manifest trước inference, append duy nhất/query |
| evaluate_rag | gold, manifest/run_id, predictions → metrics | Không gọi model hoặc tự đo timing |

Schemas validators được dùng ở ingress/egress; module thuần không import SDK. SDK
imports nằm trong factory, test inject fake. Gold chỉ đưa vào oracle context hoặc
Evaluation, không đưa vào prompt rag/closed_book.

## Parser grammar và ưu tiên citation

Open with-context có KURZ/ANTWORT và QUELLE/ZITAT. Closed-book và Claude text chỉ có
KURZ/ANTWORT. Nhãn nhận ở đầu dòng, không phân biệt biến thể nhãn; nhãn sai case là
không phải nhãn yêu cầu. Không dùng JSON hay LLM repair. Backend wrapper của Claude
ghép text blocks theo thứ tự giữ nguyên nội dung, không tự thêm newline; metadata
citations lấy từ toàn bộ response. Trích đoạn chứa xuống dòng trong plain-text mode
không phải một value hợp lệ theo grammar một dòng; có thể khớp whitespace-normalized
nếu model xuất một dòng đúng grammar.

Dạng có đáp án: KURZ/ANTWORT không rỗng/None. Open citation fields bị thiếu vẫn parse,
verifier cho missing. No-answer: đúng KEINE ANTWORT + None theo các field bắt buộc
của mode; citation có giá trị thì format_violation. Closed_book có thêm citation text
ở dạng answered=true giữ raw/has_extra_text, không tính citation metric.

Sau parse thành công và answered=true ở rag/oracle:
1. Nhiều nguồn/objects → multiple_sources, giữ mọi raw refs, không dedup chọn hộ.
2. Thiếu ref hoặc quote → missing.
3. Ref một nguồn không map được integer trong1..n → out_of_range; ref hỏng cú pháp
   không đọc được là format_violation từ parser, không fuzzy parse.
4. Quote rỗng sau trim → missing. Tìm exact trước, normalized sau trong đúng passage.
5. Không thấy đúng passage nhưng thấy ở selected khác → wrong_passage; không thấy
   ở selected nào → quote_not_found. Không search toàn corpus để sửa citation.
6. Match đúng passage → verified; tính token support riêng và offsets đúng text gốc.

Reader là adapter riêng không đi qua parser LLM. Runtime error read giữ nguyên V1.
Citation verification bug thuộc execution_status=error/error_stage=verify_citation,
khác quote không tìm thấy (một kết quả citation lỗi hợp lệ).

## Evaluation arithmetic fixture (giả lập, không phải kết quả nghiên cứu)

Tám query đủ gold của một run rag:

| Case | Tình trạng | Bucket | EM |
|---|---|---|---:|
| A | Thiếu record | execution_failure | 0 |
| B | Mất gold top20, đoán đúng, quote sai | retrieval_miss | 1 |
| C | Có gold top20 nhưng rớt top5, không trả lời | selection_failure | 0 |
| D | Có gold selected, timeout | generation_failure | 0 |
| E | Quote verified đúng gold, short answer không được quote hỗ trợ | grounding_failure | 0 |
| F | Quote verified đúng gold, short answer hỗ trợ nhưng khác reference | answer_failure | 0 |
| G | Quote verified đúng gold, hỗ trợ và EM=1 | correct | 1 |
| H | Có gold selected, KEINE ANTWORT | answer_failure | 0 |

N_eval=8; answered=true có B,E,F,G (4); answer-supported=2/4; citation accuracy=3/4;
hallucination proxy=1/4; EM=2/8. Đây là expected arithmetic fixture để phát hiện lỗi
mẫu số/bucket; test F1 dùng reference strings tính tay riêng. Thêm case format error
và empty gold/zero denominator/closed-book/oracle, không ép bộ test đúng20 dòng.

## Research protocol

Batch1; warmup một lượt10 dev query_id sorted, loại kết quả warmup chứ không loại
query khỏi lượt chính thức. Mỗi query chính thức một lần; GPU sync trước/sau đoạn GPU,
CPU verifier dùng monotonic clock. API/endpoint wall latency có network, không báo
là GPU inference-only. Record generation errors không góp mẫu latency success.

RAG có thể reuse candidate/selection artifacts để đối chiếu cùng evidence. Khi reuse,
retrieval/rerank latency trong run này là null, provenance trỏ số đo V1 riêng;
total là wall time của đường chạy thực tế, gắn measurement_mode=reused_candidates.
Không cộng p95 các stage thành p95 total. Muốn so full end-to-end latency phải chạy
fresh pipeline đồng nhất và kiểm output candidate khớp frozen artifact; không gọi
lại LLM chỉ để thay số đo trong cùng record.

Development selection: model/prompt/budget chọn trên dev, tối đa3 bộ prompt version;
giữ model/prompt/quantization cố định qua các tier/baseline khi chạy test. Oracle và
rag dùng cùng with-context prompt, closed-book prompt riêng có hash. Chọn cấu hình
cho API theo F1,EM,index đã lưu. Lưu support_rule_revision=token_contiguous_provisional
trong fixture/dev; benchmark lock cần ghi nhận quyết định sau review rule.

V1 checkpoint/data gate chưa qua thì chỉ chạy fake test. Không ghi trạng thái
benchmark-ready chỉ vì parser/client test pass.
