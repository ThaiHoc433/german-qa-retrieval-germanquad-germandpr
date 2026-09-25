# Feature Specification: RAG Answer Generation with Verifiable Citations

**Feature Branch**: `001-rag-answer-generation`

**Created**: 2026-09-12

**Status**: Draft

**Planning status**: Đủ rõ để lập plan; quy tắc FR-008 vẫn tạm chấp nhận, cần kiểm thử trên fixture/dev trước khi khóa benchmark.

**Source precedence**: Năm clarification và yêu cầu trong feature spec này cập nhật design RAG nguồn ở các điểm đã chốt. Plan Superpowers trước clarification là tham khảo; không dùng nó để ghi đè baseline buckets, oracle5, token-support, context-overflow hoặc thứ tự chọn config ở đây.

**Input**: User description: Chuyển design V2 trong `future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md` thành đặc tả chuẩn cho nhánh sinh câu trả lời tiếng Đức bằng RAG + LLM, citation kiểm chứng được và web local.

## Phạm vi

Feature này mở rộng hệ thống German Open-Domain Extractive QA V1 bằng answerer sinh câu trả lời tiếng Đức có nguồn trích dẫn. V1 vẫn là baseline độc lập: Retriever → Selection → Extractive Reader. V2 thêm `rag`, `closed_book`, `oracle_rag` qua interface Answerer chung, cùng web app local để kiểm thử trực quan.

V2 không sửa module spec V1.1, không thay đổi kết quả V1 và không chặn cổng hoàn tất V1. Các kết quả trong đặc tả này là giao thức và yêu cầu, chưa phải kết quả thực nghiệm.

RQ4: Khi corpus tăng từ base lên large, EM/F1, answer-supported rate và citation accuracy của RAG thay đổi thế nào so với closed-book và oracle giới hạn 5 passage? Bảng 4 giữ chiều rerank bật/tắt và backend; Hình 3 biểu diễn EM/answer-supported rate theo tier. Closed-book và oracle chỉ có một dòng tier=N/A. So sánh model mở với API đồng thời thay model và cơ chế citation, vì vậy không suy ra tác động riêng của cơ chế citation.

## Clarifications

### Session 2026-09-12

- Q: Có phân bucket riêng theo baseline và ghi N/A cho các metric citation của closed-book không? → A: Có. RAG dùng đủ bảy bucket; oracle_rag bỏ qua retrieval/selection; closed_book chỉ dùng execution_failure, generation_failure, answer_failure/correct theo EM. Citation metrics và answer_supported của closed_book không áp dụng.
- Q: Khi có hơn năm gold passage, oracle_rag chọn thế nào? → A: Loại trùng theo passage_id, sắp ID tăng dần rồi lấy tối đa năm; lưu danh sách được chọn và số gold passage bị bỏ. Đây là oracle giới hạn 5 passage, khác oracle Reader V1 đọc toàn bộ gold.
- Q: Khi EM và F1 chọn ra hai cấu hình khác nhau trên dev, ưu tiên metric nào cho đối chứng API? → A: Ưu tiên F1 cao nhất, rồi EM cao nhất; nếu cả hai bằng nhau, chọn cấu hình đứng trước trong thứ tự cố định đã ghi trước khi chạy dev.
- Q: Nếu câu hỏi, prompt và passage vượt giới hạn đầu vào LLM thì xử lý thế nào? → A: Ghi context_overflow, giữ nguyên passage để audit và giữ query trong N_eval; không tự cắt passage. Ngân sách phải dành chỗ cho output và khóa trước test.
- Q: Kiểm tra short_answer trong quote sau chuẩn hóa có bắt buộc khớp nguyên từ không? → A: Tạm chấp nhận khớp chuỗi token liên tiếp, không rỗng, sau normalize_answer của V1. Einstein khớp trong Albert Einstein; Ein không khớp trong Einstein. Đáp án rỗng sau chuẩn hóa không được hỗ trợ. Rà quy tắc bằng fixture/dev trước khi khóa giao thức; không thay đổi hàm EM/F1 V1.

## User Scenarios & Testing

### User Story 1 - Giữ nguyên chế độ extractive (Priority: P1)

Người dùng đặt câu hỏi tiếng Đức và nhận answer span được trích nguyên văn từ passage nguồn, giống hành vi Reader V1.

**Why this priority**: Đây là baseline cần bảo toàn để so sánh mọi answerer mới.

**Independent Test**: Chạy cùng frozen fixture và checkpoint Reader V1; answer span, passage ID, offset và trạng thái phải giống Reader V1.

**Acceptance Scenarios**:

1. **Given** câu hỏi và corpus hợp lệ, **When** chạy extractive answerer, **Then** hệ thống truy hồi tối đa 20 passage, chọn tối đa 5 passage và trả span từ passage nguồn.
2. **Given** cùng input và checkpoint, **When** chạy V1 trực tiếp hoặc qua wrapper Answerer, **Then** các field đầu ra giữ nguyên.

---

### User Story 2 - Sinh câu trả lời tiếng Đức có citation (Priority: P1)

Người dùng nhập câu hỏi tiếng Đức ở chế độ RAG. Hệ thống đưa tối đa năm passage được chọn vào LLM, nhận câu trả lời và citation, rồi kiểm tra citation với passage gốc.

**Why this priority**: Đây là giá trị chính của feature V2.

**Independent Test**: Fake model trả các kịch bản hợp lệ, sai format, không có đáp án, citation sai passage và nhiều citation; kết quả parser/citation phải xác định đúng trạng thái.

**Acceptance Scenarios**:

1. **Given** các passage được đánh số P1..P5, **When** LLM trả output hợp lệ, **Then** hệ thống map số passage về passage ID và trả `short_answer`, `answer`, `quote` cùng trạng thái citation.
2. **Given** quote khớp passage sau exact hoặc NFC/whitespace normalization, **When** citation validator chạy, **Then** citation có offset do code tính và `citation_status=verified`.
3. **Given** model trả `KURZ: KEINE ANTWORT` cùng đúng ba dòng `None`, **When** parser chạy, **Then** `generation_status=success`, `answered=false`, citation fields là null.
4. **Given** model timeout, refusal hoặc sai format, **When** inference đã bắt đầu, **Then** response HTTP 200 có `generation_status=error`, `error_type` chi tiết và không có citation giả.
5. **Given** citation đã verified với quote="Albert Einstein", **When** kiểm tra hỗ trợ đáp án, **Then** short_answer="Einstein" được hỗ trợ, "Ein" không được hỗ trợ; đáp án thành rỗng sau normalize_answer cũng không được hỗ trợ. Khớp token sau chuẩn hóa không tự tạo offset span chính xác nếu chuỗi gốc không khớp.

---

### User Story 3 - Đánh giá và so sánh answerer (Priority: P2)

Nhà nghiên cứu so sánh `closed_book`, `rag`, `oracle_rag` và extractive theo giao thức đã khóa, trên dev trước rồi chạy test chính thức. Resume hoặc rerun sửa lỗi triển khai tuân theo Constitution II: lưu lịch sử, không chọn cấu hình bằng kết quả test.

**Why this priority**: So sánh tách đóng góp của tri thức tham số, retrieval/selection và khả năng sinh câu.

**Independent Test**: Fixture dev tạo được prediction, EM/F1, answer-supported rate, citation accuracy, bucket và latency theo đúng mẫu số.

**Acceptance Scenarios**:

1. **Given** cấu hình Dense với reranker bật/tắt trên base/large, **When** chạy RAG, **Then** có bốn run riêng.
2. **Given** closed-book hoặc oracle passage, **When** chạy baseline, **Then** không nhân bản theo tier và bảng ghi tier=N/A.
3. **Given** model mở đã chọn tốt nhất trên dev, **When** chạy đối chứng API, **Then** chỉ chạy một cấu hình RAG đã khóa và không fallback sang model khác.
4. **Given** closed_book chạy thành công và EM=1, **When** phân bucket, **Then** bucket=correct dù không có passage/citation; citation_status và answer_supported là null, các metric citation hiển thị N/A.
5. **Given** oracle_rag được cấp gold hợp lệ và retrieved=[], **When** phân bucket, **Then** bỏ qua retrieval_miss/selection_failure và xét generation cùng các bước sau.
6. **Given** gold_evidences tham chiếu bảy passage khác nhau và có evidence trùng passage_id, **When** tạo context oracle_rag, **Then** lấy năm ID đầu sau loại trùng và sắp tăng dần, ghi hai passage bị bỏ; đảo thứ tự evidence đầu vào không thay đổi context hoặc nhãn P1..P5.
7. **Given** hai cấu hình dev có thứ hạng F1 và EM ngược nhau, **When** chọn cấu hình đối chứng API, **Then** cấu hình có F1 cao hơn được chọn; chỉ xét EM khi F1 bằng nhau, rồi thứ tự cố định khi cả hai bằng nhau. Dùng giá trị metric đầy đủ, không dùng số đã làm tròn để hiển thị.

---

### User Story 4 - Kiểm thử trực quan local (Priority: P3)

Người dùng mở web local, chọn answerer/cấu hình, gửi câu hỏi, xem câu trả lời, passage đã chọn, quote được tô sáng và citation status.

**Why this priority**: Web giúp kiểm tra pipeline và giải thích kết quả nhưng không thay thế batch research runner.

**Independent Test**: Chạy app trên loopback với fake pipeline; kiểm tra route public/dev, lỗi HTTP, trạng thái và khóa request trùng.

**Acceptance Scenarios**:

1. **Given** câu hỏi tự nhập, **When** chọn oracle_rag, **Then** API trả 422 vì không có gold passage.
2. **Given** tier chưa có artifact thật, **When** gọi route yêu cầu tier đó, **Then** API trả 503 trước khi inference bắt đầu.
3. **Given** cùng query ID và config đang chạy, **When** request thứ hai đến, **Then** API trả 429 `duplicate_inflight_request`.
4. **Given** closed_book, **When** request gửi kèm retriever/tier, **Then** backend ép các field về null/empty theo hợp đồng, không chỉ ẩn ở UI.

### Edge Cases

- Retriever trả ít hơn năm passage; không pad passage giả.
- Oracle có ít hơn năm gold passage khác nhau dùng toàn bộ số đó, không pad; có nhiều hơn năm thì áp dụng FR-002 và báo số bị bỏ. Số bị bỏ đếm theo passage ID khác nhau, không đếm số evidence.
- Quote xuất hiện nhiều lần; lấy lần đầu và lưu số lần xuất hiện.
- Quote ở passage khác với QUELLE; ghi `wrong_passage`, không tự sửa.
- QUELLE ngoài phạm vi hoặc nhiều nguồn; ghi `out_of_range` hoặc `multiple_sources`, giữ raw refs.
- `short_answer` không nằm trong quote dù quote verified; `answer_supported=false` và bucket grounding_failure.
- Model trả dòng thừa; vẫn parse nếu nhãn bắt buộc theo answerer/backend hợp lệ (FR-004), đồng thời ghi `has_extra_text`.
- Context thiếu bằng chứng; RAG phải trả KEINE ANTWORT hoặc trạng thái lỗi hợp lệ.
- Context vượt giới hạn sau khi tính prompt, câu hỏi, title/text, định dạng đóng gói và phần dành cho output: không cắt text, không bỏ passage, không thử lại bằng cấu hình khác; áp dụng FR-017.
- LLM/API timeout, refusal, rate limit hoặc endpoint Colab không phản hồi.
- Citation quote không khớp passage sau cả exact và normalized matching.
- Không để web route truy cập test split hoặc ghi artifact nghiên cứu.

## Requirements

### Functional Requirements

- **FR-001**: Hệ thống MUST hỗ trợ các answerer `extractive`, `closed_book`, `rag`, `oracle_rag` và model backend `open` hoặc `claude` khi answerer không phải extractive.
- **FR-002**: RAG và oracle RAG MUST nhận tối đa năm passage, đánh số P1..Pn với n≤5; model không được thấy passage ID nội bộ. RAG giữ thứ tự selection của pipeline V1. Oracle_rag lấy các passage_id từ gold_evidences, loại trùng ID, sắp chuỗi ID tăng dần theo thứ tự từ điển (không phụ thuộc locale) rồi lấy tối đa năm ID đầu. Lưu oracle_selected_passage_ids theo thứ tự đã cấp và oracle_omitted_passage_count = số ID gold khác nhau trừ số ID đã cấp. Quy tắc áp dụng giống nhau cho batch và web dev, không phụ thuộc tier/retriever. Báo cáo MUST gọi baseline này là oracle giới hạn 5 passage; không đồng nhất nó với oracle Reader V1 đọc toàn bộ gold.
- **FR-003**: Prompt MUST được lưu thành file có version và SHA-256; prompt/config/model revision được ghi trong response và manifest của batch run.
- **FR-004**: Parser MUST xét định dạng theo answerer/backend. Open rag/oracle_rag: KURZ và ANTWORT có giá trị, mỗi nhãn xuất hiện tối đa một lần ở đầu dòng; QUELLE/ZITAT thiếu, rỗng hoặc None ở dạng có đáp án dẫn tới citation_status=missing. Dạng không có đáp án bắt buộc đủ bốn dòng: KURZ: KEINE ANTWORT, ANTWORT: None, QUELLE: None, ZITAT: None. Closed_book ở mọi backend dùng prompt riêng chỉ có KURZ/ANTWORT, dạng không có đáp án là KEINE ANTWORT/None; citation không áp dụng. Claude rag/oracle_rag cũng chỉ trả text KURZ/ANTWORT nhưng dùng citation metadata riêng; wrapper kiểm citation bằng cùng quy tắc như open. Dạng không có đáp án mà vẫn có nguồn/quote có giá trị hoặc API citation là format_violation. Thiếu/rỗng KURZ hoặc ANTWORT, nhãn trùng hoặc format không đọc được cũng là format_violation. Dòng thừa lưu has_extra_text và raw output, không âm thầm mất thông tin.
- **FR-005**: Parser MUST tách generation error khỏi citation status; lỗi model/API/timeout/refusal/format đặt `generation_status=error` và citation fields là null.
- **FR-006**: Citation validator MUST hỗ trợ exact match và NFC + gộp khoảng trắng rồi map ngược offset; MUST NOT dùng fuzzy match.
- **FR-007**: Citation validator MUST ghi verified, wrong_passage, quote_not_found, out_of_range, missing hoặc multiple_sources. Chỉ chấp nhận một citation: nhiều nguồn hoặc nhiều citation object (kể cả trùng nguồn) ghi multiple_sources, citations=[], giữ toàn bộ raw_citation_refs và raw metadata. Không tự chọn, gộp hay sửa nguồn. Khi xác minh đúng một citation, chỉ tìm quote trong các passage đã cấp cho model; quote tìm thấy ở passage khác ghi wrong_passage. API offsets chỉ là audit, code tính offset chính thức. Không có field grounded; answer_supported không kiểm chứng mọi claim của câu answer.
- **FR-008**: Với citation verified, áp dụng quy tắc tạm chấp nhận: chạy normalize_answer của V1 cho short_answer và quote rồi tách theo khoảng trắng; answer_supported=true khi dãy token đáp án không rỗng xuất hiện liên tiếp và nguyên token trong dãy token quote, ngược lại false. Nếu citation không verified hoặc answerer=closed_book thì answer_supported=null. Không thay đổi normalize_answer hoặc cách tính EM/F1 của V1. short_answer_start/end chỉ có khi chuỗi short_answer gốc là chuỗi con chính xác của quote đã xác minh và map được về passage gốc; nếu chỉ khớp qua chuẩn hóa thì offsets là null. Quy tắc hỗ trợ token đang tạm chấp nhận để lập kế hoạch/fixture, chưa phải giao thức thực nghiệm đã khóa.
  - Phép token-support trên áp dụng cho rag/oracle_rag. Adapter extractive xác minh span bằng text/offset V1, quote=short_answer=span và answer_supported=true khi span hợp lệ; không dùng normalization RAG để thay đổi trạng thái hay đáp án Reader V1.
- **FR-009**: Mỗi answerer prediction MUST có `generation_status`, `answered`, citation fields, answer fields, passage provenance và run mode; extractive output không được đổi hành vi V1.
- **FR-010**: Batch evaluation MUST giữ metric V1 EM/F1 và thêm answer-supported rate, citation accuracy, hallucination proxy, format violation rate và latency theo mẫu số đã đặc tả. Với closed_book, citation_status/answer_supported là null; answer-supported rate, citation accuracy và hallucination proxy ở mức citation báo N/A, không báo 0 và không gộp vào mẫu số của rag/oracle_rag.
  - N_eval là các query có gold_evidences khác rỗng trong cùng file QA; mọi metric sau chỉ xét tập này. EM/F1 chấm short_answer, lấy max trên reference_answers như V1, trung bình trên N_eval. Query thiếu prediction, execution/generation error hoặc answered=false nhận EM=F1=0; không bỏ query lỗi khỏi mẫu số.
  - Answer-supported rate = số answer_supported=true / số answered=true.
  - Citation accuracy = số query có citation_status=verified và passage_id của citation thuộc tập ID gold_evidences / số answered=true. Không đòi EM=1; đây là đúng nguồn theo nhãn gold, không phải đánh giá toàn bộ câu sinh.
  - Hallucination proxy = số answered=true có citation_status trong {wrong_passage, quote_not_found} / số answered=true.
  - Format violation rate = số generation_status=error với error_type=format_violation / N_eval. N_eval khác số lần gọi generate; context_overflow cũng ở N_eval.
  - Latency chỉ tổng hợp các record pipeline thành công, generation_status=success và có số đo hữu hạn không âm; báo n_latency riêng cho từng bước, mean/p50/p95 nội suy tuyến tính V1. Closed_book không có verify_citation nên bước đó N/A. Web/API wall time và GPU time được gắn môi trường đo, không trộn chung thành một phân phối.
  - Mọi tỷ lệ lưu tử số/mẫu số; mẫu số 0 cho giá trị null và hiển thị N/A. Bucket counts cộng đúng N_eval theo answerer; gold rỗng đếm riêng.
- **FR-011**: Bucket MUST được xét theo answerer, dừng tại điều kiện đầu tiên khớp; `final_correct` (EM=1) lưu độc lập. Extractive giữ nguyên năm bucket V1. Các thứ tự và điều kiện cho LLM như sau:
  - `rag`: execution_failure → retrieval_miss → selection_failure → generation_failure → grounding_failure → answer_failure → correct.
  - `oracle_rag`: execution_failure → generation_failure → grounding_failure → answer_failure → correct; không xét retrieval/selection vì gold được cấp trực tiếp.
  - `closed_book`: execution_failure → generation_failure → answer_failure → correct; answer_failure khi answered=false hoặc EM<1, correct khi answered=true và EM=1. Không xét retrieval, selection hay citation.
  - `generation_failure`: generation_status=error sau khi vượt các bước trước. Với rag/oracle_rag, grounding_failure khi answered=true và (citation_status khác verified hoặc answer_supported=false); answer_failure khi answered=false hoặc (answer_supported=true và EM<1); correct khi answer_supported=true và EM=1.
- **FR-012**: Thực nghiệm RQ4 MUST dùng sáu run model mở: bốn RAG Dense×rerank×tier, một closed-book N/A và một oracle RAG N/A; API chỉ thêm một RAG config. Chọn trong bốn cấu hình RAG của model mở theo F1 trung bình trên N_eval dev cao nhất, rồi EM trung bình cao nhất; nếu cả hai bằng nhau, chọn cấu hình đứng trước trong danh sách thứ tự cố định được ghi trước khi chạy dev. So sánh metric chưa làm tròn; danh sách thứ tự và kết quả lựa chọn MUST được lưu cùng cấu hình đã khóa trước test. Không chọn lại bằng kết quả API hoặc test.
- **FR-013**: Web MUST bind 127.0.0.1, tách route public/dev, không ghi manifest/corpus/checkpoint. 422 dành cho input sai (tier ngoài enum, câu hỏi rỗng, answerer/backend không hợp lệ, oracle trên ask); tier hợp lệ nhưng chưa build hoặc tài nguyên cần thiết chưa sẵn sàng trả 503. Pipeline đã bắt đầu thì lỗi generation/format/context_overflow trả 200 kèm trạng thái; lỗi retrieval/selection runtime giữ riêng là execution failure, không gán generation error. Request trùng query_id và effective config đang chạy trả 429. Dev routes chỉ tra query_id trong file dev đã khóa; không suy split từ ID và không nhận đường dẫn dữ liệu/split do request cung cấp. Compare phải dùng cùng một câu hỏi hoặc cùng query_id dev cho hai cấu hình.
- **FR-014**: Tier chỉ khả dụng khi corpus/manifest tồn tại và fingerprint khớp; availability của cấu hình còn phụ thuộc artifact cần dùng: BM25 cần BM25 index, Dense cần encoder/FAISS/ID map, rerank bật cần reranker, extractive cần Reader, LLM cần backend/prompt. Thiếu Dense không làm BM25 unavailable; thiếu reranker không chặn rerank tắt. Biến môi trường chỉ thu hẹp allowlist. Model/checkpoint dùng chung giữa các tier theo cùng fingerprint, không nạp bản sao cho từng tier.
- **FR-015**: Research run MUST không retry hoặc fallback model; web V2 cũng không tự retry. Người dùng thử lại bằng request mới.
- **FR-016**: Test suite MUST không gọi mạng thật; LLM client phải có fake client để kiểm tra parser, citation, answerer, bucket và route.
- **FR-017**: Trước khi gửi yêu cầu sinh câu, hệ thống MUST kiểm tra độ dài đầu vào theo backend/tokenizer đã khóa, bao gồm toàn bộ prompt, câu hỏi, title/text và định dạng đóng gói. Tổng token đầu vào cộng ngân sách output MUST không vượt giới hạn context đã ghi trong cấu hình. Nếu vượt, ghi generation_status=error, error_type=context_overflow, answered/citation_status/answer_supported=null; không gọi generate, không truncate hoặc bỏ passage. Giữ danh sách passage và thông tin giới hạn/độ dài để audit. Query vẫn thuộc N_eval, EM/F1=0; gán bucket theo ưu tiên FR-011 (generation_failure nếu không có lỗi ở bước trước). Không đưa query này vào n_latency thành công. Với web, đây là lỗi của bước chuẩn bị generation trong pipeline đang xử lý, trả HTTP 200 kèm trạng thái; không nhầm với HTTP 503 do tài nguyên chưa sẵn sàng. Ngân sách context/output và cách đếm MUST được khóa trước test. Fixture phải kiểm tra vừa đủ giới hạn thì được gửi, vượt một token thì không gọi client và passage vẫn nguyên vẹn.

### Key Entities

- **Answerer**: loại bộ trả lời và backend model được chọn.
- **Passage Context**: tối đa năm passage có title, text, thứ tự Pk và ID nội bộ.
- **Oracle Context Audit**: oracle_selected_passage_ids theo thứ tự cấp cho model và oracle_omitted_passage_count tính trên gold passage ID khác nhau.
- **Generated Prediction**: short answer, câu trả lời, generation status, citation status và answer-supported flag.
- **Citation Audit**: passage ID, quote, offsets, match method, occurrence count và raw citation refs.
- **Research Run**: cấu hình, prompt/model fingerprints, corpus tier, predictions, metrics và latency.
- **Web Request**: câu hỏi, answerer, cấu hình hợp lệ, run mode và response lỗi/thành công.

## Success Criteria

### Measurable Outcomes

- **SC-001**: 100% prediction RAG hợp lệ truy ngược được tới query, passage context, prompt revision và model backend.
- **SC-002**: 100% citation trong fixture được phân loại đúng giữa verified, wrong_passage, quote_not_found, out_of_range, missing và multiple_sources.
- **SC-003**: 100% lỗi generation không tạo citation hoặc answer-supported flag không hợp lệ.
- **SC-004**: Reader V1 qua frozen fixture giữ nguyên answer span, passage ID, offset và trạng thái khi bọc trong Answerer.
- **SC-005**: Bộ test local không gọi mạng thật và bao phủ bảy bucket của rag, năm bucket áp dụng cho oracle_rag và bốn bucket của closed_book; gồm verified nhưng answer_supported=false, closed_book trả lời đúng không có citation, và oracle_rag có retrieved=[] không bị gán retrieval_miss. Mỗi query thuộc N_eval nhận đúng một bucket theo answerer.
- **SC-006**: RQ4 không chạy test trước khi prompt/model/config được khóa trên dev và không có fallback làm trộn model trong một run.
- **SC-007**: Web local phân biệt rõ public/dev-only answerer, citation status, quote và short answer; route invalid trả đúng HTTP contract.
- **SC-008**: Báo cáo RQ4 không nhân bản closed_book/oracle_rag theo tier và không trình bày kết quả thực nghiệm chưa chạy.

## Assumptions

- Module spec V1.1 frozen là nguồn chuẩn cho Retriever, Selection, Reader, Evaluation và run discipline; design V2 không sửa nó.
- Tài liệu design nguồn ở future_RAG/; artifact Spec Kit của feature ở specify/001-rag-answer-generation/. Code dự kiến đặt trong src/, app/, scripts/, tests/. Chỉ nối thật sau khi V1 có candidate top-20, corpus tier, checkpoint và run infrastructure hợp lệ.
- Model mở được chạy batch trên Colab T4 hoặc phục vụ qua endpoint; Claude API là nhánh đối chứng, không phải nguồn citation tuyệt đối.
- Tên model cụ thể, prompt cuối cùng, lượng tử hóa và context budget được chọn trên dev rồi khóa trước test.
- GermanQuAD/GermanDPR không cung cấp nhãn claim-level hallucination đầy đủ; `answer_supported` chỉ là kiểm chứng short answer/citation.
- Web chỉ chạy local, không deploy public, không authentication, không lưu lịch sử và không thay thế batch runner nghiên cứu.
- Startup API model identity được cấu hình trước dev; greedy không hứa bit-for-bit giữa hardware/backend. Token counter API có thể là ước lượng: margin và count_method khóa trước run; provider báo vượt context vẫn là context_overflow, không tự retry.
