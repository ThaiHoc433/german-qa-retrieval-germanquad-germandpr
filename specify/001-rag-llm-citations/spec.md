# Feature Specification: Grounded RAG Answer Generation with Citations

**Feature Branch**: `001-rag-llm-citations`

**Created**: 2026-09-12

**Status**: Draft

**Input**: User description: Mô tả tổng thể kiến trúc QA truy hồi đoạn văn tiếng Đức hiện tại và kiến trúc mở rộng khi thêm RAG + LLM để sinh câu trả lời kèm nguồn trích dẫn.

## Mục tiêu và phạm vi

Project hiện tại là hệ thống open-domain extractive QA tiếng Đức: người dùng nhập câu hỏi không kèm ngữ cảnh, hệ thống tìm passage trong corpus rồi trích một answer span nguyên văn. Feature này mô tả kiến trúc mở rộng có thêm một nhánh grounded RAG: sau khi passage được truy hồi và sắp xếp, một mô hình ngôn ngữ sinh câu trả lời tự nhiên nhưng chỉ được sử dụng bằng chứng trong các passage đã cung cấp, đồng thời trả về nguồn trích dẫn.

Extractive Reader vẫn là baseline của V1. Nhánh RAG + LLM là phần mở rộng V2 và không được làm thay đổi giao thức, kết quả hoặc acceptance criteria của V1.

## User Scenarios & Testing

### User Story 1 - Truy vấn extractive baseline (Priority: P1)

Người dùng nhập một câu hỏi tiếng Đức. Hệ thống tìm các passage liên quan, chọn tối đa năm passage và trích chính xác cụm từ trả lời từ một passage nguồn.

**Why this priority**: Đây là chức năng cốt lõi đã được đặc tả và là mốc so sánh bắt buộc cho mọi mở rộng.

**Independent Test**: Với một câu hỏi có gold passage và answer span, hệ thống trả về answer text, passage nguồn và vị trí span; kết quả có thể chấm bằng Exact Match và token-level F1.

**Acceptance Scenarios**:

1. **Given** câu hỏi và corpus hợp lệ, **When** người dùng gửi câu hỏi, **Then** hệ thống trả về tối đa 20 passage truy hồi, tối đa 5 passage được đưa cho Reader và answer span nếu tìm được.
2. **Given** passage chứa gold answer, **When** Reader chọn đúng span, **Then** answer text phải khớp đoạn văn nguồn theo offset và prediction ghi được passage ID.
3. **Given** không có passage hợp lệ hoặc một query gặp lỗi runtime, **When** pipeline xử lý query, **Then** hệ thống giữ prediction có trạng thái lỗi tương ứng và không âm thầm biến lỗi thành câu trả lời.

---

### User Story 2 - Sinh câu trả lời có dẫn nguồn (Priority: P2)

Người dùng gửi cùng loại câu hỏi tiếng Đức ở chế độ grounded RAG. Hệ thống dùng các passage đã chọn làm ngữ cảnh cho LLM, sinh câu trả lời dễ đọc và gắn mỗi khẳng định với một hoặc nhiều passage nguồn.

**Why this priority**: Chức năng này tăng tính tự nhiên và khả năng giải thích cho người dùng nhưng phụ thuộc vào retrieval và baseline đã ổn định.

**Independent Test**: Với fixture gồm câu hỏi, hai passage được đánh số và câu trả lời mong đợi, kiểm tra output có answer text, citation hợp lệ trỏ tới passage tồn tại và không có citation ngoài context được cấp.

**Acceptance Scenarios**:

1. **Given** top-5 passages có bằng chứng, **When** RAG mode chạy, **Then** output có câu trả lời sinh ra và danh sách citation chứa passage ID, thứ hạng và trích đoạn hỗ trợ.
2. **Given** các passage không đủ bằng chứng, **When** LLM không thể trả lời có căn cứ, **Then** hệ thống trả về trạng thái không đủ bằng chứng hoặc yêu cầu làm rõ, không tự bịa thông tin.
3. **Given** LLM trả lời nhưng citation không trỏ tới context hoặc không hỗ trợ câu trả lời, **When** lớp kiểm tra groundedness chạy, **Then** prediction bị đánh dấu không grounded và vẫn lưu để audit.
4. **Given** cùng question, corpus, retriever và top-5 passages, **When** chạy extractive mode và RAG mode, **Then** hai kết quả được lưu riêng nhưng truy vết được về cùng retrieval run.

---

### User Story 3 - So sánh hai chế độ và truy vết nguồn (Priority: P3)

Nhà nghiên cứu xem được sự khác nhau giữa answer span nguyên văn và câu trả lời sinh ra, cùng passage đã hỗ trợ mỗi kết quả.

**Why this priority**: So sánh giúp đánh giá RAG có tạo thêm giá trị hay chỉ làm tăng độ trễ và rủi ro hallucination.

**Independent Test**: Với một bộ prediction có cả hai mode, tạo được bảng đối chiếu theo query, correctness, groundedness, citation coverage và latency.

**Acceptance Scenarios**:

1. **Given** prediction của hai mode cho cùng query, **When** mở artifact so sánh, **Then** người xem phân biệt được answer nguyên văn, answer sinh và citation tương ứng.
2. **Given** một citation trỏ tới passage bị loại hoặc không tồn tại, **When** validator chạy, **Then** run bị đánh dấu không hợp lệ thay vì sửa citation tự động.

### Edge Cases

- Retriever trả về ít hơn năm passage; RAG không tự điền passage giả.
- Một câu trả lời cần bằng chứng từ nhiều passage; citation có thể chứa nhiều nguồn và phải ghi thứ tự.
- Passage có nội dung mâu thuẫn; hệ thống đánh dấu conflict hoặc uncertainty thay vì chọn nguồn không có căn cứ.
- LLM trả lời bằng tiếng Đức khác câu chữ gold nhưng vẫn được passage hỗ trợ; không dùng Exact Match extractive làm tiêu chí duy nhất.
- Context vượt ngân sách đầu vào; phải ghi rõ passage bị loại và lý do, không cắt giữa citation mà không lưu offset.
- LLM timeout, model unavailable hoặc output không parse được; lưu lỗi generation và giữ artifact retrieval đã hoàn thành.
- Citation quote không khớp nội dung passage theo offset; đánh dấu citation invalid.
- Query thuộc test hoặc dữ liệu có leakage; bị chặn theo constitution và spec V1.1.

## Requirements

### Functional Requirements

- **FR-001**: Hệ thống MUST duy trì extractive pipeline hiện tại gồm Question → Retriever → selection/reranker → Extractive Reader → answer span.
- **FR-002**: Hệ thống MUST hỗ trợ hai mode độc lập: `extractive` và `grounded_rag`.
- **FR-003**: RAG mode MUST nhận cùng câu hỏi và tập passage đã được retrieval/selection xác định; generator MUST NOT tự truy cập corpus ngoài context được ghi trong run.
- **FR-004**: RAG output MUST có answer text, generation status, danh sách citation và liên kết tới query, retrieval run, passage ID và passage rank.
- **FR-005**: Mỗi citation MUST xác định passage nguồn và một quote hoặc offset có thể kiểm tra lại trong passage gốc.
- **FR-006**: Hệ thống MUST đánh dấu `grounded=true` chỉ khi mọi citation hợp lệ và câu trả lời có bằng chứng tương ứng theo quy tắc kiểm tra đã khóa trước run.
- **FR-007**: Khi thiếu bằng chứng, citation không hợp lệ, timeout hoặc output không parse được, hệ thống MUST trả trạng thái lỗi/không đủ bằng chứng và MUST NOT bịa citation.
- **FR-008**: Extractive và RAG predictions MUST được lưu riêng, có cùng `question_key`/query ID để so sánh, không ghi đè artifact V1.
- **FR-009**: Mọi RAG run MUST truy vết được tới corpus tier, retrieval/selection configuration, LLM/model revision, prompt revision, seed nếu có, context IDs và manifest fingerprint.
- **FR-010**: Hệ thống MUST giữ nguyên các metric V1 cho retrieval và extractive QA; RAG phải bổ sung metric riêng cho answer correctness, citation validity, groundedness, unsupported-claim rate và generation latency.
- **FR-011**: RAG mode MUST ghi nhận latency generation và total latency với cùng quy tắc warm-up, batch và n_latency đã khóa cho pipeline; không trộn latency lỗi vào mẫu thành công.
- **FR-012**: Không được dùng GermanQuAD/GermanDPR test để chọn prompt, model, context budget hoặc ngưỡng groundedness; mọi lựa chọn phải khóa trên dev trước test.
- **FR-013**: Feature MUST hoạt động như mở rộng V2; việc bật RAG không được thay đổi thứ tự, output schema hoặc acceptance của extractive V1.
- **FR-014**: Tài liệu và giao diện MUST gọi đúng đây là grounded RAG/extractive QA, không mô tả hệ thống như chatbot có thể trả lời tùy ý.

### Kiến trúc logic

#### Kiến trúc hiện tại — V1 extractive

```text
Question tiếng Đức
  → BM25 hoặc Dense Retriever
  → retrieval pool tối đa 20 passage
  → Cross-Encoder Reranker hoặc lấy trực tiếp top-5
  → Extractive Reader
  → answer span + passage nguồn + offsets
```

BM25 là nhánh lexical baseline. Dense Retriever biểu diễn question và passage rồi tìm kiếm tương đồng; vector index chỉ phục vụ nhánh dense. Reader không tạo nội dung mới mà chọn start/end của span trong passage.

#### Kiến trúc mở rộng — V2 grounded RAG + LLM

```text
Question tiếng Đức
  → BM25 hoặc Dense Retriever
  → retrieval pool tối đa 20 passage
  → Reranker / direct top-5 selection
  → Context packer giữ passage ID, rank và offset
  → LLM Generator chỉ đọc context đã đóng gói
  → Answer text + citations + groundedness/status
```

Nhánh RAG dùng lại retrieval và selection để giữ cùng điều kiện bằng chứng. LLM chỉ diễn đạt hoặc tổng hợp từ context đã cấp; lớp citation validator đối chiếu từng nguồn với passage đã lưu. Extractive Reader và LLM Generator là hai consumer độc lập của top-5, không thay thế lẫn nhau trong cùng một prediction.

### Key Entities

- **Question**: câu hỏi tiếng Đức, question key, dataset/split nguồn và trạng thái đánh giá.
- **Passage**: đoạn văn đã chunk, passage ID, corpus tier, nội dung, parent document và provenance.
- **Retrieval Run**: kết quả top-k, retriever, thứ hạng, score, corpus fingerprint và run ID.
- **Selection Context**: tập tối đa năm passage được đưa cho Reader hoặc Generator, giữ rank và ID gốc.
- **Extractive Prediction**: answer span, answer text, passage ID, offsets, EM/F1 và trạng thái bucket.
- **Generated Prediction**: answer text do LLM sinh, generation status, model/prompt revision và latency.
- **Citation**: passage ID, rank, quote/offset, claim liên quan và trạng thái valid/invalid.
- **Evaluation Run Manifest**: cấu hình, fingerprint, seed, input artifacts, checkpoint bundle và điều kiện đo.

## Success Criteria

### Measurable Outcomes

- **SC-001**: 100% prediction ở cả hai mode có thể truy ngược từ query tới passage, corpus tier và run manifest; citation không được trỏ tới passage ngoài artifact của run.
- **SC-002**: Trên fixture kiểm thử, 100% citation hợp lệ được xác minh bằng passage ID và quote/offset; citation sai bị phát hiện mà không bị sửa âm thầm.
- **SC-003**: Khi chạy cùng retrieval inputs, extractive mode vẫn tạo được toàn bộ metric V1 mà không bị ảnh hưởng bởi việc bật hoặc tắt RAG mode.
- **SC-004**: 100% trường hợp thiếu bằng chứng, timeout hoặc output parse lỗi được biểu diễn bằng status rõ ràng; không có câu trả lời/citation giả được tính là đúng.
- **SC-005**: Báo cáo so sánh tách riêng answer correctness, citation validity, groundedness, unsupported-claim rate và generation latency; không dùng một điểm tổng hợp chưa được đặc tả.
- **SC-006**: Mọi checkpoint/model/prompt/config dùng để chạy test được khóa trước khi test và có fingerprint trong manifest.
- **SC-007**: Với cùng input và cấu hình deterministic đã chọn, rerun trên fixture tạo cùng citation IDs, rank và trạng thái; khác biệt do sampling phải được ghi seed và phạm vi sai khác.
- **SC-008**: Người dùng có thể phân biệt rõ trong output đâu là câu trả lời trích nguyên văn, đâu là câu trả lời sinh và đâu là nguồn hỗ trợ.

## Assumptions

- Feature được phát triển sau khi pipeline V1 đạt các gate dữ liệu, retrieval, Reader và manifest của module spec V1.1.
- V1.1 vẫn là nguồn chuẩn cho extractive pipeline; feature này cần amendment riêng trước production run RAG.
- Một generator có khả năng xử lý tiếng Đức sẽ được chọn trên dev; tên model cụ thể, prompt template và context budget chưa được khóa trong feature spec này.
- RAG không truy cập web hoặc corpus ngoài các passage đã được manifest ghi nhận.
- GermanQuAD/GermanDPR không đủ để tự chứng minh chất lượng citation cho mọi loại câu hỏi; báo cáo phải nêu giới hạn domain và nhãn.
- Câu trả lời sinh có thể diễn đạt khác gold extractive; đánh giá RAG cần human review hoặc bộ tiêu chí semantic đã định nghĩa trước.
- V2 có thể dùng cùng corpus tier base/mid/large để phân tích ảnh hưởng quy mô, nhưng không được thay đổi kết quả V1 đã đóng băng.
- Việc triển khai API, giao diện web hoặc chatbot tương tác không nằm trong feature này; feature tập trung vào pipeline và artifact có thể tái lập.
