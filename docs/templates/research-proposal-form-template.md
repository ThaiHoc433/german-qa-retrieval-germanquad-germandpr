# Research Proposal — Form trống

Template rút từ `Research_Proposal_Corruption_Hierarchical_Learning_Diffusion_LM.pdf`, bỏ hết
nội dung cụ thể của đề tài diffusion model gốc — chỉ giữ cấu trúc mục + câu hỏi gợi ý cho từng
mục + kỷ luật gắn nhãn nguồn gốc. Dùng lại cho bất kỳ đề tài nào, không riêng German QA.

**Cách dùng:** chép file này, điền vào chỗ có `> Điền:`, xóa dòng gợi ý sau khi điền xong. Mỗi
câu khẳng định trong bài viết cuối cùng phải gắn đúng 1 trong 3 nhãn ở mục 8 — không được để
trống nhãn.

---

## 0. Tiêu đề

> Điền: câu hỏi nghiên cứu chính, viết dạng câu hỏi, không viết dạng tên đề tài chung chung.

---

## 1. Research Problem & Motivation

> Điền: bối cảnh chung của hướng nghiên cứu — vì sao hướng này đang được quan tâm, cơ chế/
> vấn đề trung tâm là gì.

> Điền: các nghiên cứu đã có trả lời được gì (nêu ngắn, chi tiết để ở mục 2), và **chưa** trả
> lời được gì — chốt lại bằng một câu hỏi in đậm/blockquote, chính là tiền thân của Research
> Question ở mục 3.

Gợi ý câu hỏi tự kiểm tra mục này:
- Nếu xóa mục 2 (Related Work) đi, người đọc có hiểu được vì sao câu hỏi này quan trọng không?
- Câu hỏi chốt cuối mục có đủ cụ thể để không thể trả lời bằng "có/không" chung chung không?

---

## 2. Related Work & Research Gap

### 2.1 Nền tảng (các công trình đặt nền cho từng thành phần/kỹ thuật đang dùng)

> Điền: liệt kê từng công trình nền tảng, mỗi công trình trả lời 3 ý — nó làm gì, kết quả quan
> trọng nào của nó liên quan đến project này, và vì sao nó KHÔNG (hoặc chưa) trả lời được câu
> hỏi của project.

### 2.2 Công trình liên quan gần đây / quy mô lớn hơn

> Điền: các công trình mới hơn, cùng chủ đề nhưng khác điều kiện (quy mô, domain, metric) —
> nêu để chứng minh chủ đề "đang sống", không phải chủ đề đã cũ/đã giải quyết xong.

### 2.3 Công trình gần nhất với project (setting/phương pháp gần giống nhất)

> Điền: công trình mà project này dựa trên/mở rộng trực tiếp nhất. Nêu rõ: họ đã cố định biến
> gì mà project này sẽ thay đổi, hoặc họ chưa đo được thứ gì mà project sẽ đo.

### 2.4 Research Gap

> Điền: 1 đoạn ngắn nối lại 2.1–2.3 — khẳng định rõ khoảng trống KHÔNG phải "chưa ai nghiên
> cứu chủ đề X" (thường sai, dễ bị bắt bẻ), mà là "chưa ai kết hợp [góc nhìn A từ công trình
> này] với [góc nhìn B từ công trình kia] để trả lời [câu hỏi cụ thể]".

---

## 3. Research Question & Hypothesis

### Research Question

> Điền: 1 câu hỏi, viết dạng "How/What/Liệu... ảnh hưởng/thay đổi thế nào đến...". Tránh câu
> hỏi có thể trả lời bằng một con số duy nhất không cần giải thích.

### Hypothesis

> Điền: giả thuyết H1 — PHẢI nêu ít nhất 2 khả năng (outcome) trái ngược nhau mà experiment có
> thể tìm thấy, không chỉ nêu khả năng mà mình mong đợi đúng. Với mỗi khả năng, ghi rõ nhãn
> nguồn gốc: có bằng chứng gián tiếp nào gợi ý khả năng đó không, hay hoàn toàn suy luận.

Gợi ý tự kiểm tra: nếu hypothesis chỉ có 1 khả năng, viết thêm khả năng ngược lại — một
hypothesis không thể sai thì không phải hypothesis.

---

## 4. Proposed Experiment

### 4.1 Dataset / Experimental Setting

> Điền: dữ liệu/môi trường thực nghiệm dùng gì, vì sao chọn nó (thường: vì ground-truth/label
> có thể biết trước hoặc đo trực tiếp).

### 4.2 Model

> Điền: kiến trúc/pipeline dùng gì; nêu rõ phần nào TÁI SỬ DỤNG nguyên xi từ nơi khác (để cô
> lập biến nghiên cứu) và phần nào là phần mới của project.

### 4.3 Reference / Baseline Condition

> Điền: điều kiện tham chiếu cố định — mọi so sánh khác đối chiếu về điều kiện này. Giải thích
> vì sao chọn nó làm mốc (thường: phổ biến nhất trong literature, hoặc rẻ nhất để chạy).

### 4.4 Experimental Intervention (biến độc lập)

> Điền: biến chính mà project thay đổi có kiểm soát. Nếu biến là liên tục, cân nhắc quét nhiều
> mức thay vì chỉ 2 điểm cực trị (để thấy xu hướng, không chỉ khác biệt).

### 4.5 Controlled Variables (biến kiểm soát)

> Điền dạng bảng 2 cột — "Giữ cố định" / "Thay đổi" — liệt kê MỌI yếu tố có thể gây nhiễu kết
> quả nếu không cố định (kiến trúc, quy trình huấn luyện, budget, seed, quy trình đánh giá...).

| Giữ cố định | Thay đổi |
|---|---|
| | |

---

## 5. Evaluation & Analysis Plan

### 5.1 Metric chính

> Điền: metric tổng quát trả lời câu hỏi "hệ thống làm tốt đến đâu" ở mức cao nhất.

### 5.2 Metric phân giải mịn hơn

> Điền: metric tách theo nhóm/mức/tầng — mục tiêu là không dừng ở 1 con số tổng, mà thấy được
> nó đến từ đâu (level nào, bucket nào, điều kiện nào).

### 5.3 Chỉ số "tốc độ / độ khó"

> Điền: cách đo project cần bao nhiêu (bước huấn luyện / dữ liệu / quy mô) để đạt một ngưỡng
> chất lượng cho trước — cho phép so sánh "cấu hình nào đạt cùng chất lượng rẻ hơn/nhanh hơn".

### 5.4 Error Analysis

> Điền: câu hỏi cụ thể sẽ hỏi khi nhìn vào lỗi — lỗi tập trung ở đâu, có khác nhau giữa các
> điều kiện thực nghiệm không. Không khóa cứng taxonomy lỗi trước khi chạy nếu chưa biết failure
> pattern thực tế trông như thế nào.

### 5.5 Robustness / Ablation

> Điền: biến can thiệp sẽ được quét ở bao nhiêu mức, và mục tiêu của việc quét (đổi trơn theo
> mức? chỉ khác ở 2 đầu? có ngưỡng chuyển pha ở giữa?).

---

## 6. Scope & Feasibility

### In Scope

> Điền: liệt kê gạch đầu dòng những gì project cam kết làm.

### Out of Scope

> Điền: liệt kê tường minh những gì KHÔNG làm trong project này (kể cả thứ nghe hấp dẫn) — mục
> đích là chặn scope creep, không phải liệt kê cho đẹp.

### Feasibility

> Điền dạng bảng — thành phần hạ tầng/công cụ nào đã có sẵn và tái dùng được, thành phần nào
> phải xây mới; điều kiện compute/thời gian thực tế; có cần pilot run trước khi chạy chính thức
> không.

| Thành phần | Khả năng tái sử dụng |
|---|---|
| | |

---

## 7. References

> Điền: từng công trình đã nhắc ở mục 2, đúng định dạng trích dẫn yêu cầu (tên, năm, venue).
> Kiểm tra lại số liệu trích dẫn trước khi nộp — không suy từ trí nhớ mà không xác minh.

---

## 8. Epistemic Status

Chú thích BẮT BUỘC cho mỗi câu khẳng định trong toàn bộ tài liệu (mục 1–7), dùng đúng 1 trong
3 nhãn:

- `[Literature-supported]` — có công trình đã công bố hỗ trợ trực tiếp; nêu công trình nào ở
  mục 7.
- `[Proposed]` — quyết định thiết kế của chính project, xây trên literature nhưng project tự
  chọn (kể cả biến thể `[Proposed — informed by literature]` nếu muốn rõ hơn).
- `[To be tested]` — giả thuyết/claim/con số CHƯA có bằng chứng, sẽ được experiment trả lời;
  không được viết như thể đã đo xong.

> Điền: 1 câu tóm tắt mục tiêu cuối cùng của project — không giả định trước câu trả lời của
> Research Question.
