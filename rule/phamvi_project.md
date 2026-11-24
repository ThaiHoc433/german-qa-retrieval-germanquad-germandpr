# Phạm vi bài toán 

# Hệ thống làm
Nhận 1 câu hỏi tiếng ĐỨC, tìm đoạn văn liên quan trong một kho đoạn văn xác định, trích một đoạn chữ có sẵn trong đó làm đáp án và trả kèm bằng chứng 
(Đoạn nguồn, vị trí đáp án, điểm số, thời gian từng bước)

# Dữ liệu và vai trò 
- GermanQuAD: huấn luyện và đánh giá READER; đánh giá QA end-to-end
- GermanDPR: đoạn văn và hard negative cho retrieval (dense ở phần B)

# Kho đoạn văn (corpus)
Context GermanQuAD + positive + hard negative GermanDPR, đã tiền xử lý 
Đây là kho giới hạn, ko phải toàn bộ wikipedia. Số đoạn chính xác ghi ở GĐ2

# Không thể thực hiện
- Không trả lời câu hỏi ngoài kho: READER luôn trả 1 span; chưa có ngưỡng -> " 0 có ĐÁP ÁN "
- Không sinh câu trả lời mới: Chỉ trích từ đoạn
- Không chấm EM/F1 cho câu hỏi mới vì 0 có "ĐÁP ÁN CHUẨN"

# Giao thức đánh giá 
TRAIN: để học  
dev: chọn cấu hình 
test: chỉ chạy 1 lần sau khi đóng băng mọi lựa chọn 

## 
- Tập test: Chỉ GermanQuAD test, GermanDPR chỉ góp đoạn văn + hard negative 

hard negative: là những đoạn văn 0 chứa câu, nhưng có nội dung liên quan -> mô hình dễ nhầm lẫn 

positive passage: Đoạn văn đúng, có chưa thông tin trả lời câu hỏi 

# GERMANQuAD
-question: "câu hỏi"
-context: đoạn văn chứa thông tin 
-answers: đáp án 

# GermanDPR
- question: câu hỏi 
- answers: đáp án 
- positive_ctxs: đoạn văn đúng 
- hard_negative_ctxs: các đoạn văn sai nhưng có liên quan 
- negative_ctxs: các đoạn âm tính thông thường  (nếu có)

# 