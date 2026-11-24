
# Quy tắc tiền xử lý (ghi trước khi chạy)
R1  Offset sai, đáp án xuất hiện đúng 1 lần   -> sửa bằng tìm chuỗi, ghi span_fix_log.jsonl
R2  Đáp án xuất hiện nhiều lần / không thấy   -> cách ly vào quarantine.jsonl, không đoán
R3  Mẫu train trùng câu hỏi/passage với test  -> loại khỏi train, không bao giờ loại khỏi test
R4  Hai passage giống hệt văn bản             -> gộp 1, giữ đủ provenance trong source[]
R5  Tỷ lệ train/dev                           -> 90/10, chia theo nhóm liên thông
R6  Passage DPR vào corpus                    -> có: positive + hard_negative
R7  Seed                                      -> 42
R8  Câu hỏi trùng văn bản trong cùng split    -> gộp đáp án nếu cùng passage;
                                                 khác passage thì chỉ hợp nhất vào qrels
R9  GermanDPR không có offset                 -> tìm đáp án trong positive passage;
                                                 duy nhất -> lấy vị trí; mơ hồ -> start=None (không cách ly)