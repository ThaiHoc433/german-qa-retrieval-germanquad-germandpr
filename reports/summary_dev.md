### Kết quả tập dev

| Tầng | Chấm theo | EM [KTC 95%] | F1 [KTC 95%] |
|---|---|---|---|
| Reader, đoạn đúng cho trước | tất cả tham chiếu | 0,469 [0,441; 0,497] | 0,699 [0,679; 0,719] |
| Reader, đoạn đúng cho trước | tham chiếu đầu tiên | 0,469 [0,441; 0,497] | 0,699 [0,679; 0,719] |
| End-to-end | tất cả tham chiếu | 0,236 [0,213; 0,261] | 0,380 [0,359; 0,404] |
| End-to-end | tham chiếu đầu tiên | 0,236 [0,213; 0,261] | 0,380 [0,359; 0,404] |

| Retrieval | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | 0,322 | 0,638 | 0,752 | 0,814 | 0,452 |

| Thời gian (ms) | Trung bình | p50 | p95 |
|---|---|---|---|
| Truy hồi | 0,8 | 0,7 | 1,4 |
| Reader (5 đoạn) | 57,9 | 56,4 | 79,8 |
| Tổng | 58,9 | 57,5 | 80,7 |
