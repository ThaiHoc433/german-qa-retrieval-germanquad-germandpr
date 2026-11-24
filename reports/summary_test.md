### Kết quả tập test

| Tầng | Chấm theo | EM [KTC 95%] | F1 [KTC 95%] |
|---|---|---|---|
| Reader, đoạn đúng cho trước | tất cả tham chiếu | 0,583 [0,561; 0,605] | 0,754 [0,738; 0,769] |
| Reader, đoạn đúng cho trước | tham chiếu đầu tiên | 0,416 [0,396; 0,437] | 0,666 [0,651; 0,682] |
| End-to-end | tất cả tham chiếu | 0,330 [0,310; 0,352] | 0,456 [0,438; 0,476] |
| End-to-end | tham chiếu đầu tiên | 0,222 [0,206; 0,241] | 0,397 [0,380; 0,416] |

| Retrieval | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | 0,605 | 0,793 | 0,849 | 0,887 | 0,686 |

| Thời gian (ms) | Trung bình | p50 | p95 |
|---|---|---|---|
| Truy hồi | 0,7 | 0,6 | 1,3 |
| Reader (5 đoạn) | 66,4 | 62,9 | 103,8 |
| Tổng | 67,3 | 63,8 | 104,7 |
