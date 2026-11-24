# Model card của các reranker ứng viên (GĐ14.1)

Đọc ngày 2026-11-09 từ tệp `README.md` đã tải về (cache HuggingFace), không từ trí nhớ. Mỗi trích dẫn là nguyên văn trong card.
Mã commit ghi để ai cần thì đối chiếu đúng phiên bản.

| Model | Commit đã đọc | Đầu ra | Độ dài tối đa (config) |
|---|---|---|---|
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | 1 logit | 514 vị trí |
| `deepset/gbert-base-germandpr-reranking` | `99d016fcc5d2cb9e7355adb3a1e9efa866821723` | 2 nhãn (`0`, `1`) | 512 |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 1 logit | 8194 vị trí |
| `ml6team/cross-encoder-mmarco-german-distilbert-base` | `c6d4a9626a5025e8caa10ded1e83f9187aaca4f0` | 1 logit | 512 |

## 1. `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (ứng viên `mmarco`)

* Giấy phép: Apache-2.0. Ngôn ngữ nêu trong card gồm `de`.
* Dữ liệu huấn luyện: *"This model was trained on the MMARCO dataset. It is a machine translated version of MS MARCO using Google Translate. It was translated to 14 languages. In our experiments, we observed that it performs also well for other languages."*
* Mô hình nền: *multilingual MiniLMv2* (`nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`).
* **Không nhắc GermanQuAD hay GermanDPR.** Tiếng Đức ở đây là bản dịch máy của MS MARCO, không phải dữ liệu hỏi–đáp tiếng Đức gốc.

## 2. `deepset/gbert-base-germandpr-reranking` (ứng viên `gbert_dpr`)

* Giấy phép: MIT. Ngôn ngữ: Đức.
* *"Training data: GermanDPR train set (~ 56MB)"*; *"Eval data: GermanDPR test set"*.
* *"We trained a text pair classification model in FARM … the classifier calculates the similarity of the query and each retrieved top k document (e.g., k=10)."*
* Siêu tham số: *"max_seq_len = 512 tokens for question and passage concatenated"*, 2 epoch. (Khớp với `max_length = 512` đã dùng ở GĐ14; `14a_probe.py` đặt 510 do mặc định của script, đã ghi chú trong prereg.)
* Đánh giá trong card: BM25 + rerank top-10 trên GermanDPR test; không có số trên GermanQuAD.
* **Đã học GermanDPR train**: cùng nguồn câu hỏi với tập dev/train của mình (đã loại các câu trùng chữ khỏi tập chọn), và đoạn văn từ đó có mặt trong corpus.

## 3. `BAAI/bge-reranker-v2-m3` (ứng viên `bge_m3`)

* Giấy phép: Apache-2.0. Bảng trong card ghi mô hình nền là `bge-m3`, ngôn ngữ *Multilingual*, *"possesses strong multilingual capabilities, easy to deploy, with fast inference"*.
* Dữ liệu huấn luyện, nguyên văn: *"we train it on a mixture of multilingual datasets: bge-m3-data; quora train data; fever train data"*.
* Card dữ liệu `Shitao/bge-m3-data` liệt kê: MS MARCO, NQ, HotpotQA, TriviaQA, SQuAD, COLIEE, PubMedQA, NLI, DuReader, mMARCO-zh, T2Ranking, Law-GPT, cMedQAv2, NLI-zh, LeCaRDv2, **Mr.TyDi (11 ngôn ngữ), MIRACL (16 ngôn ngữ), MLDR (13 ngôn ngữ)**. Card không liệt kê tên các ngôn ngữ của ba bộ này.
* **Không nhắc GermanQuAD hay GermanDPR** ở cả hai card. Nhưng GermanQuAD/GermanDPR xây từ Wikipedia tiếng Đức; các bộ đa ngôn ngữ ở trên có thể chứa đoạn trùng, và card không cho đủ thông tin để kiểm điều đó.

## 4. `ml6team/cross-encoder-mmarco-german-distilbert-base` (ứng viên thăm dò, chưa có trong prereg)

* Giấy phép: Apache-2.0. Ngôn ngữ: Đức.
* *"fine-tuned cross-encoder on the MMARCO dataset which is the machine translated version of the MS MARCO dataset. As base model … distilbert-base-multilingual-cased."*
* Mẫu huấn luyện: *"`<query, positive_paragraph>` assigned to 1 or `<query, negative_paragraph>` assigned to 0. The model was trained for 1 epoch."*
* Đánh giá trong card: *"Model evaluation was done on 2000 evaluation paragraphs of the dataset"*, Accuracy 89,70; F1 86,82. Đây là độ chính xác phân loại cặp trên chính mMARCO, **không phải** chỉ số xếp hạng, nên không so được với Hit@5 của mình.
* **Không nhắc GermanQuAD hay GermanDPR.**

## 5. Tóm tắt về nguy cơ “đã thấy đoạn”

| Model | Dữ liệu huấn luyện theo card | Nhắc GermanDPR/GermanQuAD? | Kết luận cho phân tích nhóm |
|---|---|---|---|
| mmarco | mMARCO (dịch máy MS MARCO) | Không | đối chứng sạch ở mức dữ liệu được nêu |
| ml6team | mMARCO | Không | như mmarco |
| bge_m3 | bge-m3-data + quora + fever | Không | không có bằng chứng bị nhiễm, nhưng **không kiểm được trùng đoạn Wikipedia** |
| gbert_dpr | GermanDPR train | **Có** | có tiếp xúc, đã tách riêng trong phân tích |

Giới hạn của phần này: tao chỉ đọc card, không đọc bài báo hay mã huấn luyện, và không có cách kiểm trùng đoạn ở mức văn bản với dữ liệu huấn luyện của bge-m3. Vì vậy “sạch” ở đây nghĩa là *card không nêu GermanQuAD/GermanDPR*, không nghĩa là đã chứng minh không trùng.
