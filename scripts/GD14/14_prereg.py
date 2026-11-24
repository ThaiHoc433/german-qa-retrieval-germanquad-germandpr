"""GĐ14.0: ghi results/14_prereg.json TRƯỚC khi chấm bất kỳ cặp (câu hỏi, đoạn) thật nào bằng reranker.
Khóa: danh sách ứng viên và cấu hình chấm, tập chọn/xác nhận, quy tắc chọn, quy tắc nhận, kiểm tra 'đã thấy đoạn',
SHA-1 của đầu vào, của mã, và của trọng số từng model. 14c/14d/14e từ chối chạy nếu thiếu file hoặc có SHA-1 lệch.
Không ghi đè nếu đã tồn tại.

Thứ tự thực tế ở GĐ14: module reranker + test (14.2), thăm dò cấu hình (14a) và kiểm cắt token (14b, chỉ dùng độ dài và
vị trí đáp án, không dùng điểm reranker) xong TRƯỚC file này; chưa có điểm reranker nào của dữ liệu thật khi file được ghi."""
import time

from germanqa.config import RESULTS_DIR, ROOT, SEED
from germanqa.data.io_utils import file_sha1, read_json, write_json
from germanqa.evaluation.fusion_select import inputs_sha1
from germanqa.rerank.hub import local_snapshot

INPUTS = ["results/B0_select_top100.jsonl", "results/B1_select_top100_384.jsonl", "results/select_question_ids.json",
          "results/dev_unseen_question_ids.json", "results/13_selection.json", "results/14a_probe.json",
          "results/14b_truncation.json", "data/processed/corpus.jsonl", "data/processed/qrels_train.json",
          "data/processed/qrels_dev.json", "configs/hybrid.yaml"]
CODE = ["src/germanqa/rerank/cross_encoder.py", "src/germanqa/rerank/pool.py", "src/germanqa/rerank/exposure.py",
        "src/germanqa/rerank/hub.py", "src/germanqa/rerank/select.py", "src/germanqa/evaluation/decision.py", "src/germanqa/evaluation/bootstrap.py",
        "src/germanqa/retrieval/rrf.py"]
MODELS = {   # thứ tự này cũng là thứ tự 'rẻ trước' khi hòa điểm
    "mmarco": {"repo": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "relevant_label": None, "params_m": 118,
               "training": "mMARCO (dịch MS MARCO sang nhiều ngôn ngữ, có tiếng Đức); không dùng GermanQuAD/GermanDPR",
               "dpr_exposure": "không"},
    "gbert_dpr": {"repo": "deepset/gbert-base-germandpr-reranking", "relevant_label": 1, "params_m": 110,
                  "training": "GermanDPR train (model card)", "dpr_exposure": "có: câu hỏi (đã loại khỏi tập chọn theo chữ) và đoạn"},
    "bge_m3": {"repo": "BAAI/bge-reranker-v2-m3", "relevant_label": None, "params_m": 568,
               "training": "đa ngôn ngữ; dữ liệu huấn luyện không công bố đầy đủ -> coi là 'chưa kiểm chứng', "
                           "không khẳng định là sạch", "dpr_exposure": "không rõ"},
}


def main():
    out = RESULTS_DIR / "14_prereg.json"
    if out.exists():
        raise SystemExit(f"{out.name} đã tồn tại: không ghi đè (quy tắc ghi trước)")
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    unseen = read_json(RESULTS_DIR / "dev_unseen_question_ids.json")
    assert set(sel["dev"]) == set(unseen), "dev trong tập chọn phải đúng bằng dev_unseen"
    models = {}
    for key, m in MODELS.items():
        snap = local_snapshot(m["repo"])
        import pathlib
        files = {p.name: file_sha1(p) for p in sorted(pathlib.Path(snap).iterdir())
                 if p.suffix in (".safetensors", ".model") or p.name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")}
        models[key] = {**m, "commit": pathlib.Path(snap).name, "files_sha1": files}
    pre = {
        "phase": "GĐ14", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
        "purpose": "H1 = H0 (hybrid RRF top-20) + cross-encoder xếp lại + cùng reader lấy top-5. Dùng checkpoint có SẴN, KHÔNG fine-tune.",
        "inputs": inputs_sha1(ROOT, INPUTS), "code": inputs_sha1(ROOT, CODE), "models": models,
        "pool": {"source": "top-20 của H0 tái dựng từ top-100 BM25 + dense384 bằng cấu hình đã chốt ở GĐ13",
                 "reader_top_k": 5, "note": "reranker không thêm đoạn ngoài pool: trần Hit@5 = Hit@20 của H0"},
        "scoring": {"text_seen": "original_text (cùng văn bản reader đọc)", "max_length": 512, "batch_size": 16, "fp16": True,
                    "stride_windows": 128, "max_query_tokens": 64,
                    "score": "logit liên quan (model một đầu ra) hoặc log-odds của nhãn liên quan (model nhiều nhãn); không phải xác suất",
                    "tie": "điểm bằng nhau thì giữ hạng RRF cũ",
                    "note": "14a ghi max_length_used=510 cho gbert_dpr do mặc định của script thăm dò; số đúng cho BERT là 512 và là số dùng ở đây",
                    "one_pass": "chấm bằng cửa sổ một lần: điểm cửa sổ đầu = điểm khi chỉ cắt cuối đoạn, điểm lớn nhất = điểm 'windows'"},
        "overflow_variants": ["truncate", "windows"],
        "configs": {"n": 6, "grid": "3 model x 2 cách xử lý đoạn dài", "exploratory": "không có; không thử cấu hình thứ bảy"},
        "splits": {"selection": f"train-select (n={len(sel['train'])})", "confirmation": f"dev-unseen (n={len(sel['dev'])})",
                   "test": "KHÔNG mở"},
        "access_rule": "14d chỉ đọc điểm của train-select. Điểm của dev-unseen chỉ được đọc bởi 14e, sau khi 14_selection.json tồn tại.",
        "metrics": {"primary": "hit@5", "secondary_for_adoption": "hit@20", "also_reported": ["mrr@5", "mrr@10", "recall@20"]},
        "selection_rule": "Hit@5 cao nhất trên train-select (toàn bộ 1.989 câu); hòa -> MRR@10 cao hơn -> rẻ hơn "
                          "(truncate trước windows, rồi theo thứ tự mmarco, gbert_dpr, bge_m3). Hàm: rerank.select.choose_config",
        "adoption_rule": {"function": "decision.adopt_alternative", "margin": 0.01,
                          "primary": "Hit@5 chênh ghép cặp (cấu hình đã chọn - H0) > 0 với cận dưới > 0 trên train-select",
                          "secondary": "Hit@20 cận dưới >= -0.01",
                          "exposure_guard": "trên nhóm 'never' của train-select (đoạn đúng chưa từng xuất hiện trong GermanDPR train, n=816), "
                                            "chênh Hit@5 ghép cặp phải có điểm ước lượng > 0 và cận dưới >= -0.01. Test không có đoạn đúng nào "
                                            "từng là đoạn dương của DPR, nên nhóm này giống test nhất.",
                          "if_rejected": "H1 vẫn được chạy với cấu hình đã chọn để lấp ma trận GĐ15, nhưng ghi 'không nhận: chưa đủ bằng chứng' "
                                         "và khuyến nghị giữ H0 khi triển khai"},
        "exposure_analysis": {"groups": {"positive": "có đoạn đúng từng là đoạn dương của GermanDPR train",
                                         "negative": "không có đoạn dương, nhưng có đoạn đúng từng xuất hiện làm đoạn âm",
                                         "never": "không đoạn đúng nào từng xuất hiện"},
                              "report_per_config": "Hit@5 chênh so với H0 theo từng nhóm, và chênh của chênh (positive - never) với bootstrap",
                              "controls": "mmarco và bge_m3 không huấn luyện trên DPR: nếu chúng không khác nhau giữa các nhóm mà gbert_dpr khác "
                                          "nhiều thì đó là dấu hiệu hiệu ứng 'đã thấy đoạn'",
                              "counts_in_selection": {"train-select": {"positive": 963, "negative": 210, "never": 816},
                                                      "dev-unseen": {"positive": 114, "negative": 12, "never": 108}}},
        "confirmation_rule": "Chạy cấu hình đã chọn MỘT lần trên dev-unseen. H1 chỉ được gọi là hơn H0 về Hit@5 nếu cận dưới ghép cặp > 0 "
                             "trên 234 câu; ngược lại ghi 'chưa đủ bằng chứng'. EM/F1 chỉ báo cáo kèm khoảng tin cậy. Không chọn lại sau bước này.",
        "bootstrap": {"n_boot": 1000, "confidence": 0.95, "seed": 42, "unit": "câu hỏi, ghép cặp"},
        "expectation_written_before_results": "Trần Hit@5 tăng thêm khoảng 9 điểm (Hit@20 0,98 so với Hit@5 0,89 trên dev-unseen). "
                                              "Dense (Hit@5 0,816) và hybrid (0,899) hơn hẳn B0 về Hit@5 nhưng EM của hybrid (0,329) không hơn dense (0,339): "
                                              "reader là nút cổ chai, nên H1 có thể tăng Hit@5 mà EM hầu như không đổi.",
        "max_configs_tried": 6,
    }
    write_json(out, pre)
    print("đã ghi", out.name, "| đầu vào:", len(pre["inputs"]), "| mã:", len(pre["code"]), "| model:", len(pre["models"]))


if __name__ == "__main__":
    main()
