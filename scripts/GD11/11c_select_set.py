"""GĐ12.0: lập tập chọn = câu hỏi train + dev mà encoder dense chưa thấy. Chỉ đọc qa_train và qa_dev, KHÔNG đọc qa_test."""
from datasets import load_from_disk

from germanqa.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.evaluation.select_set import build_select_set


def main():
    dpr = load_from_disk(str(RAW_DIR / "germandpr"))
    qa = {s: read_jsonl(PROCESSED_DIR / f"qa_{s}.jsonl") for s in ("train", "dev")}
    sel = build_select_set(qa, [r["question"] for r in dpr["train"]])
    dev_unseen = read_json(RESULTS_DIR / "dev_unseen_question_ids.json")       # kết quả GĐ6.2
    assert sel["dev"] == sorted(dev_unseen), "phần dev của tập chọn phải đúng bằng dev_unseen của GĐ6"
    write_json(RESULTS_DIR / "select_question_ids.json", sel)
    print(sel["counts"])


if __name__ == "__main__":
    main()
