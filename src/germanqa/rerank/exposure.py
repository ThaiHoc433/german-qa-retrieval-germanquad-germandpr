"""Mức 'đã thấy đoạn' (GĐ14): đoạn đúng của một câu hỏi có từng xuất hiện trong GermanDPR train không, và với vai gì.

Encoder dense và gbert-base-germandpr-reranking đã học trên GermanDPR train. Tập chọn chỉ loại các câu hỏi có chữ trùng,
còn ĐOẠN đúng thì phần lớn vẫn từng là đoạn dương của câu khác. Test thì không có đoạn đúng nào như vậy.
Nhóm (cho mỗi câu hỏi, xét tất cả đoạn đúng trong qrels):
  positive : có ít nhất một đoạn đúng từng là đoạn DƯƠNG trong GermanDPR train
  negative : không có đoạn dương nào, nhưng có đoạn đúng từng xuất hiện với vai khác (đoạn âm)
  never    : không đoạn đúng nào từng xuất hiện trong GermanDPR train
Dữ liệu lấy từ trường 'source' của corpus.jsonl, không cần dữ liệu thô."""

GROUPS = ("positive", "negative", "never")


def passage_role(passage):
    roles = {x["role"] for x in passage.get("source", []) if x.get("dataset") == "germandpr" and x.get("split") == "train"}
    return "positive" if "positive" in roles else ("negative" if roles else "never")


def question_group(gold_ids, corpus_by_id):
    roles = [passage_role(corpus_by_id[p]) for p in gold_ids]
    return "positive" if "positive" in roles else ("negative" if "negative" in roles else "never")
