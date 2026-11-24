"""Reciprocal Rank Fusion (GĐ13.1): hợp nhất danh sách theo HẠNG, không cộng điểm thô.

    RRF(d) = tổng trên các danh sách r của  w_r / (c + rank_r(d)),   rank bắt đầu từ 1, vắng khỏi danh sách -> 0.

c là hằng số làm trơn, KHÔNG phải top-k. w_r là trọng số của nhánh r (mặc định 1: RRF thường)."""


def rrf(ranked_lists, c=60, top_k=None, weights=None):
    """ranked_lists: {tên_nhánh: [passage_id theo thứ tự giảm dần]} (hạng = vị trí + 1).
    weights: {tên_nhánh: w >= 0}; nhánh không có trong weights mang trọng số 1. None = RRF thường.
    Trả về [{'passage_id', 'score', 'rank', 'sources': {tên: hạng}}] giảm dần theo điểm RRF.
    Đồng điểm: hạng tốt nhất ở bất kỳ nhánh nào (nhỏ hơn đứng trước), rồi passage_id; kết quả tất định.
    Một passage_id lặp trong cùng một danh sách chỉ tính lần đầu (hạng tốt nhất)."""
    if c < 0:
        raise ValueError("c phải >= 0")
    weights = dict(weights or {})
    unknown = set(weights) - set(ranked_lists)
    if unknown:
        raise ValueError(f"trọng số cho nhánh không tồn tại: {sorted(unknown)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("trọng số phải >= 0")
    score, sources = {}, {}
    for name, ids in ranked_lists.items():          # thứ tự nhánh cố định -> tổng float tất định
        w = float(weights.get(name, 1.0))
        seen = set()
        for pos, pid in enumerate(ids, start=1):
            if pid in seen:
                continue
            seen.add(pid)
            score[pid] = score.get(pid, 0.0) + w / (c + pos)
            sources.setdefault(pid, {})[name] = pos
    order = sorted(score, key=lambda p: (-score[p], min(sources[p].values()), p))
    if top_k is not None:
        order = order[:max(top_k, 0)]
    return [{"passage_id": p, "score": score[p], "rank": r, "sources": sources[p]} for r, p in enumerate(order, start=1)]
