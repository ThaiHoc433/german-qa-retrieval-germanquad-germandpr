"""Gán train/dev/test hợp nhất theo question_key giữa GermanQuAD và GermanDPR —
module 2.3.e. Sửa lỗi bản thiết kế đầu: KHÔNG chia dev độc lập từng dataset."""
import random
import re
import unicodedata

DEV_RATIO = 0.10
SEED = 42


def question_key(q: str) -> str:
    text = unicodedata.normalize("NFC", q)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def assign_splits(gq_train_qs, gq_test_qs, gdpr_train_qs, gdpr_test_qs, dev_ratio=DEV_RATIO, seed=SEED):
    gq_train_keys = set(map(question_key, gq_train_qs))
    gq_test_keys = set(map(question_key, gq_test_qs))
    gdpr_train_keys = set(map(question_key, gdpr_train_qs))
    gdpr_test_keys = set(map(question_key, gdpr_test_qs))

    test_group = gq_test_keys | gdpr_test_keys
    train_pool_raw = gq_train_keys | gdpr_train_keys
    removed_due_to_leakage = train_pool_raw & test_group
    train_pool = train_pool_raw - test_group

    train_pool_sorted = sorted(train_pool)
    rng = random.Random(seed)
    dev_size = int(len(train_pool_sorted) * dev_ratio)
    dev_keys = set(rng.sample(train_pool_sorted, dev_size))
    train_keys = train_pool - dev_keys

    assignment = {k: "test" for k in test_group}
    assignment.update({k: "dev" for k in dev_keys})
    assignment.update({k: "train" for k in train_keys})

    report = {
        "test_group_size": len(test_group),
        "train_pool_before_leakage_removal": len(train_pool_raw),
        "removed_due_to_leakage": len(removed_due_to_leakage),
        "removed_examples": sorted(removed_due_to_leakage)[:10],
        "train_pool_after_leakage_removal": len(train_pool),
        "dev_size": len(dev_keys),
        "train_size": len(train_keys),
        "dev_ratio": dev_ratio,
        "seed": seed,
    }
    return assignment, report
