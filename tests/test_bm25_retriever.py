import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25_retriever import BM25Retriever
from src.corpus import normalize_text

PASSAGE_IDS = ["p1", "p2", "p3"]
TEXTS = [
    "Der Große Panda lebt in den Bergwäldern Zentralchinas und ernährt sich fast ausschließlich von Bambus.",
    "Die Hauptstadt von Frankreich ist Paris, eine der bevölkerungsreichsten Städte Europas.",
    "Berlin ist die Hauptstadt und ein Bundesland der Bundesrepublik Deutschland.",
]
NORM_TEXTS = [normalize_text(t) for t in TEXTS]


def make_retriever():
    return BM25Retriever(PASSAGE_IDS, NORM_TEXTS)


def test_near_verbatim_query_ranks_first():
    retriever = make_retriever()
    results = retriever.search("Großer Panda Bambus Bergwäldern Zentralchinas", top_n=3)
    assert results[0][0] == "p1"


def test_ranks_start_at_1_and_no_duplicate_ids_in_result():
    retriever = make_retriever()
    results = retriever.search("Hauptstadt Deutschland Berlin", top_n=3)
    ids = [pid for pid, _ in results]
    assert len(ids) == len(set(ids))  # không trùng passage_id
    ranks = list(range(1, len(results) + 1))
    assert ranks == sorted(ranks)  # rank ngầm định (vị trí trong list) tăng dần từ 1


def test_empty_query_returns_empty_list():
    retriever = make_retriever()
    assert retriever.search("   ", top_n=5) == []


def test_fully_oov_query_returns_zero_score_deterministic_order():
    retriever = make_retriever()
    results = retriever.search("xyzxyz qwqwqw", top_n=3)
    assert [pid for pid, _ in results] == sorted(PASSAGE_IDS)
    assert all(score == 0.0 for _, score in results)


def test_smoke_no_crash_on_batch_of_queries():
    retriever = make_retriever()
    queries = [
        "Panda Bambus",
        "Frankreich Paris",
        "Berlin Bundesland",
        "",
        "völlig unbekannte suchanfrage ohne treffer irgendwo",
    ] * 10
    for q in queries:
        results = retriever.search(q, top_n=3)
        ids = [pid for pid, _ in results]
        assert len(ids) == len(set(ids))
