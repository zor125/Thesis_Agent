import json

import pytest

from rank import cosine_similarity, load_query_embedding, rank_embeddings


def test_cosine_similarity_scores_identical_vectors_as_one():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_rank_embeddings_returns_top_k_in_score_order():
    items = [
        {"text": "weak", "embedding": [0.0, 1.0], "index": 0, "model": "test"},
        {"text": "best", "embedding": [1.0, 0.0], "index": 1, "model": "test"},
        {"text": "middle", "embedding": [0.5, 0.5], "index": 2, "model": "test"},
    ]

    results = rank_embeddings([1.0, 0.0], items, top_k=2)

    assert [result.text for result in results] == ["best", "middle"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score == pytest.approx(1.0)


def test_rank_embeddings_defaults_to_top_20():
    items = [
        {"text": str(index), "embedding": [float(index + 1), 1.0]}
        for index in range(25)
    ]

    results = rank_embeddings([1.0, 0.0], items)

    assert len(results) == 20


def test_load_query_embedding_reads_embedding_py_output(tmp_path):
    query_file = tmp_path / "query.json"
    query_file.write_text(
        json.dumps([{"text": "query", "embedding": [0.1, 0.2]}]),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {"query_embedding": None, "query_embedding_file": query_file},
    )()

    assert load_query_embedding(args) == [0.1, 0.2]
