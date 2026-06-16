from types import SimpleNamespace

import pytest

from embedding import create_embedding, create_embeddings, normalize_text


class FakeEmbeddings:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            model=request["model"],
            data=[
                SimpleNamespace(index=index, embedding=[float(index), 0.5])
                for index, _ in enumerate(request["input"])
            ],
        )


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


def test_create_embeddings_calls_openai_client():
    client = FakeClient()

    results = create_embeddings(
        ["first text", "second text"],
        model="text-embedding-3-small",
        dimensions=256,
        client=client,
    )

    assert client.embeddings.request == {
        "input": ["first text", "second text"],
        "model": "text-embedding-3-small",
        "encoding_format": "float",
        "dimensions": 256,
    }
    assert results[0].embedding == [0.0, 0.5]
    assert results[1].index == 1


def test_create_embedding_returns_single_result():
    result = create_embedding("hello", client=FakeClient())

    assert result.text == "hello"
    assert result.embedding == [0.0, 0.5]


def test_create_embeddings_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one text"):
        create_embeddings([], client=FakeClient())


def test_normalize_text_collapses_whitespace():
    assert normalize_text("hello\n\n   world") == "hello world"
