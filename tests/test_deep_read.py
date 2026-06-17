from types import SimpleNamespace

from deep_read import analyze_full_text


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(output_text="## Paper Type\n\n- Type: Survey\n- 근거: survey입니다.\n\n## Taxonomy\n\n분석")


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_analyze_full_text_uses_full_text_excerpt():
    client = FakeClient()

    result = analyze_full_text(
        "Paper title",
        "Full text with official code https://github.com/example/full-text-code",
        abstract="Abstract",
        client=client,
        model="test-model",
    )

    assert result.paper_type == "Survey"
    assert "## Paper Type" in result.markdown
    assert "## Code / Resources" in result.markdown
    assert "https://github.com/example/full-text-code" in result.markdown
    assert client.responses.request["model"] == "test-model"
    assert "reasoning" not in client.responses.request
    assert "## 스타트업 아이디어" in client.responses.request["instructions"]
    assert "## For Me" in client.responses.request["instructions"]
    assert "## Can I Build It?" in client.responses.request["instructions"]
    assert "## Key Figure / Core Diagram" in client.responses.request["instructions"]
    assert "## Reading Path" in client.responses.request["instructions"]
    assert "## Evaluation" in client.responses.request["instructions"]
    assert "## Next Action" in client.responses.request["instructions"]
    assert "## Research Position" in client.responses.request["instructions"]
    assert "## Comparison Table" in client.responses.request["instructions"]
    assert "Paper, Planning, Memory, Tool, Benchmark" in client.responses.request["instructions"]
    assert "## If I Were Building This" in client.responses.request["instructions"]
    assert "## Open Questions" in client.responses.request["instructions"]
    assert "## Future Work Ideas" in client.responses.request["instructions"]
    assert "Novelty" in client.responses.request["instructions"]
    assert "Relevance to My Interests" in client.responses.request["instructions"]
    assert "Recommended Tech Stack" in client.responses.request["instructions"]
    assert "survey" in client.responses.request["instructions"]
    assert "Survey, Research, Benchmark, Dataset, System, Position" in client.responses.request["instructions"]
    assert "Mark exactly 1-2 recommended actions with [x]" in client.responses.request["instructions"]
    assert "Abstract" in client.responses.request["input"]
    assert "Full text" in client.responses.request["input"]
