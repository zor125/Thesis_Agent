from types import SimpleNamespace

from summarize import summarize_abstract


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(output_text="## 한 줄 요약\n\ntest")


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_summarize_abstract_uses_responses_api():
    client = FakeClient()

    result = summarize_abstract("Paper title", "Abstract text", client=client, model="test-model")

    assert result.startswith("## 한 줄 요약")
    assert client.responses.request["model"] == "test-model"
    assert "reasoning" not in client.responses.request
    assert "## 내 프로젝트 적용 아이디어" in client.responses.request["instructions"]
    assert "## My Insight" in client.responses.request["instructions"]
    assert "## Startup Idea" in client.responses.request["instructions"]
    assert "## Project Idea" in client.responses.request["instructions"]
    assert "## Related Topics" in client.responses.request["instructions"]
    assert "Paper title" in client.responses.request["input"]
    assert "Abstract text" in client.responses.request["input"]
