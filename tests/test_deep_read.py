from types import SimpleNamespace

from deep_read import analyze_full_text


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(output_text="## 핵심 기여\n\n분석")


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_analyze_full_text_uses_full_text_excerpt():
    client = FakeClient()

    result = analyze_full_text("Paper title", "Full text", client=client, model="test-model")

    assert "핵심 기여" in result
    assert client.responses.request["model"] == "test-model"
    assert "reasoning" not in client.responses.request
    assert "## 스타트업 아이디어" in client.responses.request["instructions"]
    assert "Full text" in client.responses.request["input"]
