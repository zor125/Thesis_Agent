from types import SimpleNamespace

from summarize import analyze_abstract, summarize_abstract


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            output_text="""{
              "one_sentence_summary": "A one sentence summary.",
              "tldr": "Line one.\\nLine two.\\nLine three.",
              "key_contributions": "Contribution.",
              "why_important": "Important.",
              "difference_from_previous_work": "Different.",
              "limitations": "Limited.",
              "my_insight": "Insight.",
              "can_i_build_it": {
                "difficulty": "⭐⭐⭐☆☆",
                "time_estimate": "1 week",
                "need_gpu": "No",
                "need_dataset": "Yes",
                "undergraduate_friendly": "Yes",
                "suggested_mini_project": "Build a small demo."
              },
              "startup_idea": "Startup.",
              "project_idea": "Project.",
              "related_topics": ["NLP Evaluation", "Dataset"],
              "tags": ["nlp", "dataset", "evaluation"]
            }"""
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_summarize_abstract_uses_responses_api():
    client = FakeClient()

    result = summarize_abstract("Paper title", "Abstract text", client=client, model="test-model")

    assert result.startswith("## One Sentence Summary")
    assert client.responses.request["model"] == "test-model"
    assert "reasoning" not in client.responses.request
    assert "strict JSON" in client.responses.request["instructions"]
    assert "Write all human-readable analysis fields in Korean" in client.responses.request["instructions"]
    assert "tags in lowercase kebab-case English" in client.responses.request["instructions"]
    assert "Paper title" in client.responses.request["input"]
    assert "Abstract text" in client.responses.request["input"]


def test_analyze_abstract_parses_dynamic_tags_and_topics():
    analysis = analyze_abstract("Paper title", "Abstract text", client=FakeClient(), model="test-model")

    assert analysis.tags == ["nlp", "dataset", "evaluation"]
    assert analysis.related_topics == ["NLP Evaluation", "Dataset"]
    assert analysis.can_i_build_it.suggested_mini_project == "Build a small demo."
