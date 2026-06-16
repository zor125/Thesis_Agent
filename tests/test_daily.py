from datetime import date
from types import SimpleNamespace

import pytest

import daily
from fetch import Paper


def test_get_interest_sentence_reads_config():
    sentence = daily.get_interest_sentence({"interests": ["Agent", "RAG"]})

    assert sentence.startswith("I am interested in research papers about Agent, RAG.")
    assert "building AI systems" in sentence


def test_build_interest_sentence_uses_natural_language_profile():
    sentence = daily.build_interest_sentence(["LLM Agents", "Coding Agents", "Tool Use"])

    assert "LLM Agents, Coding Agents, Tool Use" in sentence
    assert "research automation workflows" in sentence


def test_get_interest_sentence_keeps_legacy_sentence_config():
    assert daily.get_interest_sentence({"interests": {"sentence": "AI agents"}}) == "AI agents"


def test_get_obsidian_vault_path_prefers_env(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", "obsidian")

    assert daily.get_obsidian_vault_path({"vault_path": "/Users/me/Vault"}) == daily.Path("obsidian")


def test_rank_papers_by_interest_skips_failed_papers(monkeypatch):
    calls = []

    def fake_create_embeddings(texts, *, model):
        calls.append(texts)
        embeddings = []
        for index, text in enumerate(texts):
            if text == "interest" or "best paper" in text:
                embedding = [1.0, 0.0]
            else:
                embedding = [0.0, 1.0]
            embeddings.append(SimpleNamespace(index=index, embedding=embedding))
        return embeddings

    monkeypatch.setattr(daily, "create_embeddings", fake_create_embeddings)
    papers = [
        {"title": "best paper", "summary": "useful", "arxiv_id": "1"},
        {"title": "weak paper", "summary": "less useful", "arxiv_id": "3"},
    ]

    ranked = daily.rank_papers_by_interest(
        papers,
        "interest",
        embedding_model="test-embedding",
        top_k=20,
    )

    assert [item.paper["title"] for item in ranked] == ["best paper", "weak paper"]
    assert ranked[0].rank == 1
    assert calls == [["interest", "best paper\n\nuseful", "weak paper\n\nless useful"]]


def test_build_embedding_input_uses_title_and_summary():
    assert daily.build_embedding_input({"title": "Title", "summary": "Summary"}) == "Title\n\nSummary"


def test_count_embedding_candidates_counts_title_or_summary():
    papers = [
        {"title": "Title", "summary": ""},
        {"title": "", "summary": "Summary"},
        {"title": "", "summary": ""},
    ]

    assert daily.count_embedding_candidates(papers) == 2


def test_normalize_paper_accepts_fetch_paper_dataclass():
    paper = Paper(
        arxiv_id="2606.12345v1",
        title="Title",
        authors=["Ada"],
        summary="Summary",
        published="2026-06-16T00:00:00Z",
        updated="2026-06-16T00:00:00Z",
        categories=["cs.AI"],
        pdf_url="http://arxiv.org/pdf/2606.12345v1",
        entry_url="http://arxiv.org/abs/2606.12345v1",
    )

    normalized = daily.normalize_paper(paper)

    assert normalized["title"] == "Title"
    assert normalized["summary"] == "Summary"


def test_save_daily_markdown_uses_date_title_filename(tmp_path):
    ranked = [
        daily.RankedPaper(
            rank=1,
            score=0.99,
            paper={
                "title": "Useful AI Paper",
                "summary": "Abstract",
                "authors": ["Ada"],
                "categories": ["cs.AI"],
            },
            short_summary="Short summary",
        )
    ]

    saved_paths = daily.save_daily_markdown(
        ranked,
        tmp_path,
        folder="papers",
        target_date=date(2026, 6, 16),
        interests=["Agent", "RAG"],
    )

    assert tmp_path / "papers" / "2026-06-16-Useful AI Paper.md" in saved_paths
    assert tmp_path / "papers" / "2026-06-16-Top20.md" in saved_paths
    markdown = (tmp_path / "papers" / "2026-06-16-Useful AI Paper.md").read_text(encoding="utf-8")
    assert "- Similarity Score: 0.990000" in markdown
    assert "## My Insight" in markdown
    assert "## Startup Idea" in markdown
    assert "## Project Idea" in markdown
    assert "## Related Topics" in markdown
    assert "- [[Agent]]" in markdown
    assert "- [[RAG]]" in markdown


def test_infer_related_topics_adds_keyword_topics_and_categories():
    topics = daily.infer_related_topics(
        {
            "title": "Retrieval Agents for Robotics",
            "summary": "A policy improves embodied robot planning.",
            "categories": ["cs.AI"],
        },
        ["Agent"],
    )

    assert "Agent" in topics
    assert "RAG" in topics
    assert "Robotics" in topics
    assert "RL" in topics
    assert "cs.AI" in topics


def test_save_daily_markdown_rejects_empty_top_papers(tmp_path):
    with pytest.raises(ValueError, match="No top papers"):
        daily.save_daily_markdown([], tmp_path, folder="papers", target_date=date(2026, 6, 16))


def test_enrich_ranked_papers_keeps_paper_with_summary_fallback(monkeypatch, tmp_path):
    def fake_summarize_abstract(title, abstract, *, model):
        if title == "bad":
            raise RuntimeError("summary failed")
        return "summary"

    monkeypatch.setattr(daily, "summarize_abstract", fake_summarize_abstract)
    ranked = [
        daily.RankedPaper(rank=1, score=1.0, paper={"title": "bad", "summary": "x"}),
        daily.RankedPaper(rank=2, score=0.9, paper={"title": "good", "summary": "x"}),
    ]

    enriched = daily.enrich_ranked_papers(
        ranked,
        tmp_path,
        summary_model="summary-model",
        deep_read_model="deep-model",
        deep_read_count=0,
    )

    assert [item.paper["title"] for item in enriched] == ["bad", "good"]
    assert "OpenAI 요약 생성에 실패했습니다" in enriched[0].short_summary


def test_enrich_ranked_papers_keeps_top_paper_with_deep_read_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(daily, "summarize_abstract", lambda title, abstract, *, model: "summary")

    def fake_deep_read_pdf(title, pdf_url, work_dir, *, model):
        raise RuntimeError("pdf failed")

    monkeypatch.setattr(daily, "deep_read_pdf", fake_deep_read_pdf)
    ranked = [
        daily.RankedPaper(
            rank=1,
            score=1.0,
            paper={
                "title": "top",
                "summary": "abstract",
                "entry_url": "http://arxiv.org/abs/1",
                "pdf_url": "http://arxiv.org/pdf/1",
            },
        )
    ]

    enriched = daily.enrich_ranked_papers(
        ranked,
        tmp_path,
        summary_model="summary-model",
        deep_read_model="deep-model",
        deep_read_count=1,
    )

    assert len(enriched) == 1
    assert "PDF 심층 분석에 실패했습니다" in enriched[0].deep_analysis
