from datetime import date
import threading
import time
from types import SimpleNamespace

import pytest

import daily
from fetch import Paper
from summarize import BuildPlan, PaperAnalysis


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
                "arxiv_id": "2606.12345v2",
                "title": "Useful AI Paper",
                "summary": "Abstract",
                "authors": ["Ada"],
                "categories": ["cs.AI"],
            },
            analysis=PaperAnalysis(
                one_sentence_summary="One sentence.",
                tldr="TLDR.",
                key_contributions="Contributions.",
                why_important="Important.",
                difference_from_previous_work="Different.",
                limitations="Limitations.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(
                    difficulty="⭐⭐⭐⭐⭐",
                    time_estimate="1 week",
                    need_gpu="No",
                    need_dataset="Yes",
                    undergraduate_friendly="Yes",
                    suggested_mini_project="Build a demo.",
                ),
                startup_idea="Startup.",
                project_idea="Project.",
                related_topics=["NLP Evaluation", "Dataset"],
                tags=["nlp", "dataset", "evaluation"],
            ),
        )
    ]

    saved_paths = daily.save_daily_markdown(
        ranked,
        ranked,
        tmp_path,
        folder="papers",
        target_date=date(2026, 6, 16),
        interests=["Agent", "RAG"],
        deep_read_k=0,
    )

    assert tmp_path / "papers" / "Daily" / "2026-06-16.md" in saved_paths
    assert tmp_path / "papers" / "Papers" / "2606.12345-useful-ai-paper.md" in saved_paths
    markdown = (tmp_path / "papers" / "Papers" / "2606.12345-useful-ai-paper.md").read_text(encoding="utf-8")
    assert "- Interest Level: ⭐⭐⭐⭐⭐" in markdown
    assert "## Should I Read This?" in markdown
    assert "* Target Audience:" in markdown
    assert "* Worth Reading:" in markdown
    assert "  * Novelty:" in markdown
    assert "  * Research Value:" in markdown
    assert "  * Practical Impact:" in markdown
    assert "  * Project Potential:" in markdown
    assert "## Remember Only One Thing" in markdown
    assert "## One Big Question" in markdown
    assert "이 평가 방식은 실제 에이전트 성능 차이를 얼마나 공정하게 드러낼 수 있을까?" in markdown
    assert "## Connect To My Research" in markdown
    assert "* Agent:" in markdown
    assert "* RAG:" in markdown
    assert "★★★★★" in markdown or "★★★☆☆" in markdown or "★☆☆☆☆" in markdown
    assert "## Why Ranked Top5?" in markdown
    assert "## Better Can I Build It?" in markdown
    assert "* GPU Requirement: No" in markdown
    assert "* Dataset Requirement: Yes" in markdown
    assert "* Framework Recommendation:" in markdown
    assert "## Better Startup Idea" in markdown
    assert "* Target Customer:" in markdown
    assert "* Competitive Advantage:" in markdown
    assert "## Better Project Idea" in markdown
    assert "* Beginner:" in markdown
    assert "* Intermediate:" in markdown
    assert "* Advanced:" in markdown
    assert "## Can I Build It?" not in markdown
    assert "## My Insight" in markdown
    assert "## Startup Idea" in markdown
    assert "## Project Idea" in markdown
    assert "## Related Topics" in markdown
    assert markdown.count("## My Insight") == 1
    assert markdown.count("## Startup Idea") == 1
    assert markdown.count("## Project Idea") == 1
    assert markdown.count("## Related Topics") == 1
    assert markdown.count("## Should I Read This?") == 1
    assert markdown.count("## Better Can I Build It?") == 1
    assert markdown.count("## Better Startup Idea") == 1
    assert markdown.count("## Better Project Idea") == 1
    assert markdown.count("## Next Action") == 1
    assert "* [ ] Read Deep Note" in markdown
    assert "* [ ] Clone Code" in markdown
    assert "* [x] Reproduce" in markdown
    assert "* [x] Add to Idea List" in markdown
    assert "- [[NLP Evaluation]]" in markdown
    assert "- [[Dataset]]" in markdown
    assert '  - "nlp"' in markdown
    assert "## Related Papers" in markdown
    assert "No strong related papers found." in markdown


def test_save_daily_markdown_links_existing_related_papers_with_cache(monkeypatch, tmp_path, caplog):
    caplog.set_level(daily.logging.DEBUG)
    cache_path = tmp_path / ".cache" / "embeddings.json"
    monkeypatch.setattr(daily, "RELATED_EMBEDDING_CACHE_PATH", cache_path)
    base_dir = tmp_path / "AI Papers"
    existing_dir = base_dir / "Papers"
    existing_dir.mkdir(parents=True)
    existing_note = existing_dir / "StoryBench.md"
    existing_note.write_text(
        """---
title: "StoryBench"
tags:
  - "benchmark"
  - "evaluation"
---
# StoryBench

## One Sentence Summary

스토리 평가 벤치마크입니다.

## Abstract
<details>
<summary>Original Abstract</summary>

story benchmark evaluation

</details>
""",
        encoding="utf-8",
    )

    calls = []

    def fake_create_embeddings(texts, *, model):
        calls.append(texts)
        return [
            SimpleNamespace(index=index, embedding=[1.0, 0.0])
            for index, _ in enumerate(texts)
        ]

    monkeypatch.setattr(daily, "create_embeddings", fake_create_embeddings)
    ranked = [
        daily.RankedPaper(
            rank=1,
            score=0.91,
            paper={
                "title": "New Story Evaluation",
                "summary": "story benchmark evaluation for agents",
                "authors": ["Ada"],
                "categories": ["cs.AI"],
            },
            analysis=PaperAnalysis(
                one_sentence_summary="요약.",
                tldr="요약.",
                key_contributions="기여.",
                why_important="중요.",
                difference_from_previous_work="차이.",
                limitations="한계.",
                my_insight="인사이트.",
                can_i_build_it=BuildPlan(),
                startup_idea="스타트업.",
                project_idea="프로젝트.",
                related_topics=["Benchmark"],
                tags=["benchmark", "evaluation"],
            ),
            embedding=[1.0, 0.0],
        )
    ]

    daily.save_daily_markdown(
        ranked,
        ranked,
        tmp_path,
        folder="AI Papers",
        target_date=date(2026, 6, 17),
        embedding_model="test-embedding",
    )

    markdown = (base_dir / "Papers" / "new-story-evaluation.md").read_text(encoding="utf-8")
    assert "[[Papers/StoryBench|StoryBench]]" in markdown
    assert "  - Common Point:" in markdown
    assert "  - Difference:" in markdown
    assert calls
    assert cache_path.exists()
    assert "Embedding cache hits: 0" in caplog.text
    assert "Embedding cache misses: 1" in caplog.text
    assert "Related paper embedding calls: 1" in caplog.text
    assert '  - "benchmark"' in markdown


def test_prepare_related_embedding_context_batches_missing_current_papers(monkeypatch, tmp_path):
    base_dir = tmp_path / "AI Papers"
    existing_dir = base_dir / "Papers"
    existing_dir.mkdir(parents=True)
    (existing_dir / "Existing.md").write_text(
        """---
title: "Existing"
tags:
  - "agent"
---
# Existing

## One Sentence Summary

Agent note.
""",
        encoding="utf-8",
    )

    calls = []

    def fake_create_embeddings(texts, *, model):
        calls.append(texts)
        return [
            SimpleNamespace(index=index, embedding=[float(index + 1), 0.0])
            for index, _ in enumerate(texts)
        ]

    monkeypatch.setattr(daily, "create_embeddings", fake_create_embeddings)
    ranked = [
        daily.RankedPaper(rank=1, score=0.8, paper={"title": "A", "summary": "agent"}),
        daily.RankedPaper(rank=2, score=0.7, paper={"title": "B", "summary": "agent"}),
    ]

    context = daily.prepare_related_embedding_context(
        ranked,
        base_dir,
        embedding_model="test-embedding",
        cache_path=tmp_path / ".cache" / "embeddings.json",
    )

    assert len(calls) == 1
    assert len(calls[0]) == 3
    assert context.cache_hits == 0
    assert context.cache_misses == 1
    assert context.embedding_calls == 1
    assert ranked[0].embedding == [2.0, 0.0]
    assert ranked[1].embedding == [3.0, 0.0]


def test_find_related_papers_uses_shared_research_axes_even_with_low_embedding_similarity(tmp_path):
    base_dir = tmp_path / "AI Papers"
    note_path = base_dir / "Papers" / "AgentBench.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text("# AgentBench\n\nAgent benchmark for tool use evaluation.", encoding="utf-8")
    note = {
        "path": str(note_path),
        "title": "AgentBench",
        "tags": "agent, benchmark",
        "abstract": "Agent benchmark for tool use evaluation.",
        "summary": "Compares agents with tool use benchmark tasks.",
        "text": "Agent benchmark tool use evaluation.",
    }
    ranked = daily.RankedPaper(
        rank=1,
        score=0.8,
        paper={
            "title": "Planning Agents",
            "summary": "A planning agent benchmark with tool use.",
        },
        analysis=PaperAnalysis(
            one_sentence_summary="Planning agent benchmark.",
            tldr="Agent benchmark.",
            key_contributions="Planning and tool use evaluation.",
            why_important="Important.",
            difference_from_previous_work="Different.",
            limitations="Limitations.",
            my_insight="Insight.",
            can_i_build_it=BuildPlan(),
            startup_idea="Startup.",
            project_idea="Project.",
            related_topics=["Agent", "Planning", "Benchmark", "Tool Use"],
            tags=["agent", "planning", "benchmark", "tool-use"],
        ),
        embedding=[1.0, 0.0],
    )

    related = daily.find_related_papers(
        ranked,
        base_dir,
        current_path=base_dir / "Papers" / "planning-agents.md",
        embedding_model="test-embedding",
        related_notes=[note],
        note_embeddings={str(note_path): [0.0, 1.0]},
        min_score=0.25,
    )

    assert related
    assert related[0].title == "AgentBench"
    assert "Agent" in related[0].axes
    assert "Benchmark" in related[0].axes
    assert "공통 연구 축" in related[0].common_point
    assert "축" in related[0].difference


def test_save_daily_markdown_recommends_past_papers_from_memory(monkeypatch, tmp_path):
    memory_path = tmp_path / ".cache" / "paper_memory.json"
    monkeypatch.setattr(daily, "PAPER_MEMORY_DB_PATH", memory_path)
    memory_path.parent.mkdir(parents=True)
    memory_path.write_text(
        """
[
  {
    "title": "Past Agent Memory",
    "tags": ["agent", "memory"],
    "embedding": [1.0, 0.0],
    "summary": "Past memory paper.",
    "paper_type": "Research",
    "project_idea": "Build a memory agent.",
    "link": "Papers/past-agent-memory",
    "source_path": "/vault/Papers/past-agent-memory.md",
    "note_type": "Paper"
  }
]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    ranked = [
        daily.RankedPaper(
            rank=1,
            score=0.9,
            paper={"title": "New Agent Memory", "summary": "agent memory planning"},
            analysis=PaperAnalysis(
                one_sentence_summary="New agent memory.",
                tldr="Agent memory.",
                key_contributions="Contributions.",
                why_important="Important.",
                difference_from_previous_work="Different.",
                limitations="Limitations.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(),
                startup_idea="Startup.",
                project_idea="Build a new memory agent.",
                related_topics=["Agent", "Memory"],
                tags=["agent", "memory"],
            ),
            embedding=[1.0, 0.0],
        )
    ]

    daily.save_daily_markdown(
        ranked,
        ranked,
        tmp_path,
        folder="AI Papers",
        target_date=date(2026, 6, 17),
        deep_read_k=0,
    )

    markdown = (tmp_path / "AI Papers" / "Papers" / "new-agent-memory.md").read_text(encoding="utf-8")
    memory_payload = memory_path.read_text(encoding="utf-8")

    assert "## Past Papers Memory" in markdown
    assert "[[Papers/past-agent-memory|Past Agent Memory]]" in markdown
    assert "- Project Idea: Build a memory agent." in markdown
    assert "New Agent Memory" in memory_payload
    assert "Build a new memory agent." in memory_payload


def test_format_pipeline_summary_includes_counts_and_output_path(tmp_path):
    stats = daily.PipelineStats(
        fetched_papers=100,
        candidate_papers=20,
        saved_daily_notes=1,
        saved_paper_notes=5,
        saved_deep_notes=2,
        embedding_calls=2,
        response_calls=7,
        cache_hits=12,
        cache_misses=5,
        runtime_seconds=142.34,
        output_dir=tmp_path / "AI Papers",
    )

    summary = daily.format_pipeline_summary(stats)

    assert "✅ Daily pipeline completed" in summary
    assert "- Fetched papers: 100" in summary
    assert "- Candidate papers: 20" in summary
    assert "- Saved paper notes: 5" in summary
    assert "- Deep read papers: 2" in summary
    assert "- Embedding calls: 2" in summary
    assert "- Response calls: 7" in summary
    assert "- Cache hits: 12" in summary
    assert "- Cache misses: 5" in summary
    assert "- Runtime: 142.3s" in summary
    assert f"- Output: {tmp_path / 'AI Papers'}" in summary


def test_infer_related_topics_adds_keyword_topics_and_categories():
    topics = daily.infer_related_topics(
        {
            "title": "Retrieval Agents for Robotics",
            "summary": "A policy improves embodied robot planning.",
            "categories": ["cs.AI"],
        },
        ["Agent"],
    )

    assert "RAG" in topics
    assert "Robotics" in topics
    assert "RL" in topics
    assert "cs.AI" in topics


def test_interest_level_from_similarity_score():
    assert daily.interest_level(0.75) == "⭐⭐⭐⭐⭐"
    assert daily.interest_level(0.60) == "⭐⭐⭐⭐☆"
    assert daily.interest_level(0.45) == "⭐⭐⭐☆☆"
    assert daily.interest_level(0.30) == "⭐⭐☆☆☆"
    assert daily.interest_level(0.29) == "⭐☆☆☆☆"


def test_save_daily_markdown_rejects_empty_top_papers(tmp_path):
    with pytest.raises(ValueError, match="No top papers"):
        daily.save_daily_markdown([], [], tmp_path, folder="papers", target_date=date(2026, 6, 16))


def test_save_daily_markdown_writes_daily_index_five_papers_and_two_deep_notes(tmp_path):
    ranked = []
    for index in range(20):
        topic = "Dataset" if index % 2 == 0 else "Robotics"
        ranked.append(
            daily.RankedPaper(
                rank=index + 1,
                score=0.8 - index * 0.01,
                paper={
                    "title": f"{topic} Paper {index}",
                    "summary": f"This paper is about {topic}.",
                    "authors": ["Ada"],
                    "categories": ["cs.AI"],
                },
                analysis=PaperAnalysis(
                    one_sentence_summary=f"{topic} summary.",
                    tldr="TLDR.",
                    key_contributions="Contributions.",
                    why_important="Important.",
                    difference_from_previous_work="Different.",
                    limitations="Limitations.",
                    my_insight="Insight.",
                    can_i_build_it=BuildPlan(),
                    startup_idea="Startup.",
                    project_idea="Project.",
                    related_topics=[topic, f"{topic} Evaluation", "cs.AI"],
                    tags=[topic, "evaluation", "ai"],
                ),
                deep_analysis="Deep analysis." if index < 2 else "",
            )
        )

    saved_paths = daily.save_daily_markdown(
        ranked,
        ranked[:5],
        tmp_path,
        folder="AI Papers",
        target_date=date(2025, 6, 16),
        deep_read_k=2,
    )

    assert len(saved_paths) == 8
    assert tmp_path / "AI Papers" / "Daily" / "2025-06-16.md" in saved_paths
    paper_notes = [path for path in saved_paths if path.parent.name == "Papers"]
    deep_notes = [path for path in saved_paths if path.parent.name == "Deep"]
    assert len(paper_notes) == 5
    assert len(deep_notes) == 2
    notes = paper_notes
    assert all(path.exists() and path.read_text(encoding="utf-8").strip() for path in notes)
    first = notes[0].read_text(encoding="utf-8")
    second = notes[1].read_text(encoding="utf-8")
    assert first.count("## Related Topics") == 1
    assert first.count("## My Insight") == 1
    assert "[[Dataset]]" in first
    assert "[[Robotics]]" in second
    assert first != second
    daily_index = (tmp_path / "AI Papers" / "Daily" / "2025-06-16.md").read_text(encoding="utf-8")
    assert "| Rank | Score | Interest Level | Title | Tags | Saved | Deep Read |" in daily_index
    assert "# 📰 Today's Headlines" in daily_index
    assert "# 📊 Topic Distribution" in daily_index
    assert "# 💎 Hidden Gem" in daily_index
    assert "# 🚀 This Week Build" in daily_index
    assert "# 📅 Research Timeline" in daily_index
    assert "## 🔥 Must Read Today" in daily_index
    assert "## 📈 Today's Research Trends" in daily_index
    assert "## 🏆 Editor's Pick" in daily_index
    assert "## 📚 Recommended Reading Order" in daily_index
    assert "## 💡 Today's Project" in daily_index
    assert "## Top20 Candidates" in daily_index
    assert daily_index.count("|") > 20
    assert "Yes" in daily_index
    assert "No" in daily_index


def test_render_daily_index_includes_interest_level_and_tags():
    ranked = [
        daily.RankedPaper(
            rank=1,
            score=0.76,
            paper={"arxiv_id": "2506.14683v1", "title": "Tagged Paper", "summary": "summary"},
            analysis=PaperAnalysis(
                one_sentence_summary="Summary.",
                tldr="TLDR.",
                key_contributions="Contributions.",
                why_important="Important.",
                difference_from_previous_work="Different.",
                limitations="Limitations.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(),
                startup_idea="Startup.",
                project_idea="Project.",
                related_topics=["Dataset"],
                tags=["dataset", "evaluation"],
            ),
        )
    ]

    markdown = daily.render_daily_index(ranked, date(2025, 6, 16), saved_ranks={1}, deep_ranks={1})

    assert "| Rank | Score | Interest Level | Title | Tags | Saved | Deep Read |" in markdown
    assert "# 🧭 Today's One Sentence" in markdown
    assert "# 📰 Today's Headlines" in markdown
    assert "* Research Trend:" in markdown
    assert "* Benchmark:" in markdown
    assert "* Application:" in markdown
    assert "# 📊 Topic Distribution" in markdown
    assert "Dataset (1)" in markdown
    assert "#" in markdown
    assert "# 💎 Hidden Gem" in markdown
    assert "- Why Read:" in markdown
    assert "- Future Potential:" in markdown
    assert "- Reason:" in markdown
    assert "# 🚀 This Week Build" in markdown
    assert "- First Step:" in markdown
    assert "# 📅 Research Timeline" in markdown
    assert "↓" in markdown or "[[Papers/2506.14683-tagged-paper|Tagged Paper]]" in markdown
    assert "## 🔥 Must Read Today" in markdown
    assert "Reason:" in markdown
    assert "Contributions." in markdown
    assert "Why it matters:" in markdown
    assert "## 📈 Today's Research Trends" in markdown
    assert "- Dataset: 1 papers" in markdown
    assert "## 🏆 Editor's Pick" in markdown
    assert "- Paper: [[Papers/2506.14683-tagged-paper|Tagged Paper]]" in markdown
    assert "- Novelty:" in markdown
    assert "- Impact: ⭐⭐⭐⭐⭐" in markdown
    assert "- Research Value:" in markdown
    assert "- Project Potential:" in markdown
    assert "## 📚 Recommended Reading Order" in markdown
    assert "## 💡 Today's Project" in markdown
    assert "- Project: Project." in markdown
    assert "- Tech Stack: Python, OpenAI API, arXiv metadata, Obsidian Markdown" in markdown
    assert "## Top20 Candidates" in markdown
    assert "⭐⭐⭐⭐⭐" in markdown
    assert "`dataset`" in markdown
    assert "[[Papers/2506.14683-tagged-paper|Tagged Paper]]" in markdown
    assert "| 1 | 0.760000 | ⭐⭐⭐⭐⭐ | [[Papers/2506.14683-tagged-paper|Tagged Paper]]" in markdown


def test_render_daily_index_newspaper_uses_natural_fallback_text():
    ranked = [
        daily.RankedPaper(
            rank=1,
            score=0.7,
            paper={"title": "Fallback Paper", "summary": "Agent benchmark"},
            analysis=daily.build_analysis_fallback({"title": "Fallback Paper", "summary": "Agent benchmark"}),
        )
    ]

    markdown = daily.render_daily_index(ranked, date(2025, 6, 16), saved_ranks={1}, deep_ranks=set())
    newspaper = markdown.split("## 🔥 Must Read Today", 1)[0]

    assert "# 📰 Today's Headlines" in newspaper
    assert "자동 생성 실패" not in markdown
    assert "요약 생성에 실패" not in markdown
    assert "판단할 수 없습니다" not in markdown


def test_daily_newspaper_sections_use_contributions_counts_and_buildable_project():
    ranked = [
        daily.RankedPaper(
            rank=1,
            score=0.8,
            paper={"title": "AgentEval", "summary": "agent benchmark evaluation"},
            analysis=PaperAnalysis(
                one_sentence_summary="AgentEval evaluates tool-use agents.",
                tldr="TLDR.",
                key_contributions="새로운 Agent 평가 프로토콜을 제안합니다.",
                why_important="Agent 평가 자동화에 중요합니다.",
                difference_from_previous_work="기존 벤치마크보다 도구 사용 실패를 더 잘 드러냅니다.",
                limitations="Small benchmark.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(
                    difficulty="⭐⭐⭐☆☆",
                    time_estimate="2 days",
                    need_gpu="No",
                    need_dataset="Yes",
                    undergraduate_friendly="Yes",
                    suggested_mini_project="Create 10 tool-use tasks and score agent failures.",
                ),
                startup_idea="Startup.",
                project_idea="Vague idea.",
                related_topics=["Agent", "Benchmark"],
                tags=["agent", "benchmark", "evaluation"],
            ),
        ),
        daily.RankedPaper(
            rank=6,
            score=0.55,
            paper={"title": "MemoryBench", "summary": "memory benchmark dataset"},
            analysis=PaperAnalysis(
                one_sentence_summary="MemoryBench tests long context memory.",
                tldr="TLDR.",
                key_contributions="장기 메모리 평가 데이터셋을 제안합니다.",
                why_important="Memory 평가에 중요합니다.",
                difference_from_previous_work="기존 평가보다 장기 문맥 실패를 더 잘 드러냅니다.",
                limitations="Small dataset.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(),
                startup_idea="Startup.",
                project_idea="Project.",
                related_topics=["Memory", "Benchmark"],
                tags=["memory", "benchmark", "dataset"],
            ),
        ),
    ]

    markdown = daily.render_daily_index(ranked, date(2025, 6, 16), saved_ranks={1}, deep_ranks=set())

    assert "# 🧭 Today's One Sentence" in markdown
    assert "Agent" in markdown
    assert "* Research Trend:" in markdown
    assert "* Benchmark:" in markdown
    assert "* Application:" in markdown
    assert "Benchmark (2)" in markdown
    assert "## 🔥 Must Read Today" in markdown
    assert "새로운 Agent 평가 프로토콜을 제안합니다." in markdown
    assert "- Why Read:" in markdown
    assert "- Novelty:" in markdown
    assert "- Future Potential:" in markdown
    assert "- Project: Mini Agent Evaluation Dashboard" in markdown
    assert "- Difficulty: ⭐⭐⭐☆☆" in markdown
    assert "- Time: 5 days" in markdown
    assert "- Tech Stack:" in markdown
    assert "- First Step: Create 10 tool-use tasks and score agent failures." in markdown


def test_daily_editor_timeline_and_reading_order_have_roles_and_reasons():
    ranked = [
        daily.RankedPaper(
            rank=1,
            score=0.9,
            paper={"title": "Foundation Paper", "summary": "core concepts"},
            analysis=PaperAnalysis(
                one_sentence_summary="Foundation summary.",
                tldr="TLDR.",
                key_contributions="문제 설정을 정리합니다.",
                why_important="Important.",
                difference_from_previous_work="Different.",
                limitations="Limitations.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(),
                startup_idea="Startup.",
                project_idea="Project.",
                related_topics=["Foundation"],
                tags=["foundation"],
            ),
        ),
        daily.RankedPaper(
            rank=2,
            score=0.8,
            paper={"title": "Application Paper", "summary": "agent tool software application"},
            analysis=PaperAnalysis(
                one_sentence_summary="Application summary.",
                tldr="TLDR.",
                key_contributions="도구 사용 응용을 제안합니다.",
                why_important="Important.",
                difference_from_previous_work="Different.",
                limitations="Limitations.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(),
                startup_idea="Startup.",
                project_idea="Project.",
                related_topics=["Agent"],
                tags=["agent", "tool"],
            ),
        ),
        daily.RankedPaper(
            rank=3,
            score=0.7,
            paper={"title": "Benchmark Paper", "summary": "benchmark evaluation metric"},
            analysis=PaperAnalysis(
                one_sentence_summary="Benchmark summary.",
                tldr="TLDR.",
                key_contributions="평가 기준을 제안합니다.",
                why_important="Important.",
                difference_from_previous_work="Different.",
                limitations="Limitations.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(),
                startup_idea="Startup.",
                project_idea="Project.",
                related_topics=["Benchmark"],
                tags=["benchmark", "evaluation"],
            ),
        ),
        daily.RankedPaper(
            rank=4,
            score=0.6,
            paper={"title": "Survey Paper", "summary": "survey taxonomy overview"},
            analysis=PaperAnalysis(
                one_sentence_summary="Survey summary.",
                tldr="TLDR.",
                key_contributions="분류 체계를 정리합니다.",
                why_important="Important.",
                difference_from_previous_work="Different.",
                limitations="Limitations.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(),
                startup_idea="Startup.",
                project_idea="Project.",
                related_topics=["Survey"],
                tags=["survey"],
            ),
        ),
    ]

    timeline = "\n".join(daily.render_research_timeline(ranked, saved_ranks={1, 2, 3, 4}))
    order = "\n".join(daily.render_recommended_reading_order(ranked, saved_ranks={1, 2, 3, 4}))

    assert "(Foundation)" in timeline
    assert "(Application)" in timeline
    assert "연결 이유:" in timeline
    assert order.index("Foundation:") < order.index("Benchmark:")
    assert order.index("Benchmark:") < order.index("Application:")
    assert order.index("Application:") < order.index("Survey:")


def test_render_daily_index_normalizes_category_tags_in_topic_distribution():
    ranked = [
        daily.RankedPaper(
            rank=1,
            score=0.7,
            paper={"title": "AI Paper", "summary": "agent", "categories": ["cs.AI"]},
            analysis=PaperAnalysis(
                one_sentence_summary="Summary.",
                tldr="TLDR.",
                key_contributions="Contributions.",
                why_important="Important.",
                difference_from_previous_work="Different.",
                limitations="Limitations.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(),
                startup_idea="Startup.",
                project_idea="Project.",
                related_topics=["cs-ai"],
                tags=["cs.AI"],
            ),
        ),
        daily.RankedPaper(
            rank=2,
            score=0.6,
            paper={"title": "NLP Paper", "summary": "language", "categories": ["cs.CL"]},
            analysis=PaperAnalysis(
                one_sentence_summary="Summary.",
                tldr="TLDR.",
                key_contributions="Contributions.",
                why_important="Important.",
                difference_from_previous_work="Different.",
                limitations="Limitations.",
                my_insight="Insight.",
                can_i_build_it=BuildPlan(),
                startup_idea="Startup.",
                project_idea="Project.",
                related_topics=["cs-cl"],
                tags=["cs.CL"],
            ),
        ),
    ]

    distribution = "\n".join(daily.render_topic_distribution(ranked))

    assert "AI (1)" in distribution
    assert "NLP (1)" in distribution
    assert "cs-ai #" not in distribution
    assert "cs-cl #" not in distribution


def test_render_daily_paper_markdown_adds_code_resources_from_abstract():
    ranked = daily.RankedPaper(
        rank=1,
        score=0.8,
        paper={
            "title": "Code Paper",
            "summary": (
                "Official code: https://github.com/example/code-paper. "
                "Related implementation: https://github.com/example/code-paper-reimpl. "
                "Ignore incomplete link https://github.com/. "
                "Project page: https://code-paper.example.com. "
                "Dataset: https://huggingface.co/datasets/example/code-paper-data"
            ),
        },
        analysis=PaperAnalysis(
            one_sentence_summary="Summary.",
            tldr="TLDR.",
            key_contributions="Contributions.",
            why_important="Important.",
            difference_from_previous_work="Different.",
            limitations="Limitations.",
            my_insight="Insight.",
            can_i_build_it=BuildPlan(),
            startup_idea="Startup.",
            project_idea="Project.",
            related_topics=["Coding Agent"],
            tags=["coding-agent"],
        ),
    )

    markdown = daily.render_daily_paper_markdown(ranked, date(2025, 6, 16))

    resources_section = markdown.split("## Code / Resources", 1)[1].split("## Related Papers", 1)[0]
    assert "## Code / Resources" in markdown
    assert "- Official Code: https://github.com/example/code-paper" in resources_section
    assert "- Related Implementation: https://github.com/example/code-paper-reimpl" in resources_section
    assert "https://github.com/." not in resources_section
    assert "- Project Page: https://code-paper.example.com" in resources_section
    assert "- HuggingFace: https://huggingface.co/datasets/example/code-paper-data" in resources_section
    assert "- Dataset: https://huggingface.co/datasets/example/code-paper-data" in resources_section


def test_render_deep_markdown_adds_evaluation_when_missing():
    ranked = daily.RankedPaper(
        rank=1,
        score=0.7,
        paper={"title": "Deep Paper", "summary": "summary"},
        deep_analysis="## 핵심 기여\n\n분석",
        paper_type="Research",
    )

    markdown = daily.render_deep_markdown(ranked, date(2025, 6, 16))

    assert 'paper_type: "Research"' in markdown
    assert "## Paper Type" in markdown
    assert "- Type: Research" in markdown
    assert "## For Me" in markdown
    assert "- Relevance to My Interests: ⭐⭐⭐⭐☆" in markdown
    assert "- How it connects to Agent/RAG/Reasoning/Coding Agent:" in markdown
    assert "## Can I Build It?" in markdown
    assert "- Need Fine-tuning:" in markdown
    assert "- Recommended Tech Stack:" in markdown
    assert "- Beginner Version:" in markdown
    assert "- Advanced Version:" in markdown
    assert "## Key Figure / Core Diagram" in markdown
    assert "```text" in markdown
    assert "## Reading Path" in markdown
    assert "## Evaluation" in markdown
    assert "- Novelty:" in markdown
    assert "- Worth Reading:" in markdown
    assert "## Next Action" in markdown
    assert "- [x] Build mini prototype" in markdown
    assert "- 추천 이유:" in markdown
    assert "## Research Position" in markdown
    assert "## Comparison Table" in markdown
    assert "| Paper | Planning | Memory | Tool | Benchmark |" in markdown
    assert "## If I Were Building This" in markdown
    assert "## Open Questions" in markdown
    assert "## Future Work Ideas" in markdown
    assert "## Related Papers" in markdown
    assert "No strong related papers found." in markdown


def test_render_deep_markdown_adds_code_resources_from_deep_analysis():
    ranked = daily.RankedPaper(
        rank=1,
        score=0.7,
        paper={"title": "Deep Code Paper", "summary": "summary"},
        deep_analysis="## Paper Type\n\n- Type: Research\n\nOfficial code: https://github.com/example/deep-code",
        paper_type="Research",
    )

    markdown = daily.render_deep_markdown(ranked, date(2025, 6, 16))

    assert "## Code / Resources" in markdown
    assert "- Official Code: https://github.com/example/deep-code" in markdown


def test_render_deep_markdown_includes_related_paper_reasons():
    ranked = daily.RankedPaper(
        rank=1,
        score=0.7,
        paper={"title": "Deep Paper", "summary": "summary"},
        deep_analysis="## Paper Type\n\n- Type: Research\n\n## Method\n\n분석",
        paper_type="Research",
    )
    related = [
        daily.RelatedPaper(
            title="Prior Agent Paper",
            link="Papers/Prior Agent Paper",
            score=0.8123,
            reason="둘 다 Agent 평가 프로토콜을 다룹니다.",
        )
    ]

    markdown = daily.render_deep_markdown(ranked, date(2025, 6, 16), related_papers=related)

    assert "- [[Papers/Prior Agent Paper|Prior Agent Paper]] (0.812)" in markdown
    assert "  - Common Point: 둘 다 Agent 평가 프로토콜을 다룹니다." in markdown
    assert "  - Difference:" in markdown


def test_enrich_ranked_papers_keeps_paper_with_summary_fallback(monkeypatch, tmp_path):
    def fake_analyze_abstract(title, abstract, *, model):
        if title == "bad":
            raise RuntimeError("summary failed")
        return daily.build_analysis_fallback({"title": title, "summary": abstract})

    monkeypatch.setattr(daily, "analyze_abstract", fake_analyze_abstract)
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
    assert enriched[0].analysis is not None
    assert "추가 확인이 필요한 논문입니다" in enriched[0].analysis.one_sentence_summary


def test_enrich_ranked_papers_keeps_top_paper_with_deep_read_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        daily,
        "analyze_abstract",
        lambda title, abstract, *, model: daily.build_analysis_fallback({"title": title, "summary": abstract}),
    )

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


def test_enrich_ranked_papers_limits_concurrency(monkeypatch, tmp_path):
    lock = threading.Lock()
    active_summary = 0
    max_active_summary = 0
    active_deep = 0
    max_active_deep = 0

    def fake_analyze_abstract(title, abstract, *, model):
        nonlocal active_summary, max_active_summary
        with lock:
            active_summary += 1
            max_active_summary = max(max_active_summary, active_summary)
        time.sleep(0.02)
        with lock:
            active_summary -= 1
        return daily.build_analysis_fallback({"title": title, "summary": abstract})

    class FakeDeepResult:
        paper_type = "Research"
        markdown = "## Paper Type\n\n- Type: Research\n\n## Method\n\nDeep"

    def fake_deep_read_pdf(title, pdf_url, work_dir, *, abstract, model):
        nonlocal active_deep, max_active_deep
        with lock:
            active_deep += 1
            max_active_deep = max(max_active_deep, active_deep)
        time.sleep(0.02)
        with lock:
            active_deep -= 1
        return FakeDeepResult()

    monkeypatch.setattr(daily, "analyze_abstract", fake_analyze_abstract)
    monkeypatch.setattr(daily, "deep_read_pdf", fake_deep_read_pdf)
    ranked = [
        daily.RankedPaper(
            rank=index + 1,
            score=0.9 - index * 0.01,
            paper={
                "title": f"paper {index}",
                "summary": "abstract",
                "pdf_url": f"http://example.com/{index}.pdf",
            },
        )
        for index in range(6)
    ]
    stats = daily.PipelineStats()

    enriched = daily.enrich_ranked_papers(
        ranked,
        tmp_path,
        summary_model="summary-model",
        deep_read_model="deep-model",
        deep_read_count=2,
        max_concurrency=3,
        stats=stats,
    )

    assert [item.rank for item in enriched] == [1, 2, 3, 4, 5, 6]
    assert max_active_summary <= 3
    assert max_active_deep <= 3
    assert enriched[0].deep_analysis
    assert enriched[1].deep_analysis
    assert not enriched[2].deep_analysis
    assert stats.response_calls == 8
