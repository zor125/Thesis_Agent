from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from deep_read import deep_read_pdf
from embedding import create_embeddings
from fetch import build_query_url, expanded_max_results, fetch_feed, filter_by_published_date, parse_feed
from memory_store import MemoryEntry, MemoryRecommendation, load_memory_db, recommend_memory_entries, upsert_memory_entries
from rank import cosine_similarity
from resources import extract_code_resources, render_code_resources
from save import (
    escape_markdown_table,
    escape_wikilink_label,
    normalize_paper_type,
    normalize_tags,
    paper_filename,
    render_front_matter,
    safe_filename,
)
from summarize import BuildPlan, PaperAnalysis, analyze_abstract


DEFAULT_CATEGORY = "cs.AI"
DEFAULT_MAX_RESULTS = 100
DEFAULT_TOP_K = 20
DEFAULT_DEEP_READ_COUNT = 3
RELATED_EMBEDDING_CACHE_PATH = Path(".cache") / "embeddings.json"
PAPER_MEMORY_DB_PATH = Path(".cache") / "paper_memory.json"
DEFAULT_VAULT_DIR = Path("obsidian")
DEFAULT_FOLDER = "papers"


@dataclass(frozen=True)
class RankedPaper:
    rank: int
    score: float
    paper: dict[str, Any]
    analysis: PaperAnalysis | None = None
    deep_analysis: str = ""
    paper_type: str = ""
    embedding: list[float] | None = None


@dataclass(frozen=True)
class RelatedPaper:
    title: str
    link: str
    score: float
    reason: str = ""
    common_point: str = ""
    difference: str = ""
    axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelatedEmbeddingContext:
    notes: list[dict[str, str]]
    note_embeddings: dict[str, list[float]]
    cache_hits: int = 0
    cache_misses: int = 0
    embedding_calls: int = 0


@dataclass
class PipelineStats:
    fetched_papers: int = 0
    candidate_papers: int = 0
    saved_daily_notes: int = 0
    saved_paper_notes: int = 0
    saved_deep_notes: int = 0
    embedding_calls: int = 0
    response_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    runtime_seconds: float = 0.0
    output_dir: Path | None = None


def run_daily(config_path: Path, *, today: date | None = None, stats: PipelineStats | None = None) -> list[Path]:
    stats = stats or PipelineStats()
    started_at = time.perf_counter()
    config = load_config(config_path)
    fetch_config = config.get("fetch", {})
    ranking_config = config.get("ranking", {})
    obsidian_config = config.get("obsidian", {})
    models_config = config.get("models", {})

    target_date = today or datetime.now(timezone.utc).date()
    category = str(fetch_config.get("category", DEFAULT_CATEGORY))
    max_results = int(fetch_config.get("max_results", DEFAULT_MAX_RESULTS))
    candidate_k = int(ranking_config.get("candidate_k", ranking_config.get("top_k", DEFAULT_TOP_K)))
    save_k = int(ranking_config.get("save_k", min(5, candidate_k)))
    deep_read_count = int(ranking_config.get("deep_read_k", DEFAULT_DEEP_READ_COUNT))
    interest = get_interest_sentence(config)
    vault_dir = get_obsidian_vault_path(obsidian_config)
    folder = str(obsidian_config.get("folder", DEFAULT_FOLDER))
    work_dir = Path(".paper_agent_cache") / target_date.isoformat()
    embedding_model = str(models_config.get("embedding", "text-embedding-3-small"))
    summary_model = str(models_config.get("summary", "gpt-5.5"))
    deep_read_model = str(models_config.get("deep_read", summary_model))
    stats.output_dir = vault_dir / folder

    papers = fetch_papers(category, target_date, max_results)
    stats.fetched_papers = len(papers)
    logging.info("Fetched papers: %s", len(papers))
    logging.debug("Embedding candidate papers: %s", count_embedding_candidates(papers))

    candidate_papers = rank_papers_by_interest(
        papers,
        interest,
        embedding_model=embedding_model,
        top_k=candidate_k,
        stats=stats,
    )
    stats.candidate_papers = len(candidate_papers)
    logging.debug("Candidate papers: %s", len(candidate_papers))
    if not candidate_papers:
        raise ValueError("Ranking produced no top papers. Check fetch results, embeddings, and config.")

    saved_papers = enrich_ranked_papers(
        candidate_papers[:save_k],
        work_dir,
        summary_model=summary_model,
        deep_read_model=deep_read_model,
        deep_read_count=deep_read_count,
        stats=stats,
    )
    saved_paths = save_daily_markdown(
        candidate_papers,
        saved_papers,
        vault_dir,
        folder=folder,
        target_date=target_date,
        interests=get_interest_topics(config),
        deep_read_k=deep_read_count,
        embedding_model=embedding_model,
        stats=stats,
    )
    stats.runtime_seconds = time.perf_counter() - started_at
    return saved_paths


def fetch_papers(category: str, target_date: date, max_results: int) -> list[dict[str, Any]]:
    raw_max_results = expanded_max_results(max_results)
    url = build_query_url(category, raw_max_results)
    raw_papers = parse_feed(fetch_feed(url, timeout=30))
    papers = filter_by_published_date(raw_papers, target_date)[:max_results]
    if not papers:
        logging.info("Fetched 0 papers after filtering by published date %s", target_date.isoformat())
    return [normalize_paper(paper) for paper in papers]


def rank_papers_by_interest(
    papers: Sequence[dict[str, Any]],
    interest: str,
    *,
    embedding_model: str,
    top_k: int,
    stats: PipelineStats | None = None,
) -> list[RankedPaper]:
    candidates: list[tuple[dict[str, Any], str]] = []
    for paper in papers:
        title = str(paper.get("title", "")).strip()
        summary = str(paper.get("summary", "")).strip()
        if not title and not summary:
            logging.warning("Skipping paper without title and summary: %s", paper.get("arxiv_id", "unknown"))
            continue
        candidates.append((paper, build_embedding_input(paper)))

    if not candidates:
        return []

    logging.debug("Embedding batch size: %s", len(candidates) + 1)
    try:
        if stats is not None:
            stats.embedding_calls += 1
        embedding_results = create_embeddings(
            [interest, *[embedding_input for _, embedding_input in candidates]],
            model=embedding_model,
        )
    except Exception as exc:
        logging.warning("Embedding batch failed: %s", exc)
        return []

    interest_embedding = embedding_results[0].embedding
    paper_embeddings = embedding_results[1:]
    scored: list[tuple[float, dict[str, Any], list[float]]] = []

    for (paper, _), embedding_result in zip(candidates, paper_embeddings):
        title = str(paper.get("title", "")).strip()
        try:
            score = cosine_similarity(interest_embedding, embedding_result.embedding)
        except Exception as exc:
            logging.warning("Skipping paper after ranking failure: %s (%s)", title, exc)
            continue

        scored.append((score, paper, embedding_result.embedding))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        RankedPaper(rank=rank, score=score, paper=paper, embedding=embedding)
        for rank, (score, paper, embedding) in enumerate(scored[:top_k], start=1)
    ]


def normalize_paper(paper: Any) -> dict[str, Any]:
    if isinstance(paper, dict):
        return paper
    if is_dataclass(paper):
        return asdict(paper)
    if hasattr(paper, "__dict__"):
        return dict(paper.__dict__)
    raise TypeError(f"Unsupported paper format: {type(paper).__name__}")


def build_embedding_input(paper: dict[str, Any]) -> str:
    title = str(paper.get("title", "")).strip()
    summary = str(paper.get("summary", "")).strip()
    return f"{title}\n\n{summary}".strip()


def count_embedding_candidates(papers: Sequence[dict[str, Any]]) -> int:
    return sum(1 for paper in papers if build_embedding_input(paper))


def enrich_ranked_papers(
    ranked_papers: Sequence[RankedPaper],
    work_dir: Path,
    *,
    summary_model: str,
    deep_read_model: str,
    deep_read_count: int,
    max_concurrency: int = 3,
    stats: PipelineStats | None = None,
) -> list[RankedPaper]:
    if not ranked_papers:
        return []

    max_workers = max(1, min(max_concurrency, 3, len(ranked_papers)))
    if stats is not None:
        stats.response_calls += len(ranked_papers)

    analyses: dict[int, PaperAnalysis] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_ranked_abstract, ranked, summary_model): index
            for index, ranked in enumerate(ranked_papers)
        }
        for future in as_completed(futures):
            index = futures[future]
            analyses[index] = future.result()

    deep_candidates = [ranked for ranked in ranked_papers if ranked.rank <= deep_read_count]
    if stats is not None:
        stats.response_calls += sum(1 for ranked in deep_candidates if ranked.paper.get("pdf_url"))

    deep_results: dict[int, tuple[str, str]] = {}
    if deep_candidates:
        deep_workers = max(1, min(max_concurrency, 3, len(deep_candidates)))
        with ThreadPoolExecutor(max_workers=deep_workers) as executor:
            futures = {
                executor.submit(read_ranked_pdf_deep, ranked, work_dir, deep_read_model): index
                for index, ranked in enumerate(ranked_papers)
                if ranked.rank <= deep_read_count
            }
            for future in as_completed(futures):
                index = futures[future]
                deep_results[index] = future.result()

    enriched = []
    for index, ranked in enumerate(ranked_papers):
        paper_type, deep_analysis_markdown = deep_results.get(
            index,
            (classify_paper_type_fallback(ranked.paper), ""),
        )
        enriched.append(
            RankedPaper(
                rank=ranked.rank,
                score=ranked.score,
                paper=ranked.paper,
                analysis=analyses[index],
                deep_analysis=deep_analysis_markdown,
                paper_type=paper_type,
                embedding=ranked.embedding,
            )
        )
    return enriched


def analyze_ranked_abstract(ranked: RankedPaper, summary_model: str) -> PaperAnalysis:
    paper = ranked.paper
    title = str(paper.get("title", "Untitled Paper"))
    abstract = str(paper.get("summary", ""))
    try:
        return analyze_abstract(title, abstract, model=summary_model)
    except Exception as exc:
        logging.warning("Using fallback summary after abstract summary failure: %s (%s)", title, exc)
        return build_analysis_fallback(paper)


def read_ranked_pdf_deep(ranked: RankedPaper, work_dir: Path, deep_read_model: str) -> tuple[str, str]:
    paper = ranked.paper
    title = str(paper.get("title", "Untitled Paper"))
    abstract = str(paper.get("summary", ""))
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        logging.warning("Using fallback deep analysis for %s: no pdf_url", title)
        paper_type = classify_paper_type_fallback(paper)
        return paper_type, build_deep_read_fallback(paper, "no pdf_url", paper_type=paper_type)
    try:
        deep_analysis = deep_read_pdf(
            title,
            str(pdf_url),
            work_dir / safe_filename(str(paper.get("arxiv_id", title))),
            abstract=abstract,
            model=deep_read_model,
        )
        return deep_analysis.paper_type, deep_analysis.markdown
    except Exception as exc:
        logging.warning("Using fallback deep analysis after deep read failure: %s (%s)", title, exc)
        paper_type = classify_paper_type_fallback(paper)
        return paper_type, build_deep_read_fallback(paper, exc, paper_type=paper_type)


def build_analysis_fallback(paper: dict[str, Any]) -> PaperAnalysis:
    title = str(paper.get("title", "Untitled Paper")).strip()
    abstract = str(paper.get("summary", "")).strip()
    return PaperAnalysis(
        one_sentence_summary=f"{title}의 초록과 메타데이터를 바탕으로 추가 확인이 필요한 논문입니다.",
        tldr=abstract or "초록이 없습니다.",
        key_contributions="초록 기준으로 핵심 기여를 직접 확인할 가치가 있습니다.",
        why_important="Useful reference for Agent and Coding Agent research.",
        difference_from_previous_work="기존 연구와의 차이는 원문 비교를 통해 확인하는 것이 좋습니다.",
        limitations="한계점은 원문 실험 설정과 평가 결과를 확인해야 합니다.",
        my_insight="관심사와 유사도가 높아 읽을 후보로 남겨둘 만합니다.",
        can_i_build_it=BuildPlan(
            difficulty="⭐⭐⭐☆☆",
            time_estimate="Unknown",
            need_gpu="Unknown",
            need_dataset="Unknown",
            undergraduate_friendly="Unknown",
            suggested_mini_project="초록을 읽고 핵심 방법을 작은 예제로 재현할 수 있는지 검토합니다.",
        ),
        startup_idea="논문의 문제 설정을 작은 도구나 평가 자동화 제품으로 바꿀 수 있는지 검토합니다.",
        project_idea="초록의 핵심 태스크를 기준으로 1주일 안에 가능한 미니 프로토타입을 설계합니다.",
        related_topics=infer_related_topics(paper),
        tags=infer_dynamic_tags(paper),
    )


def build_deep_read_fallback(paper: dict[str, Any], reason: object, *, paper_type: str = "Research") -> str:
    entry_url = str(paper.get("entry_url", "")).strip()
    pdf_url = str(paper.get("pdf_url", "")).strip()
    return "\n".join(
        [
            "## Paper Type",
            "",
            f"- Type: {paper_type}",
            "- 근거: PDF 심층 분석 실패로 제목과 초록 기반 fallback 분류를 사용했습니다.",
            "",
            "## 핵심 기여",
            "",
            f"_PDF 심층 분석에 실패했습니다: {reason}_",
            "",
            "## 방법론",
            "",
            "_PDF 분석 실패로 판단할 수 없습니다._",
            "",
            "## 실험 결과",
            "",
            "_PDF 분석 실패로 판단할 수 없습니다._",
            "",
            "## 한계점",
            "",
            "_PDF 분석 실패로 판단할 수 없습니다._",
            "",
            "## 구현 난이도",
            "",
            "_PDF 분석 실패로 판단할 수 없습니다._",
            "",
            "## 대학생 프로젝트 아이디어",
            "",
            "_PDF 분석 실패로 판단할 수 없습니다._",
            "",
            "## 스타트업 아이디어",
            "",
            "_PDF 분석 실패로 판단할 수 없습니다._",
            "",
            default_deep_evaluation(),
            "",
            default_next_action(),
            "",
            "## Links",
            "",
            f"- arXiv: {entry_url}",
            f"- PDF: {pdf_url}",
        ]
    )


def count_deep_read_papers(ranked_papers: Sequence[RankedPaper]) -> int:
    return sum(1 for ranked in ranked_papers if ranked.deep_analysis.strip())


def save_daily_markdown(
    candidate_papers: Sequence[RankedPaper],
    saved_papers: Sequence[RankedPaper],
    vault_dir: Path,
    *,
    folder: str,
    target_date: date,
    interests: Sequence[str] | None = None,
    deep_read_k: int = DEFAULT_DEEP_READ_COUNT,
    embedding_model: str = "text-embedding-3-small",
    stats: PipelineStats | None = None,
) -> list[Path]:
    if not candidate_papers:
        raise ValueError("No top papers to save. Refusing to write an empty Top20.md.")

    base_dir = vault_dir / folder
    daily_dir = base_dir / "Daily"
    papers_dir = base_dir / "Papers"
    deep_dir = base_dir / "Deep"
    daily_dir.mkdir(parents=True, exist_ok=True)
    papers_dir.mkdir(parents=True, exist_ok=True)
    deep_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    saved_by_rank = {ranked.rank: ranked for ranked in saved_papers}
    deep_by_rank = {
        ranked.rank: ranked
        for ranked in saved_papers
        if ranked.rank <= deep_read_k and ranked.deep_analysis.strip()
    }
    related_context = prepare_related_embedding_context(
        saved_papers,
        base_dir,
        embedding_model=embedding_model,
        stats=stats,
    )
    memory_entries = load_memory_db(PAPER_MEMORY_DB_PATH)

    daily_ranked_papers = [saved_by_rank.get(ranked.rank, ranked) for ranked in candidate_papers]

    index_path = daily_dir / f"{target_date.isoformat()}.md"
    index_path.write_text(
        render_daily_index(daily_ranked_papers, target_date, saved_ranks=set(saved_by_rank), deep_ranks=set(deep_by_rank)),
        encoding="utf-8",
    )
    saved_paths.append(index_path)
    if stats is not None:
        stats.saved_daily_notes = 1
    logging.debug("Saved daily index: %s", index_path)

    paper_note_count = 0
    for ranked in saved_papers:
        title = str(ranked.paper.get("title", "Untitled Paper"))
        output_path = papers_dir / f"{paper_note_filename(ranked.paper)}.md"
        memory_recommendations = recommend_memory_entries(
            memory_entries,
            ranked.embedding,
            exclude_paths={str(output_path)},
            limit=5,
        )
        related_papers = find_related_papers(
            ranked,
            base_dir,
            current_path=output_path,
            embedding_model=embedding_model,
            related_notes=related_context.notes,
            note_embeddings=related_context.note_embeddings,
        )
        try:
            output_path.write_text(
                render_daily_paper_markdown(
                    ranked,
                    target_date,
                    interests=interests or [],
                    related_papers=related_papers,
                    memory_recommendations=memory_recommendations,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logging.warning("Skipping paper note after save failure: %s (%s)", title, exc)
            continue
        saved_paths.append(output_path)
        paper_note_count += 1

    deep_note_count = 0
    for ranked in deep_by_rank.values():
        title = str(ranked.paper.get("title", "Untitled Paper"))
        output_path = deep_dir / f"{paper_note_filename(ranked.paper)}.md"
        memory_recommendations = recommend_memory_entries(
            memory_entries,
            ranked.embedding,
            exclude_paths={str(output_path)},
            limit=5,
        )
        related_papers = find_related_papers(
            ranked,
            base_dir,
            current_path=output_path,
            embedding_model=embedding_model,
            related_notes=related_context.notes,
            note_embeddings=related_context.note_embeddings,
        )
        try:
            output_path.write_text(
                render_deep_markdown(
                    ranked,
                    target_date,
                    related_papers=related_papers,
                    memory_recommendations=memory_recommendations,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logging.warning("Skipping deep note after save failure: %s (%s)", title, exc)
            continue
        saved_paths.append(output_path)
        deep_note_count += 1

    logging.debug("Saved paper notes: %s", paper_note_count)
    logging.debug("Deep read papers: %s", deep_note_count)
    memory_count = update_paper_memory_db(
        saved_papers,
        base_dir,
        paper_ranks=set(saved_by_rank),
        deep_ranks=set(deep_by_rank),
    )
    logging.debug("Updated paper memory entries: %s", memory_count)
    if stats is not None:
        stats.saved_paper_notes = paper_note_count
        stats.saved_deep_notes = deep_note_count
    logging.debug("Saved paper notes to: %s (%s files)", papers_dir, paper_note_count)
    logging.debug("Saved deep notes to: %s (%s files)", deep_dir, deep_note_count)
    return saved_paths


def render_daily_paper_markdown(
    ranked: RankedPaper,
    target_date: date,
    *,
    interests: Sequence[str] | None = None,
    related_papers: Sequence[RelatedPaper] | None = None,
    memory_recommendations: Sequence[MemoryRecommendation] | None = None,
) -> str:
    paper = ranked.paper
    title = str(paper.get("title", "Untitled Paper"))
    authors = paper.get("authors") if isinstance(paper.get("authors"), list) else []
    analysis = ranked.analysis or build_analysis_fallback(paper)
    categories = paper.get("categories") if isinstance(paper.get("categories"), list) else []
    dynamic_tags = normalize_tags(analysis.tags or infer_dynamic_tags(paper), max_tags=6)
    related_topics = normalize_related_topics(analysis.related_topics or infer_related_topics(paper))
    front_matter = {
        "title": title,
        "date": target_date.isoformat(),
        "rank": ranked.rank,
        "score": f"{ranked.score:.6f}",
        "arxiv_id": paper.get("arxiv_id", ""),
        "arxiv_url": paper.get("entry_url", ""),
        "pdf_url": paper.get("pdf_url", ""),
        "tags": normalize_tags(["paper-agent", "arxiv", "daily", *dynamic_tags]),
    }
    resources = extract_code_resources(*paper_resource_texts(paper))
    lines = [
        render_front_matter(front_matter),
        f"# {title}",
        "",
        "## Ranking",
        "",
        f"- Rank: {ranked.rank}",
        f"- Similarity: {ranked.score:.6f}",
        f"- Interest Level: {interest_level(ranked.score)}",
        "",
        "## Metadata",
        "",
        f"- Authors: {', '.join(str(author) for author in authors) if authors else 'Unknown'}",
        f"- arXiv ID: {paper.get('arxiv_id', '')}",
        f"- Published: {paper.get('published', '')}",
        f"- Categories: {', '.join(str(category) for category in categories)}",
        f"- arXiv: {paper.get('entry_url', '')}",
        f"- PDF: {paper.get('pdf_url', '')}",
        "",
        "## One Sentence Summary",
        "",
        analysis.one_sentence_summary or "_No one sentence summary available._",
        "",
        "## TL;DR",
        "",
        analysis.tldr or "_No TL;DR available._",
        "",
        *render_knowledge_card_sections(ranked, analysis, related_topics),
        "## Key Contributions",
        "",
        analysis.key_contributions or "_No key contributions available._",
        "",
        "## Why Important",
        "",
        analysis.why_important or "_No importance analysis available._",
        "",
        "## Difference From Previous Work",
        "",
        analysis.difference_from_previous_work or "_No comparison available._",
        "",
        "## Limitations",
        "",
        analysis.limitations or "_No limitations available._",
        "",
        "## My Insight",
        "",
        analysis.my_insight or "_No personal research insight available._",
        "",
        "## Startup Idea",
        "",
        analysis.startup_idea or "_No startup idea available._",
        "",
        "## Project Idea",
        "",
        analysis.project_idea or "_No project idea available._",
        "",
        "## Related Topics",
        "",
        *[f"- [[{escape_wikilink_label(topic)}]]" for topic in related_topics],
        "",
        *render_code_resources(resources),
        "## Related Papers",
        "",
        *render_related_papers(related_papers or []),
        "",
        "## Past Papers Memory",
        "",
        *render_memory_recommendations(memory_recommendations or []),
        "",
        "## Abstract",
        "<details>",
        "<summary>Original Abstract</summary>",
        "",
        str(paper.get("summary", "")).strip(),
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


KNOWLEDGE_CARD_INTERESTS = [
    "Agent",
    "Coding Agent",
    "RAG",
    "Memory",
    "Reasoning",
    "MCP",
    "Long Context",
    "Robotics",
]


def render_knowledge_card_sections(
    ranked: RankedPaper,
    analysis: PaperAnalysis,
    related_topics: Sequence[str],
) -> list[str]:
    can_build = analysis.can_i_build_it
    project = analysis.project_idea or can_build.suggested_mini_project
    return [
        "## Should I Read This?",
        "",
        f"* Target Audience: {target_audience_for_paper(ranked.paper, analysis)}",
        f"* Prerequisite Knowledge: {prerequisite_knowledge_for_paper(ranked.paper, related_topics)}",
        f"* Expected Gain: {analysis.why_important or analysis.one_sentence_summary or '논문의 핵심 아이디어를 빠르게 파악할 수 있습니다.'}",
        f"* Estimated Reading Time: {estimated_reading_time(ranked.score)}",
        "* Worth Reading:",
        f"  * Novelty: {novelty_rating(ranked)}",
        f"  * Research Value: {research_value_rating(ranked)}",
        f"  * Practical Impact: {practical_impact_rating(ranked)}",
        f"  * Project Potential: {project_potential_rating(ranked)}",
        "",
        "## Remember Only One Thing",
        "",
        analysis.one_sentence_summary or "이 논문의 핵심은 초록과 메타데이터를 직접 확인해야 합니다.",
        "",
        "## One Big Question",
        "",
        one_big_question_for_paper(ranked.paper, analysis),
        "",
        "## Connect To My Research",
        "",
        *render_interest_connections(ranked.paper, analysis, related_topics),
        "",
        "## Why Ranked Top5?",
        "",
        why_ranked_top5(ranked, analysis),
        "",
        "## Better Can I Build It?",
        "",
        f"* Difficulty: {can_build.difficulty}",
        f"* Time Estimate: {can_build.time_estimate}",
        f"* GPU Requirement: {can_build.need_gpu}",
        f"* Dataset Requirement: {can_build.need_dataset}",
        f"* Framework Recommendation: {framework_recommendation_for_paper(ranked.paper, analysis)}",
        f"* Beginner Version: {can_build.suggested_mini_project}",
        f"* Intermediate Version: {intermediate_project_version(project)}",
        f"* Advanced Version: {advanced_project_version(project)}",
        "",
        "## Better Startup Idea",
        "",
        *render_better_startup_idea(analysis),
        "",
        "## Better Project Idea",
        "",
        f"* Beginner: {can_build.suggested_mini_project}",
        f"* Intermediate: {intermediate_project_version(project)}",
        f"* Advanced: {advanced_project_version(project)}",
        "",
        "## Next Action",
        "",
        *render_paper_next_actions(ranked, analysis),
        "",
    ]


def target_audience_for_paper(paper: dict[str, Any], analysis: PaperAnalysis) -> str:
    text = paper_topic_text(paper, analysis)
    if contains_any(text, ["agent", "tool", "planning"]):
        return "LLM Agent와 AI 자동화 시스템을 만드는 연구자/개발자"
    if contains_any(text, ["retrieval", "rag", "memory"]):
        return "RAG, 메모리, 지식 검색 파이프라인을 개선하려는 개발자"
    if contains_any(text, ["robot", "robotics", "embodied"]):
        return "Robotics와 embodied AI에 관심 있는 연구자"
    if contains_any(text, ["code", "coding", "software", "program"]):
        return "Coding Agent와 소프트웨어 엔지니어링 자동화에 관심 있는 개발자"
    return "LLM 응용 연구 흐름을 빠르게 파악하려는 독자"


def prerequisite_knowledge_for_paper(paper: dict[str, Any], related_topics: Sequence[str]) -> str:
    topics = ", ".join(unique_preserving_order([*related_topics, *infer_related_topics(paper)])[:4])
    return topics or "LLM 기본 개념, 논문 초록 읽기, 간단한 Python 실험"


def estimated_reading_time(score: float) -> str:
    if score >= 0.75:
        return "15-25 minutes"
    if score >= 0.60:
        return "10-15 minutes"
    return "3-7 minutes"


def one_big_question_for_paper(paper: dict[str, Any], analysis: PaperAnalysis) -> str:
    text = paper_topic_text(paper, analysis)
    if contains_any(text, ["benchmark", "evaluation", "leaderboard"]):
        return "이 평가 방식은 실제 에이전트 성능 차이를 얼마나 공정하게 드러낼 수 있을까?"
    if contains_any(text, ["dataset", "corpus", "annotation"]):
        return "이 데이터셋은 어떤 새로운 실패 사례를 관찰 가능하게 만드는가?"
    if contains_any(text, ["coding", "code", "software", "program"]):
        return "이 접근은 AI Software Engineer가 실제 개발 루프에서 맡을 수 있는 일을 어디까지 넓히는가?"
    if contains_any(text, ["agent", "agents", "tool", "planning", "planner"]):
        return "이 논문은 에이전트가 계획, 도구 사용, 피드백을 더 안정적으로 연결하게 만드는가?"
    if contains_any(text, ["retrieval", "rag", "memory", "long context", "long-context"]):
        return "검색과 기억 구조를 바꾸면 장기 작업의 정확도와 비용이 실제로 얼마나 개선되는가?"
    if contains_any(text, ["reasoning", "logic", "chain-of-thought", "rl"]):
        return "이 방법은 추론 과정을 더 깊게 만들 뿐 아니라 검증 가능하게 만들 수 있는가?"
    if contains_any(text, ["robot", "robotics", "embodied"]):
        return "언어 모델의 계획 능력이 물리 환경의 불확실성까지 견딜 수 있는가?"
    topics = unique_preserving_order([*analysis.related_topics, *analysis.tags, *infer_related_topics(paper)])
    if topics:
        return f"{topics[0]} 연구에서 이 논문이 새로 열어주는 가장 작은 실험 단위는 무엇인가?"
    return "이 논문의 핵심 아이디어를 작은 재현 실험으로 줄이면 어떤 가설을 검증할 수 있을까?"


def render_interest_connections(
    paper: dict[str, Any],
    analysis: PaperAnalysis,
    related_topics: Sequence[str],
) -> list[str]:
    text = paper_topic_text(paper, analysis, related_topics)
    keywords = {
        "Agent": ["agent", "agents", "tool", "planning", "planner"],
        "Coding Agent": ["coding", "code", "program", "software", "developer"],
        "RAG": ["rag", "retrieval", "retrieve", "retriever", "augmented"],
        "Memory": ["memory", "memorization", "persistent", "state"],
        "Reasoning": ["reasoning", "reason", "logic", "planning", "chain-of-thought"],
        "MCP": ["mcp", "model context protocol", "tool protocol"],
        "Long Context": ["long context", "long-context", "context length", "context window"],
        "Robotics": ["robot", "robotics", "embodied", "manipulation"],
    }
    lines = []
    for interest in KNOWLEDGE_CARD_INTERESTS:
        lines.append(f"* {interest}: {research_connection_stars(text, keywords[interest])}")
    return lines


def research_connection_stars(text: str, keywords: Sequence[str]) -> str:
    matches = sum(1 for keyword in keywords if keyword in text)
    if matches >= 3:
        return "★★★★★"
    if matches == 2:
        return "★★★★☆"
    if matches == 1:
        return "★★★☆☆"
    return "★☆☆☆☆"


def practical_impact_rating(ranked: RankedPaper) -> str:
    analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
    text = paper_topic_text(ranked.paper, analysis)
    score = ranked.score
    if contains_any(text, ["system", "tool", "deployment", "automation", "software", "agent", "retrieval", "dataset"]):
        score = max(score, 0.60)
    return interest_level(score)


def why_ranked_top5(ranked: RankedPaper, analysis: PaperAnalysis) -> str:
    reasons = [f"similarity score {ranked.score:.6f}로 오늘 후보 중 {ranked.rank}위에 올랐습니다."]
    text = paper_topic_text(ranked.paper, analysis)
    if contains_any(text, ["agent", "coding", "software", "tool"]):
        reasons.append("Agent/Coding Agent 관심사와 직접 연결됩니다.")
    elif contains_any(text, ["rag", "retrieval", "memory", "long context"]):
        reasons.append("RAG, Memory, Long Context 연구 흐름과 연결됩니다.")
    elif contains_any(text, ["benchmark", "evaluation", "dataset"]):
        reasons.append("평가나 데이터셋 관점에서 후속 프로젝트의 기준점이 될 수 있습니다.")
    else:
        reasons.append("초록의 핵심 주제가 현재 관심사 문장과 충분히 가깝습니다.")
    return " ".join(reasons)


def render_paper_next_actions(ranked: RankedPaper, analysis: PaperAnalysis) -> list[str]:
    text = paper_topic_text(ranked.paper, analysis)
    deep_checked = "[x]" if ranked.deep_analysis.strip() else "[ ]"
    code_checked = "[x]" if contains_any(text, ["github", "code", "implementation"]) else "[ ]"
    reproduce_checked = "[x]" if contains_any(text, ["benchmark", "dataset", "evaluation", "experiment"]) else "[ ]"
    idea_checked = "[x]" if ranked.score >= 0.60 else "[ ]"
    return [
        f"* {deep_checked} Read Deep Note",
        f"* {code_checked} Clone Code",
        f"* {reproduce_checked} Reproduce",
        f"* {idea_checked} Add to Idea List",
    ]


def framework_recommendation_for_paper(paper: dict[str, Any], analysis: PaperAnalysis) -> str:
    text = paper_topic_text(paper, analysis)
    frameworks = ["Python"]
    if contains_any(text, ["agent", "tool", "planning"]):
        frameworks.extend(["OpenAI API", "LangGraph"])
    if contains_any(text, ["rag", "retrieval", "embedding"]):
        frameworks.extend(["FAISS", "sentence-transformers"])
    if contains_any(text, ["dataset", "benchmark", "evaluation"]):
        frameworks.extend(["pandas", "pytest", "Weights & Biases"])
    if contains_any(text, ["robot", "robotics"]):
        frameworks.extend(["ROS2", "PyBullet"])
    return ", ".join(unique_preserving_order(frameworks))


def render_better_startup_idea(analysis: PaperAnalysis) -> list[str]:
    startup = analysis.startup_idea or "논문 아이디어를 실제 업무 자동화 문제에 적용합니다."
    return [
        "* Target Customer: AI 기능을 제품이나 내부 업무에 적용하려는 팀",
        f"* Pain Point: {analysis.why_important or '기존 방식으로는 정확도, 비용, 운영 안정성을 동시에 맞추기 어렵습니다.'}",
        f"* MVP: {startup}",
        "* Revenue Model: SaaS 구독 또는 사용량 기반 과금",
        f"* Competitive Advantage: {analysis.difference_from_previous_work or '논문 기반 접근을 빠르게 제품화해 도메인별 워크플로에 맞출 수 있습니다.'}",
    ]


def intermediate_project_version(project: str) -> str:
    return f"{project}를 작은 데이터셋과 자동 평가 스크립트까지 포함한 재현 실험으로 확장합니다."


def advanced_project_version(project: str) -> str:
    return f"{project}를 실제 사용 시나리오, ablation, 모니터링까지 포함한 end-to-end 프로토타입으로 확장합니다."


def paper_topic_text(
    paper: dict[str, Any],
    analysis: PaperAnalysis,
    extra_topics: Sequence[str] | None = None,
) -> str:
    parts = [
        str(paper.get("title", "")),
        str(paper.get("summary", "")),
        analysis.one_sentence_summary,
        analysis.tldr,
        analysis.key_contributions,
        analysis.why_important,
        analysis.difference_from_previous_work,
        " ".join(analysis.related_topics),
        " ".join(analysis.tags),
        " ".join(extra_topics or []),
    ]
    return " ".join(part for part in parts if part).lower()


def contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def render_related_papers(related_papers: Sequence[RelatedPaper]) -> list[str]:
    if not related_papers:
        return ["No strong related papers found."]
    lines: list[str] = []
    for paper in related_papers:
        common_point = paper.common_point or paper.reason or "초록, 태그, 요약 embedding이 유사합니다."
        difference = paper.difference or "현재 논문과 기존 노트의 세부 방법, 실험 설정, 적용 범위는 원문 비교가 필요합니다."
        lines.extend(
            [
                f"- [[{paper.link}|{escape_wikilink_label(paper.title)}]] ({paper.score:.3f})",
                f"  - Axes: {', '.join(paper.axes) if paper.axes else 'No shared research axis detected'}",
                f"  - Common Point: {common_point}",
                f"  - Difference: {difference}",
            ]
        )
    return lines


def render_memory_recommendations(recommendations: Sequence[MemoryRecommendation]) -> list[str]:
    if not recommendations:
        return ["No past papers found in memory."]
    lines: list[str] = []
    for recommendation in recommendations:
        tags = ", ".join(recommendation.tags[:5]) if recommendation.tags else "no tags"
        paper_type = recommendation.paper_type or "Unknown"
        project_idea = recommendation.project_idea or "Project idea not recorded."
        lines.extend(
            [
                f"- [[{recommendation.link}|{escape_wikilink_label(recommendation.title)}]] ({recommendation.score:.3f})",
                f"  - Type: {paper_type}",
                f"  - Tags: {tags}",
                f"  - Project Idea: {project_idea}",
            ]
        )
    return lines


def paper_resource_texts(paper: dict[str, Any]) -> list[str]:
    texts = [
        str(paper.get("title", "")),
        str(paper.get("summary", "")),
        str(paper.get("entry_url", "")),
        str(paper.get("pdf_url", "")),
        str(paper.get("arxiv_id", "")),
    ]
    for key in ("authors", "categories", "links"):
        value = paper.get(key)
        if isinstance(value, list):
            texts.append(" ".join(str(item) for item in value))
        elif value:
            texts.append(str(value))
    return texts


def paper_note_filename(paper: dict[str, Any]) -> str:
    return paper_filename(
        str(paper.get("title", "Untitled Paper")),
        str(paper.get("arxiv_id", "")),
    )


def paper_note_link(paper: dict[str, Any]) -> str:
    title = str(paper.get("title", "Untitled Paper"))
    return f"[[Papers/{paper_note_filename(paper)}|{escape_wikilink_label(title)}]]"


def update_paper_memory_db(
    ranked_papers: Sequence[RankedPaper],
    base_dir: Path,
    *,
    paper_ranks: set[int],
    deep_ranks: set[int],
    memory_path: Path | None = None,
) -> int:
    memory_path = memory_path or PAPER_MEMORY_DB_PATH
    entries: list[MemoryEntry] = []
    for ranked in ranked_papers:
        if ranked.rank in paper_ranks:
            paper_path = base_dir / "Papers" / f"{paper_note_filename(ranked.paper)}.md"
            entries.append(build_memory_entry(ranked, paper_path, base_dir, note_type="Paper"))
        if ranked.rank in deep_ranks:
            deep_path = base_dir / "Deep" / f"{paper_note_filename(ranked.paper)}.md"
            entries.append(build_memory_entry(ranked, deep_path, base_dir, note_type="Deep"))
    return len(upsert_memory_entries(memory_path, entries))


def build_memory_entry(ranked: RankedPaper, note_path: Path, base_dir: Path, *, note_type: str) -> MemoryEntry:
    paper = ranked.paper
    analysis = ranked.analysis or build_analysis_fallback(paper)
    tags = normalize_tags(analysis.tags or infer_dynamic_tags(paper), max_tags=8)
    summary = "\n\n".join(
        part for part in [analysis.one_sentence_summary, analysis.tldr, str(paper.get("summary", "")).strip()] if part
    )
    return MemoryEntry(
        title=str(paper.get("title", "Untitled Paper")),
        tags=tags,
        embedding=ranked.embedding or [],
        summary=summary,
        paper_type=normalize_paper_type(ranked.paper_type or classify_paper_type_fallback(paper)),
        project_idea=analysis.project_idea or analysis.can_i_build_it.suggested_mini_project,
        link=obsidian_link_for_note(note_path, base_dir),
        source_path=str(note_path),
        note_type=note_type,
    )


def prepare_related_embedding_context(
    ranked_papers: Sequence[RankedPaper],
    base_dir: Path,
    *,
    embedding_model: str,
    cache_path: Path | None = None,
    stats: PipelineStats | None = None,
) -> RelatedEmbeddingContext:
    cache_path = cache_path or RELATED_EMBEDDING_CACHE_PATH
    notes = load_existing_research_notes(base_dir, exclude_path=base_dir / "__new_note__.md")
    cache = load_embedding_cache(cache_path)
    note_embeddings: dict[str, list[float]] = {}
    missing_notes: list[tuple[dict[str, str], str]] = []
    cache_hits = 0
    cache_misses = 0

    for note in notes:
        fingerprint = note_cache_fingerprint(note, embedding_model)
        cached = cache.get(note["path"])
        if cached and cached.get("fingerprint") == fingerprint and cached.get("embedding"):
            note_embeddings[note["path"]] = cached["embedding"]
            cache_hits += 1
        else:
            missing_notes.append((note, fingerprint))
            cache_misses += 1

    missing_ranked = [ranked for ranked in ranked_papers if not ranked.embedding] if notes else []
    inputs = [note["text"] for note, _ in missing_notes]
    inputs.extend(build_embedding_input(ranked.paper) for ranked in missing_ranked)
    embedding_calls = 0

    if inputs:
        embedding_calls = 1
        if stats is not None:
            stats.embedding_calls += 1
        embeddings = create_embeddings(inputs, model=embedding_model)
        note_count = len(missing_notes)
        for (note, fingerprint), embedding in zip(missing_notes, embeddings[:note_count]):
            cache[note["path"]] = {
                "fingerprint": fingerprint,
                "embedding": embedding.embedding,
            }
            note_embeddings[note["path"]] = embedding.embedding

        for ranked, embedding in zip(missing_ranked, embeddings[note_count:]):
            object.__setattr__(ranked, "embedding", embedding.embedding)

        if missing_notes:
            save_embedding_cache(cache_path, cache)

    if stats is not None:
        stats.cache_hits += cache_hits
        stats.cache_misses += cache_misses
    logging.debug("Embedding cache hits: %s", cache_hits)
    logging.debug("Embedding cache misses: %s", cache_misses)
    logging.debug("Related paper embedding calls: %s", embedding_calls)
    return RelatedEmbeddingContext(
        notes=notes,
        note_embeddings=note_embeddings,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        embedding_calls=embedding_calls,
    )


def find_related_papers(
    ranked: RankedPaper,
    base_dir: Path,
    *,
    current_path: Path,
    embedding_model: str,
    related_notes: Sequence[dict[str, str]] | None = None,
    note_embeddings: dict[str, list[float]] | None = None,
    limit: int = 5,
    min_score: float = 0.25,
) -> list[RelatedPaper]:
    existing_notes = list(related_notes) if related_notes is not None else load_existing_research_notes(base_dir, exclude_path=current_path)
    current_resolved = current_path.resolve()
    current_title = str(ranked.paper.get("title", "")).strip()
    existing_notes = [
        note for note in existing_notes
        if note["title"].strip().lower() != current_title.lower()
        and Path(note["path"]).resolve() != current_resolved
    ]
    if not existing_notes:
        return []

    current_embedding = ranked.embedding
    if current_embedding:
        try:
            if note_embeddings is None:
                note_embeddings = get_cached_note_embeddings(
                    existing_notes,
                    embedding_model=embedding_model,
                    cache_path=RELATED_EMBEDDING_CACHE_PATH,
                )
            scored = [
                (
                    combined_related_score(
                        cosine_similarity(current_embedding, note_embeddings[note["path"]]),
                        ranked,
                        note,
                    ),
                    note,
                )
                for note in existing_notes
                if note["path"] in note_embeddings
            ]
        except Exception as exc:
            logging.warning("Falling back to lexical related-paper matching: %s", exc)
            scored = lexical_related_scores(ranked, existing_notes)
    else:
        scored = lexical_related_scores(ranked, existing_notes)

    scored.sort(key=lambda item: item[0], reverse=True)
    related = []
    for score, note in scored:
        if score < min_score:
            continue
        related.append(
            RelatedPaper(
                title=note["title"],
                link=obsidian_link_for_note(Path(note["path"]), base_dir),
                score=score,
                reason=related_reason(ranked, note),
                common_point=related_common_point(ranked, note),
                difference=related_difference(ranked, note),
                axes=tuple(sorted(shared_research_axes(ranked, note))),
            )
        )
        if len(related) >= limit:
            break
    return related


def related_reason(ranked: RankedPaper, note: dict[str, str]) -> str:
    return related_common_point(ranked, note)


def related_common_point(ranked: RankedPaper, note: dict[str, str]) -> str:
    shared_axes = sorted(shared_research_axes(ranked, note))
    if shared_axes:
        axes = ", ".join(shared_axes)
        return f"공통 연구 축은 {axes}입니다. 두 논문 모두 이 축을 중심으로 문제 설정이나 시스템 설계를 해석할 수 있습니다."

    current_labels = related_labels_for_ranked_paper(ranked)
    note_labels = related_labels_for_existing_note(note)
    shared_labels = sorted(current_labels & note_labels)
    if shared_labels:
        labels = ", ".join(shared_labels[:3])
        return f"둘 다 {labels} 주제와 연결되어 있어 같은 연구 흐름에서 읽기 좋습니다."

    current_tokens = tokenize_related_text(build_embedding_input(ranked.paper))
    note_tokens = tokenize_related_text(note["text"])
    shared = sorted(current_tokens & note_tokens)
    useful_shared = [token for token in shared if len(token) > 3 and token not in RELATED_REASON_STOPWORDS][:3]
    if useful_shared:
        return f"공통 키워드({', '.join(useful_shared)})가 반복되어 문제 설정이나 평가 관점이 겹칩니다."
    return "초록, 태그, 요약 embedding이 유사합니다."


def related_difference(ranked: RankedPaper, note: dict[str, str]) -> str:
    current_axes = research_axes_for_ranked_paper(ranked)
    note_axes = research_axes_for_existing_note(note)
    current_only_axes = sorted(current_axes - note_axes)
    note_only_axes = sorted(note_axes - current_axes)
    if current_only_axes and note_only_axes:
        return f"현재 논문은 {', '.join(current_only_axes[:3])} 축이 더 강하고, 기존 노트는 {', '.join(note_only_axes[:3])} 축이 더 강합니다."
    if current_only_axes:
        return f"현재 논문은 {', '.join(current_only_axes[:3])} 축이 더 두드러집니다."
    if note_only_axes:
        return f"기존 노트는 {', '.join(note_only_axes[:3])} 축이 더 두드러집니다."

    current_labels = related_labels_for_ranked_paper(ranked)
    note_labels = related_labels_for_existing_note(note)
    current_only = sorted(current_labels - note_labels)
    note_only = sorted(note_labels - current_labels)
    if current_only and note_only:
        return f"현재 논문은 {', '.join(current_only[:2])} 쪽이 강하고, 기존 노트는 {', '.join(note_only[:2])} 쪽 맥락이 더 강합니다."
    if current_only:
        return f"현재 논문은 {', '.join(current_only[:2])} 관점이 더 두드러집니다."
    if note_only:
        return f"기존 노트는 {', '.join(note_only[:2])} 관점이 더 두드러집니다."
    return "주제는 가깝지만 방법론, 데이터, 실험 범위의 차이를 원문에서 비교해야 합니다."


RESEARCH_AXIS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Agent": (
        "agent",
        "agents",
        "autonomous agent",
        "llm agent",
        "multi-agent",
        "multi agent",
    ),
    "Memory": (
        "memory",
        "memories",
        "episodic",
        "semantic memory",
        "persistent",
        "reflection",
        "experience replay",
    ),
    "Planning": (
        "planning",
        "planner",
        "plan",
        "task decomposition",
        "decomposition",
        "trajectory",
        "reasoning-action",
    ),
    "Benchmark": (
        "benchmark",
        "benchmarks",
        "evaluation",
        "leaderboard",
        "metric",
        "metrics",
        "baseline",
        "swe-bench",
        "agentbench",
    ),
    "Tool Use": (
        "tool use",
        "tool-use",
        "tool",
        "tools",
        "function calling",
        "api call",
        "external tool",
        "action",
    ),
    "RAG": (
        "rag",
        "retrieval",
        "retriever",
        "retrieve",
        "retrieval-augmented",
        "augmented generation",
        "vector search",
    ),
    "MCP": (
        "mcp",
        "model context protocol",
        "context protocol",
    ),
    "Long Context": (
        "long context",
        "long-context",
        "longcontext",
        "context window",
        "context length",
        "long sequence",
        "long-context reasoning",
    ),
}


def combined_related_score(embedding_score: float, ranked: RankedPaper, note: dict[str, str]) -> float:
    axis_score = related_axis_score(ranked, note)
    label_score = related_label_score(ranked, note)
    if axis_score == 0:
        return (embedding_score * 0.55) + (label_score * 0.20)
    return (axis_score * 0.60) + (embedding_score * 0.30) + (label_score * 0.10)


def related_axis_score(ranked: RankedPaper, note: dict[str, str]) -> float:
    current_axes = research_axes_for_ranked_paper(ranked)
    note_axes = research_axes_for_existing_note(note)
    if not current_axes or not note_axes:
        return 0.0
    shared = current_axes & note_axes
    if not shared:
        return 0.0
    overlap = len(shared) / min(len(current_axes), len(note_axes))
    jaccard = len(shared) / len(current_axes | note_axes)
    return (overlap * 0.70) + (jaccard * 0.30)


def related_label_score(ranked: RankedPaper, note: dict[str, str]) -> float:
    current_labels = related_labels_for_ranked_paper(ranked)
    note_labels = related_labels_for_existing_note(note)
    if not current_labels or not note_labels:
        return 0.0
    shared = current_labels & note_labels
    return len(shared) / max(len(current_labels), len(note_labels)) if shared else 0.0


def shared_research_axes(ranked: RankedPaper, note: dict[str, str]) -> set[str]:
    return research_axes_for_ranked_paper(ranked) & research_axes_for_existing_note(note)


def research_axes_for_ranked_paper(ranked: RankedPaper) -> set[str]:
    analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
    text = paper_topic_text(ranked.paper, analysis)
    return research_axes_from_text(text)


def research_axes_for_existing_note(note: dict[str, str]) -> set[str]:
    return research_axes_from_text(note.get("text", ""))


def research_axes_from_text(text: str) -> set[str]:
    lowered = str(text or "").lower().replace("_", " ").replace("-", " ")
    axes = set()
    for axis, keywords in RESEARCH_AXIS_KEYWORDS.items():
        for keyword in keywords:
            normalized_keyword = keyword.lower().replace("_", " ").replace("-", " ")
            if normalized_keyword in lowered:
                axes.add(axis)
                break
    return axes


RELATED_REASON_STOPWORDS = {
    "paper",
    "papers",
    "study",
    "method",
    "model",
    "models",
    "using",
    "based",
    "with",
    "from",
    "this",
    "that",
    "summary",
    "abstract",
    "deep",
    "analysis",
}


def related_labels_for_ranked_paper(ranked: RankedPaper) -> set[str]:
    analysis = ranked.analysis
    labels: list[str] = []
    if analysis is not None:
        labels.extend(analysis.related_topics)
        labels.extend(analysis.tags)
    labels.extend(infer_related_topics(ranked.paper))
    labels.extend(infer_dynamic_tags(ranked.paper))
    return normalize_related_label_set(labels)


def related_labels_for_existing_note(note: dict[str, str]) -> set[str]:
    labels: list[str] = []
    labels.extend(note.get("tags", "").split(","))
    labels.extend(extract_title_topics(note.get("title", "")))
    labels.extend(extract_title_topics(note.get("summary", "")))
    labels.extend(extract_title_topics(note.get("abstract", "")))
    return normalize_related_label_set(labels)


def normalize_related_label_set(values: Sequence[str]) -> set[str]:
    labels = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        display = display_trend_label(cleaned)
        if len(display) > 2:
            labels.add(display)
    return labels


def load_existing_research_notes(base_dir: Path, *, exclude_path: Path) -> list[dict[str, str]]:
    notes = []
    exclude_resolved = exclude_path.resolve()
    for subdir in ("Papers", "Deep"):
        note_dir = base_dir / subdir
        if not note_dir.exists():
            continue
        for path in note_dir.glob("*.md"):
            if path.resolve() == exclude_resolved:
                continue
            text = path.read_text(encoding="utf-8")
            note = extract_note_fields(path, text)
            if note["title"]:
                notes.append(note)
    return notes


def extract_note_fields(path: Path, text: str) -> dict[str, str]:
    title = extract_markdown_title(text) or path.stem
    tags = ", ".join(extract_frontmatter_tags(text))
    abstract = extract_between(text, "<summary>Original Abstract</summary>", "</details>")
    summary = extract_sections_text(
        text,
        [
            "One Sentence Summary",
            "TL;DR",
            "Key Contributions",
            "Why Important",
            "Connect To My Research",
            "Research Position",
            "Comparison Table",
            "For Me",
            "Reading Path",
            "Deep Analysis",
        ],
    )
    return {
        "path": str(path),
        "title": title,
        "tags": tags,
        "abstract": abstract,
        "summary": summary,
        "text": "\n\n".join(part for part in [title, tags, summary, abstract] if part.strip()),
    }


def extract_markdown_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("title:"):
            return stripped.split(":", 1)[1].strip().strip('"')
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def extract_frontmatter_tags(text: str) -> list[str]:
    lines = text.splitlines()
    tags = []
    in_tags = False
    for line in lines:
        stripped = line.strip()
        if stripped == "tags:":
            in_tags = True
            continue
        if in_tags:
            if stripped.startswith("- "):
                tags.append(stripped[2:].strip().strip('"'))
                continue
            if stripped and not line.startswith(" "):
                break
    return tags


def extract_between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    after_start = text.split(start, 1)[1]
    return after_start.split(end, 1)[0].strip() if end in after_start else after_start.strip()


def extract_sections_text(text: str, section_names: Sequence[str]) -> str:
    sections = []
    for section_name in section_names:
        marker = f"## {section_name}"
        if marker not in text:
            continue
        body = text.split(marker, 1)[1]
        body = body.split("\n## ", 1)[0].strip()
        sections.append(body)
    return "\n\n".join(sections)


def get_cached_note_embeddings(
    notes: Sequence[dict[str, str]],
    *,
    embedding_model: str,
    cache_path: Path,
) -> dict[str, list[float]]:
    cache = load_embedding_cache(cache_path)
    missing_notes = []
    for note in notes:
        path = Path(note["path"])
        stat = path.stat()
        cache_key = note["path"]
        fingerprint = note_fingerprint(note["text"], stat.st_mtime_ns, stat.st_size, embedding_model)
        cached = cache.get(cache_key)
        if not cached or cached.get("fingerprint") != fingerprint:
            missing_notes.append((note, fingerprint))

    if missing_notes:
        embeddings = create_embeddings(
            [note["text"] for note, _ in missing_notes],
            model=embedding_model,
        )
        for (note, fingerprint), embedding in zip(missing_notes, embeddings):
            cache[note["path"]] = {
                "fingerprint": fingerprint,
                "embedding": embedding.embedding,
            }
        save_embedding_cache(cache_path, cache)

    return {
        note["path"]: cache[note["path"]]["embedding"]
        for note in notes
        if note["path"] in cache
    }


def load_embedding_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_embedding_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def note_cache_fingerprint(note: dict[str, str], embedding_model: str) -> str:
    path = Path(note["path"])
    stat = path.stat()
    return note_fingerprint(note["text"], stat.st_mtime_ns, stat.st_size, embedding_model)


def note_fingerprint(text: str, mtime_ns: int, size: int, embedding_model: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{embedding_model}:{mtime_ns}:{size}:{digest}"


def lexical_related_scores(
    ranked: RankedPaper,
    notes: Sequence[dict[str, str]],
) -> list[tuple[float, dict[str, str]]]:
    current_tokens = tokenize_related_text(build_embedding_input(ranked.paper))
    scored = []
    for note in notes:
        note_tokens = tokenize_related_text(note["text"])
        if not current_tokens or not note_tokens:
            score = 0.0
        else:
            score = len(current_tokens & note_tokens) / len(current_tokens | note_tokens)
        scored.append((combined_related_score(score, ranked, note), note))
    return scored


def tokenize_related_text(text: str) -> set[str]:
    return {
        token.lower()
        for token in "".join(char if char.isalnum() else " " for char in text).split()
        if len(token) > 2
    }


def obsidian_link_for_note(path: Path, base_dir: Path) -> str:
    relative = path.relative_to(base_dir).with_suffix("")
    return str(relative)


def render_deep_markdown(
    ranked: RankedPaper,
    target_date: date,
    *,
    related_papers: Sequence[RelatedPaper] | None = None,
    memory_recommendations: Sequence[MemoryRecommendation] | None = None,
) -> str:
    paper = ranked.paper
    title = str(paper.get("title", "Untitled Paper"))
    paper_type = normalize_paper_type(ranked.paper_type or classify_paper_type_fallback(paper))
    deep_body = ensure_deep_quality_sections(
        ensure_paper_type_in_deep_markdown(
            ranked.deep_analysis.strip() or "_No deep analysis available._",
            paper_type,
        ),
        ranked.score,
    )
    resource_lines = []
    if "## Code / Resources" not in deep_body:
        resources = extract_code_resources(*paper_resource_texts(paper), ranked.deep_analysis)
        resource_lines = render_code_resources(resources)
    front_matter = {
        "title": f"Deep Read - {title}",
        "date": target_date.isoformat(),
        "paper_type": paper_type,
        "rank": ranked.rank,
        "score": f"{ranked.score:.6f}",
        "arxiv_id": paper.get("arxiv_id", ""),
        "arxiv_url": paper.get("entry_url", ""),
        "pdf_url": paper.get("pdf_url", ""),
        "tags": normalize_tags(["paper-agent", "arxiv", "daily", "deep-read", *infer_dynamic_tags(paper)]),
    }
    return "\n".join(
        [
            render_front_matter(front_matter),
            f"# Deep Read - {title}",
            "",
            "## Source",
            "",
            f"- Paper: {paper_note_link(paper)}",
            f"- Rank: {ranked.rank}",
            f"- Similarity: {ranked.score:.6f}",
            f"- arXiv: {paper.get('entry_url', '')}",
            f"- PDF: {paper.get('pdf_url', '')}",
            "",
            "## Deep Analysis",
            "",
            deep_body,
            "",
            *resource_lines,
            "## Related Papers",
            "",
            *render_related_papers(related_papers or []),
            "",
            "## Past Papers Memory",
            "",
            *render_memory_recommendations(memory_recommendations or []),
            "",
            render_personal_research_notes_template(),
            "",
        ]
    )


def render_personal_research_notes_template() -> str:
    return "\n".join(
        [
            "---",
            "",
            "# 🧠 Personal Research Notes",
            "",
            "## 💡 My Idea",
            "",
            "(Leave blank for user)",
            "",
            "---",
            "",
            "## ❓ My Question",
            "",
            "(Leave blank for user)",
            "",
            "---",
            "",
            "## ⚠️ My Criticism",
            "",
            "(Leave blank for user)",
            "",
            "---",
            "",
            "## 🔬 My Experiment Idea",
            "",
            "(Leave blank for user)",
            "",
            "---",
            "",
            "## 🛠️ Can I Build It?",
            "",
            "- [ ] Yes",
            "- [ ] Maybe",
            "- [ ] No",
            "",
            "Reason:",
            "",
            "---",
            "",
            "## 🔗 Related Notes",
            "",
            "- [[ ]]",
            "",
            "- [[ ]]",
            "",
            "- [[ ]]",
            "",
            "---",
            "",
            "## ⭐ Personal Interest",
            "",
            "☆☆☆☆☆",
            "",
            "---",
        ]
    )


def classify_paper_type_fallback(paper: dict[str, Any]) -> str:
    text = f"{paper.get('title', '')}\n{paper.get('summary', '')}".lower()
    if any(keyword in text for keyword in ["survey", "review", "taxonomy", "overview"]):
        return "Survey"
    if any(keyword in text for keyword in ["benchmark", "leaderboard", "baseline", "metrics"]):
        return "Benchmark"
    if any(keyword in text for keyword in ["dataset", "corpus", "annotation", "annotated"]):
        return "Dataset"
    if any(keyword in text for keyword in ["system", "architecture", "deployment", "pipeline"]):
        return "System"
    if any(keyword in text for keyword in ["position", "perspective", "vision", "manifesto"]):
        return "Position"
    return "Research"


def ensure_paper_type_in_deep_markdown(markdown: str, paper_type: str) -> str:
    if "## Paper Type" in markdown:
        return markdown
    return f"## Paper Type\n\n- Type: {paper_type}\n- 근거: 저장 단계에서 보강한 fallback 분류입니다.\n\n{markdown}"


def ensure_deep_evaluation(markdown: str) -> str:
    if "## Evaluation" in markdown:
        return markdown
    return f"{markdown.rstrip()}\n\n{default_deep_evaluation()}"


def ensure_next_action(markdown: str) -> str:
    if "## Next Action" in markdown:
        return markdown
    return f"{markdown.rstrip()}\n\n{default_next_action()}"


def ensure_deep_quality_sections(markdown: str, score: float) -> str:
    markdown = ensure_for_me(markdown, score)
    markdown = ensure_deep_buildability(markdown)
    markdown = ensure_key_figure(markdown)
    markdown = ensure_reading_path(markdown)
    markdown = ensure_deep_evaluation(markdown)
    markdown = ensure_next_action(markdown)
    return ensure_research_notebook_sections(markdown)


def ensure_for_me(markdown: str, score: float) -> str:
    if "## For Me" in markdown:
        return markdown
    return f"{markdown.rstrip()}\n\n{default_for_me(score)}"


def ensure_deep_buildability(markdown: str) -> str:
    if "## Can I Build It?" in markdown:
        return markdown
    return f"{markdown.rstrip()}\n\n{default_deep_buildability()}"


def ensure_key_figure(markdown: str) -> str:
    if "## Key Figure / Core Diagram" in markdown:
        return markdown
    return f"{markdown.rstrip()}\n\n{default_key_figure()}"


def ensure_reading_path(markdown: str) -> str:
    if "## Reading Path" in markdown:
        return markdown
    return f"{markdown.rstrip()}\n\n{default_reading_path()}"


def ensure_research_notebook_sections(markdown: str) -> str:
    for heading, default_section in [
        ("## Research Position", default_research_position),
        ("## Comparison Table", default_comparison_table),
        ("## If I Were Building This", default_if_i_were_building_this),
        ("## Open Questions", default_open_questions),
        ("## Future Work Ideas", default_future_work_ideas),
    ]:
        if heading not in markdown:
            markdown = f"{markdown.rstrip()}\n\n{default_section()}"
    return markdown


def default_for_me(score: float) -> str:
    revisit = "Yes" if score >= 0.45 else "No"
    return "\n".join(
        [
            "## For Me",
            "",
            f"- Relevance to My Interests: {interest_level(score)}",
            "- Why it matters to me: 관심사 임베딩과 유사도가 높아 Agent/RAG/Reasoning 관점에서 후속 확인 가치가 있습니다.",
            "- How it connects to Agent/RAG/Reasoning/Coding Agent: 초록과 PDF 분석을 바탕으로 도구 사용, 검색, 추론, 자동화 흐름과의 연결 가능성을 검토해야 합니다.",
            f"- Should I revisit this later: {revisit}",
        ]
    )


def default_deep_buildability() -> str:
    return "\n".join(
        [
            "## Can I Build It?",
            "",
            "- Difficulty: ⭐⭐⭐☆☆",
            "- Time Estimate: 3-7 days for a small prototype",
            "- Need GPU: Unknown",
            "- Need Dataset: Unknown",
            "- Need Fine-tuning: Unknown",
            "- Solo Developer Possible: Yes, if scoped to a minimal reproduction",
            "- Recommended Tech Stack: Python, OpenAI API, arXiv/PDF utilities, lightweight evaluation scripts",
            "- Beginner Version: 초록 기반 데모와 작은 샘플 평가부터 구현합니다.",
            "- Advanced Version: 논문 방법론의 핵심 실험을 재현하고 자동 평가 파이프라인으로 확장합니다.",
        ]
    )


def default_key_figure() -> str:
    return "\n".join(
        [
            "## Key Figure / Core Diagram",
            "",
            "```text",
            "Paper Question",
            "↓",
            "Proposed Method / Taxonomy",
            "↓",
            "Experiments or Evidence",
            "↓",
            "Findings",
            "↓",
            "Practical Takeaway",
            "```",
        ]
    )


def default_reading_path() -> str:
    return "\n".join(
        [
            "## Reading Path",
            "",
            "- 논문이 다루는 핵심 태스크 정의",
            "- 사용된 데이터셋 또는 벤치마크",
            "- 비교 대상 baseline 방법",
            "- 평가 metric과 재현 조건",
            "- 관련 Agent/RAG/Reasoning 선행 연구",
        ]
    )


def default_research_position() -> str:
    return "\n".join(
        [
            "## Research Position",
            "",
            "이 논문은 Agent/RAG/Reasoning 연구가 단순 모델 성능 비교에서 실제 시스템 설계, 평가 프로토콜, 재현 가능한 도구 체계로 이동하는 흐름 위에 놓입니다. 정확한 연구사적 위치는 인용 관계를 추가 확인해야 하지만, 현재 노트 기준으로는 Planning, Memory, Tool Use, Benchmark 중 어떤 축을 강화하는지 판단하는 기준점으로 사용할 수 있습니다.",
        ]
    )


def default_comparison_table() -> str:
    return "\n".join(
        [
            "## Comparison Table",
            "",
            "| Paper | Planning | Memory | Tool | Benchmark |",
            "| --- | --- | --- | --- | --- |",
            "| ReAct | 명시적 reasoning-action 루프를 제공합니다. | 장기 기억은 제한적입니다. | 외부 도구 호출을 핵심 구성으로 사용합니다. | 주로 태스크별 평가에 의존합니다. |",
            "| Reflexion | 실패 후 재계획과 자기 반성을 강조합니다. | verbal memory를 통해 경험을 누적합니다. | 도구 사용 자체보다 피드백 루프가 중심입니다. | 반복 성능 개선을 평가합니다. |",
            "| SWE-bench / AgentBench | 실제 작업 단위의 계획 능력을 간접 평가합니다. | 메모리 설계는 시스템별로 다릅니다. | 코드 실행, 검색, 환경 상호작용을 포함할 수 있습니다. | 에이전트 성능을 비교하는 기준점입니다. |",
        ]
    )


def default_if_i_were_building_this() -> str:
    return "\n".join(
        [
            "## If I Were Building This",
            "",
            "- 먼저 논문 주장을 Planning, Memory, Tool, Benchmark 네 모듈로 분해하고, 각 모듈을 독립적으로 끄고 켤 수 있게 설계합니다.",
            "- 최소 재현 버전은 작은 공개 데이터셋과 5~10개 대표 태스크로 시작해 end-to-end 실패 사례를 빠르게 수집합니다.",
            "- 평가 코드는 논문 주장과 직접 연결되는 metric 중심으로 분리하고, baseline과 ablation을 같은 스크립트에서 반복 실행할 수 있게 만듭니다.",
            "- 구현체는 프롬프트, 메모리 저장소, 도구 인터페이스, 평가 harness를 분리해 후속 실험이 쉬운 구조로 만듭니다.",
        ]
    )


def default_open_questions() -> str:
    return "\n".join(
        [
            "## Open Questions",
            "",
            "- Planning, Memory, Tool Use 중 실제 성능 향상에 가장 크게 기여하는 요소는 무엇인가?",
            "- 이 방법은 더 다양한 도메인, 긴 작업, noisy input에서도 안정적으로 유지되는가?",
            "- 성능 향상이 모델 규모, 프롬프트 설계, 데이터 구성 중 어느 요인에 가장 크게 의존하는가?",
            "- 실패 사례는 어떤 패턴을 보이며, 실제 사용 환경에서 치명적인 failure mode로 이어지는가?",
            "- 더 단순한 baseline과 비교해도 비용 대비 충분히 의미 있는 개선인가?",
        ]
    )


def default_future_work_ideas() -> str:
    return "\n".join(
        [
            "## Future Work Ideas",
            "",
            "- 이 논문의 핵심 방법을 작은 Agent/RAG/Coding Agent 파이프라인에 붙여 모듈별 ablation을 수행합니다.",
            "- 실패 사례를 수집해 Planning 실패, Memory 실패, Tool 실패, Benchmark mismatch로 분류하는 evaluation checklist를 만듭니다.",
            "- 공개 구현이나 pseudo-code를 기반으로 1주일짜리 reproducibility report를 작성합니다.",
            "- 관련 논문 3편과 같은 태스크에서 비교해 연구 위치를 더 명확히 드러내는 mini benchmark를 구성합니다.",
        ]
    )


def default_deep_evaluation() -> str:
    return "\n".join(
        [
            "## Evaluation",
            "",
            "- Novelty: ⭐⭐⭐☆☆ - PDF 분석이 실패했거나 평가 근거가 부족해 중간 수준으로 임시 평가합니다.",
            "- Impact: ⭐⭐⭐☆☆ - 실제 영향력은 본문과 인용 맥락을 추가 확인해야 합니다.",
            "- Practicality: ⭐⭐⭐☆☆ - 구현 가능성은 방법론 세부사항 확인 전까지 보수적으로 평가합니다.",
            "- Reproducibility: ⭐⭐☆☆☆ - 코드, 데이터셋, 실험 설정 확인 없이는 재현성을 낮게 봅니다.",
            "- Worth Reading: ⭐⭐⭐☆☆ - 관심 주제와 연결되지만 원문 확인이 필요합니다.",
        ]
    )


def default_next_action() -> str:
    return "\n".join(
        [
            "## Next Action",
            "",
            "- [ ] Read full paper",
            "- [ ] Find GitHub implementation",
            "- [ ] Search related papers",
            "- [x] Build mini prototype",
            "- [x] Write blog summary",
            "- [ ] Skip",
            "- 추천 이유: 1주일 안에 작은 재현 실험과 요약 글 작성까지 완료할 수 있는 실행 가능한 액션입니다.",
        ]
    )


def infer_related_topics(paper: dict[str, Any], interests: Sequence[str] | None = None) -> list[str]:
    categories = paper.get("categories", [])
    category_text = " ".join(str(category) for category in categories if category) if isinstance(categories, list) else ""
    text = " ".join(
        [
            str(paper.get("title", "")),
            str(paper.get("summary", "")),
            category_text,
        ]
    ).lower()
    topic_keywords = {
        "RAG": ["rag", "retrieval", "retrieve", "retriever", "augmented generation"],
        "Agent": ["agent", "agents", "tool use", "planning"],
        "Reasoning": ["reasoning", "reason", "chain-of-thought", "planning"],
        "Coding Agent": ["code", "coding", "programming", "software"],
        "Multi Agent": ["multi-agent", "multi agent", "multiagent"],
        "Long Context": ["long context", "context length", "long-context"],
        "LLM Memory": ["memory", "persistent memory"],
        "Robotics": ["robot", "robotics", "embodied"],
        "RL": ["reinforcement learning", "rl", "policy", "reward"],
    }

    topics: list[str] = []
    for topic, keywords in topic_keywords.items():
        if any(keyword in text for keyword in keywords):
            topics.append(topic)

    if isinstance(categories, list):
        topics.extend(str(category) for category in categories if category)

    if len(unique_preserving_order(topics)) < 3:
        topics.extend(extract_title_topics(str(paper.get("title", ""))))
    if len(unique_preserving_order(topics)) < 3:
        topics.extend(extract_title_topics(str(paper.get("summary", ""))))

    return normalize_related_topics(topics)


def extract_title_topics(title: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "based",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "towards",
        "using",
        "with",
    }
    words = [
        word
        for word in "".join(char if char.isalnum() else " " for char in title).split()
        if len(word) > 2 and word.lower() not in stopwords
    ]
    topics = []
    for word in words:
        topics.append(word.upper() if word.isupper() else word.title())
    return topics


def infer_dynamic_tags(paper: dict[str, Any]) -> list[str]:
    topics = infer_related_topics(paper)
    categories = paper.get("categories", [])
    category_tags = [str(category) for category in categories] if isinstance(categories, list) else []
    return normalize_tags([*topics, *category_tags], max_tags=6)


def normalize_related_topics(values: Sequence[str]) -> list[str]:
    return unique_preserving_order(values)[:6]


def interest_level(score: float) -> str:
    if score >= 0.75:
        return "⭐⭐⭐⭐⭐"
    if score >= 0.60:
        return "⭐⭐⭐⭐☆"
    if score >= 0.45:
        return "⭐⭐⭐☆☆"
    if score >= 0.30:
        return "⭐⭐☆☆☆"
    return "⭐☆☆☆☆"


def unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def render_daily_index(
    ranked_papers: Sequence[RankedPaper],
    target_date: date,
    *,
    saved_ranks: set[int] | None = None,
    deep_ranks: set[int] | None = None,
) -> str:
    saved_ranks = saved_ranks or set()
    deep_ranks = deep_ranks or set()
    front_matter = {
        "title": f"{target_date.isoformat()} Daily Paper Candidates",
        "date": target_date.isoformat(),
        "tags": ["paper-agent", "ranking", "daily"],
    }
    lines = [
        render_front_matter(front_matter),
        f"# {target_date.isoformat()} Daily Paper Candidates",
        "",
        *render_research_newspaper_sections(ranked_papers, saved_ranks=saved_ranks),
        "",
        *render_daily_briefing_sections(ranked_papers, saved_ranks=saved_ranks, deep_ranks=deep_ranks),
        "",
        "## Top20 Candidates",
        "",
        "| Rank | Score | Interest Level | Title | Tags | Saved | Deep Read |",
        "| ---: | ---: | :---: | --- | --- | :---: | :---: |",
    ]
    for ranked in ranked_papers:
        title = str(ranked.paper.get("title", "Untitled Paper"))
        analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
        tags = ", ".join(f"`{tag}`" for tag in normalize_tags(analysis.tags or infer_dynamic_tags(ranked.paper), max_tags=6))
        if ranked.rank in saved_ranks:
            title_text = paper_note_link(ranked.paper)
        else:
            title_text = escape_markdown_table(title)
        lines.append(
            f"| {ranked.rank} | {ranked.score:.6f} | {interest_level(ranked.score)} | "
            f"{title_text} | {tags} | {yes_no(ranked.rank in saved_ranks)} | {yes_no(ranked.rank in deep_ranks)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_research_newspaper_sections(
    ranked_papers: Sequence[RankedPaper],
    *,
    saved_ranks: set[int],
) -> list[str]:
    return [
        "# 🧭 Today's One Sentence",
        "",
        render_todays_one_sentence(ranked_papers),
        "",
        "# 📰 Today's Headlines",
        "",
        *render_todays_headlines(ranked_papers),
        "",
        "# 📊 Topic Distribution",
        "",
        *render_topic_distribution(ranked_papers),
        "",
        "# 💎 Hidden Gem",
        "",
        *render_hidden_gem(ranked_papers, saved_ranks=saved_ranks),
        "",
        "# 🚀 This Week Build",
        "",
        *render_this_week_build(ranked_papers[:5]),
        "",
        "# 📅 Research Timeline",
        "",
        *render_research_timeline(ranked_papers[:5], saved_ranks=saved_ranks),
    ]


def render_todays_headlines(ranked_papers: Sequence[RankedPaper]) -> list[str]:
    if not ranked_papers:
        return [
            "* Research Trend: AI 연구 후보 수집 대기",
            "* Benchmark: Agent Evaluation 흐름 점검 필요",
            "* Application: Coding Agent 프로젝트 아이디어 탐색 지속",
        ]

    return [
        f"* Research Trend: {research_trend_headline(ranked_papers)}",
        f"* Benchmark: {benchmark_headline(ranked_papers)}",
        f"* Application: {application_headline(ranked_papers)}",
    ]


def research_trend_headline(ranked_papers: Sequence[RankedPaper]) -> str:
    trend_counts = collect_trend_counts(ranked_papers)
    top_labels = [label for label, _ in trend_counts[:3]]
    if not top_labels:
        return "Top20이 여러 주제로 분산되어 있어 새로운 연구 흐름을 직접 비교할 필요가 있습니다."
    contribution = trend_contribution_phrase(top_labels[0], ranked_papers)
    return f"{', '.join(top_labels)} 흐름이 두드러지며, 핵심은 {contribution}입니다."


def benchmark_headline(ranked_papers: Sequence[RankedPaper]) -> str:
    paper = first_paper_matching(ranked_papers, ["benchmark", "evaluation", "metric", "leaderboard"])
    if paper is None:
        return "명확한 벤치마크 논문은 적지만 Top20 전반에서 평가 기준 확인이 필요합니다."
    analysis = paper.analysis or build_analysis_fallback(paper.paper)
    return f"{short_title(paper)}가 {contribution_summary(analysis)}을 통해 평가 기준을 강화합니다."


def application_headline(ranked_papers: Sequence[RankedPaper]) -> str:
    paper = first_paper_matching(ranked_papers, ["agent", "tool", "coding", "software", "rag", "robot", "system"])
    if paper is None:
        return "응용 관점에서는 아직 직접 구현 가능한 프로젝트 후보를 선별해야 합니다."
    analysis = paper.analysis or build_analysis_fallback(paper.paper)
    return f"{short_title(paper)}는 {practical_project_name(paper, analysis)}로 축소해 구현해볼 수 있습니다."


def first_paper_matching(ranked_papers: Sequence[RankedPaper], keywords: Sequence[str]) -> RankedPaper | None:
    for ranked in ranked_papers:
        analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
        if contains_any(paper_topic_text(ranked.paper, analysis), keywords):
            return ranked
    return ranked_papers[0] if ranked_papers else None


def headline_for_trend(label: str, ranked_papers: Sequence[RankedPaper] | None = None) -> str:
    contribution = trend_contribution_phrase(label, ranked_papers or [])
    headlines = {
        "AI": f"AI 연구가 {contribution} 중심으로 확장",
        "NLP": f"NLP 연구가 {contribution} 쪽으로 이동",
        "Machine Learning": f"Machine Learning 연구가 {contribution}을 강화",
        "Software Engineering": f"AI Software Engineer 연구가 {contribution}에 집중",
        "Information Retrieval": f"RAG와 검색 연구가 {contribution}을 재부상",
        "Agent": f"Agent 연구가 {contribution}을 중심으로 활발",
        "RAG": f"RAG 파이프라인 연구가 {contribution}을 강화",
        "Reasoning": f"Reasoning 연구가 {contribution}으로 진화",
        "Coding Agent": f"AI Software Engineer 연구가 {contribution}으로 구체화",
        "Multi Agent": f"Multi Agent 연구가 {contribution}을 확대",
        "Long Context": f"Long Context 연구가 {contribution}을 고도화",
        "Memory": f"LLM Memory 연구가 {contribution}으로 실용화",
        "Robotics": f"Robotics와 LLM Planning 연구가 {contribution}으로 연결",
        "Benchmark": f"Benchmark 연구가 {contribution}을 기준점으로 제시",
        "Evaluation": f"Evaluation 연구가 {contribution}을 정교화",
        "Dataset": f"AI Dataset 연구가 {contribution}을 새롭게 정의",
    }
    return headlines.get(label, f"{label} 연구가 {contribution} 흐름을 강화")


def trend_contribution_phrase(label: str, ranked_papers: Sequence[RankedPaper]) -> str:
    for ranked in ranked_papers:
        analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
        labels = [
            *normalize_related_topics(analysis.related_topics or infer_related_topics(ranked.paper)),
            *normalize_tags(analysis.tags or infer_dynamic_tags(ranked.paper), max_tags=6),
        ]
        if label not in {display_trend_label(item) for item in labels}:
            continue
        text = paper_topic_text(ranked.paper, analysis)
        if contains_any(text, ["benchmark", "evaluation", "metric"]):
            return "평가 기준과 벤치마크"
        if contains_any(text, ["dataset", "corpus", "annotation"]):
            return "데이터셋과 태스크 정의"
        if contains_any(text, ["tool", "agent", "planning", "planner"]):
            return "에이전트 설계와 도구 사용"
        if contains_any(text, ["memory", "long context", "retrieval", "rag"]):
            return "지식 검색과 장기 문맥"
        return contribution_summary(analysis)
    return "새 문제 설정"


def render_topic_distribution(ranked_papers: Sequence[RankedPaper]) -> list[str]:
    trend_counts = collect_trend_counts(ranked_papers)
    if not trend_counts:
        return ["Research (0)", ""]
    lines: list[str] = []
    for label, count in trend_counts[:8]:
        lines.extend([f"{label} ({count})", topic_bar(count), ""])
    return lines[:-1]


def topic_bar(count: int) -> str:
    return "#" * max(1, min(12, count))


def render_hidden_gem(
    ranked_papers: Sequence[RankedPaper],
    *,
    saved_ranks: set[int],
) -> list[str]:
    if not ranked_papers:
        return [
            "- Paper: 오늘은 추천할 후보가 없습니다.",
            "- Reason: 수집된 논문이 생기면 낮은 rank 중 신선한 아이디어를 자동으로 고릅니다.",
        ]

    candidates = list(ranked_papers[5:]) or list(ranked_papers)
    gem = max(candidates, key=lambda ranked: (hidden_gem_score(ranked), ranked.score, -ranked.rank))
    analysis = gem.analysis or build_analysis_fallback(gem.paper)
    return [
        f"- Paper: {daily_index_title(gem, saved_ranks=saved_ranks)}",
        f"- Rank: {gem.rank}",
        f"- Why Read: {hidden_gem_why_read(gem, analysis)}",
        f"- Novelty: {novelty_rating(gem)}",
        f"- Future Potential: {hidden_gem_future_potential(gem, analysis)}",
        f"- Reason: {hidden_gem_reason(gem, analysis)}",
    ]


def hidden_gem_score(ranked: RankedPaper) -> float:
    analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
    text = paper_topic_text(ranked.paper, analysis)
    novelty_keywords = [
        "novel",
        "new",
        "first",
        "benchmark",
        "dataset",
        "taxonomy",
        "framework",
        "system",
        "empirical",
        "evaluation",
    ]
    return float(sum(1 for keyword in novelty_keywords if keyword in text))


def novelty_rating(ranked: RankedPaper) -> str:
    score = hidden_gem_score(ranked)
    if score >= 4:
        return "⭐⭐⭐⭐⭐"
    if score >= 3:
        return "⭐⭐⭐⭐☆"
    if score >= 2:
        return "⭐⭐⭐☆☆"
    if score >= 1:
        return "⭐⭐☆☆☆"
    return "⭐☆☆☆☆"


def hidden_gem_reason(ranked: RankedPaper, analysis: PaperAnalysis) -> str:
    text = paper_topic_text(ranked.paper, analysis)
    if contains_any(text, ["benchmark", "evaluation"]):
        return "순위는 낮지만 평가 기준이나 벤치마크 관점이 강해 후속 연구의 기준점이 될 수 있습니다."
    if contains_any(text, ["dataset", "corpus"]):
        return "새 데이터나 태스크 정의가 포함되어 있어 프로젝트 아이디어로 전환하기 쉽습니다."
    if contains_any(text, ["novel", "first", "new"]):
        return "초록에서 새로운 문제 설정이나 접근 방식이 드러나며, 상위권 논문과 다른 방향의 신선함이 있습니다."
    return f"상위권은 아니지만 {newspaper_text(analysis.one_sentence_summary, '관심 주제와 연결되는 독립적인 아이디어')} 때문에 따로 저장해둘 가치가 있습니다."


def hidden_gem_why_read(ranked: RankedPaper, analysis: PaperAnalysis) -> str:
    contribution = contribution_summary(analysis)
    return f"Top5 밖에서도 {contribution}이 뚜렷해 후속 아이디어를 얻기 좋습니다."


def hidden_gem_future_potential(ranked: RankedPaper, analysis: PaperAnalysis) -> str:
    text = paper_topic_text(ranked.paper, analysis)
    if contains_any(text, ["benchmark", "evaluation"]):
        return "새 평가 기준이나 리더보드로 확장될 가능성이 있습니다."
    if contains_any(text, ["dataset", "corpus"]):
        return "데이터셋 기반 후속 연구와 미니 프로젝트로 확장하기 쉽습니다."
    if contains_any(text, ["agent", "tool", "planning"]):
        return "Agent 자동화 기능이나 도구 사용 실험으로 구현해볼 수 있습니다."
    return "관련 주제의 비교 실험이나 Obsidian 연구 아이디어로 재활용할 수 있습니다."


def render_this_week_build(ranked_papers: Sequence[RankedPaper]) -> list[str]:
    if not ranked_papers:
        return [
            "- Project: 이번 주에는 arXiv 수집과 Obsidian 저장 파이프라인을 안정화합니다.",
            "- Difficulty: ⭐⭐☆☆☆",
            "- Time: 2-3 days",
            "- Tech Stack: Python, OpenAI API, Obsidian Markdown",
            "- First Step: 하루치 논문을 다시 실행해 저장 결과를 점검합니다.",
        ]

    base = ranked_papers[0]
    analysis = base.analysis or build_analysis_fallback(base.paper)
    build_plan = analysis.can_i_build_it
    project = practical_project_name(base, analysis)
    return [
        f"- Project: {project}",
        f"- Difficulty: {build_plan.difficulty}",
        f"- Time: {estimate_build_time(base, analysis)}",
        f"- Tech Stack: {framework_recommendation_for_paper(base.paper, analysis)}",
        f"- First Step: {newspaper_text(build_plan.suggested_mini_project, '가장 작은 입력 예제로 핵심 아이디어를 검증합니다.')}",
    ]


def estimate_build_time(ranked: RankedPaper, analysis: PaperAnalysis) -> str:
    text = paper_topic_text(ranked.paper, analysis)
    difficulty = analysis.can_i_build_it.difficulty.count("⭐")
    if contains_any(text, ["fine-tuning", "training", "robot", "robotics"]) or difficulty >= 5:
        return "2 weeks"
    if contains_any(text, ["benchmark", "dataset", "evaluation", "rag", "memory"]) or difficulty >= 4:
        return "5 days"
    return "2 days"


def practical_project_name(ranked: RankedPaper, analysis: PaperAnalysis) -> str:
    text = paper_topic_text(ranked.paper, analysis)
    if contains_any(text, ["benchmark", "evaluation"]):
        return "Mini Agent Evaluation Dashboard"
    if contains_any(text, ["coding", "software", "code"]):
        return "Coding Agent Task Runner"
    if contains_any(text, ["rag", "retrieval"]):
        return "Paper RAG Quality Checker"
    if contains_any(text, ["memory", "long context"]):
        return "LLM Memory Experiment Tracker"
    if contains_any(text, ["dataset", "corpus"]):
        return "Dataset Inspection and Scoring Tool"
    if contains_any(text, ["robot", "robotics"]):
        return "LLM Robotics Planning Simulator"
    return "AI Paper Insight Prototype"


def render_todays_one_sentence(ranked_papers: Sequence[RankedPaper]) -> str:
    if not ranked_papers:
        return "오늘은 수집된 후보가 없어 AI 연구 흐름을 판단하기 어렵습니다."
    trend_counts = collect_trend_counts(ranked_papers)
    labels = [label for label, _ in trend_counts[:3]]
    if not labels:
        return "오늘 AI 연구는 다양한 주제가 분산되어 있어 Top20 후보를 직접 훑어볼 가치가 있습니다."
    contribution = trend_contribution_phrase(labels[0], ranked_papers)
    if len(labels) == 1:
        return f"오늘 AI 연구는 {labels[0]}를 중심으로 {contribution}을 강화하는 흐름입니다."
    return f"오늘 AI 연구는 {', '.join(labels[:-1])}, {labels[-1]}를 중심으로 {contribution}을 강화하는 흐름입니다."


def contribution_summary(analysis: PaperAnalysis) -> str:
    for value in [analysis.key_contributions, analysis.difference_from_previous_work, analysis.one_sentence_summary]:
        cleaned = newspaper_text(value, "")
        if cleaned:
            return truncate_sentence(cleaned)
    return "핵심 기여"


def truncate_sentence(value: str, limit: int = 90) -> str:
    cleaned = " ".join(str(value).split())
    return cleaned if len(cleaned) <= limit else f"{cleaned[:limit].rstrip()}..."


def render_research_timeline(
    ranked_papers: Sequence[RankedPaper],
    *,
    saved_ranks: set[int],
) -> list[str]:
    if not ranked_papers:
        return ["오늘은 연결할 Top5 논문이 없습니다."]

    lines: list[str] = []
    for index, ranked in enumerate(ranked_papers, start=1):
        lines.append(daily_index_title(ranked, saved_ranks=saved_ranks))
        lines.append("")
        lines.append(f"({timeline_role(ranked)})")
        if index < len(ranked_papers):
            next_ranked = ranked_papers[index]
            lines.extend(["", "↓", "", f"연결 이유: {timeline_connection_reason(ranked, next_ranked)}", ""])
    return lines


def timeline_role(ranked: RankedPaper) -> str:
    category = reading_stage(ranked)
    labels = {
        "Foundation": "Foundation",
        "Benchmark": "Evaluation",
        "Method": "Method",
        "Application": "Application",
        "Survey": "Survey",
    }
    return labels.get(category, category)


def timeline_connection_reason(current: RankedPaper, next_ranked: RankedPaper) -> str:
    current_stage = reading_stage(current)
    next_stage = reading_stage(next_ranked)
    if current_stage != next_stage:
        return f"{current_stage} 관점에서 {next_stage} 관점으로 넘어가며 연구 맥락을 넓힙니다."
    current_analysis = current.analysis or build_analysis_fallback(current.paper)
    next_analysis = next_ranked.analysis or build_analysis_fallback(next_ranked.paper)
    current_labels = set(normalize_tags(current_analysis.tags or infer_dynamic_tags(current.paper), max_tags=6))
    next_labels = set(normalize_tags(next_analysis.tags or infer_dynamic_tags(next_ranked.paper), max_tags=6))
    shared = sorted(current_labels & next_labels)
    if shared:
        return f"둘 다 {', '.join(shared[:2])} 주제를 다루므로 비교해서 읽기 좋습니다."
    return "앞 논문의 문제 설정을 다음 논문의 방법이나 응용 관점과 연결해 볼 수 있습니다."


def newspaper_text(value: str, fallback: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return fallback
    failure_markers = ["자동 생성 실패", "요약 생성에 실패", "판단할 수 없습니다"]
    if any(marker in cleaned for marker in failure_markers):
        return fallback
    return cleaned


def render_daily_briefing_sections(
    ranked_papers: Sequence[RankedPaper],
    *,
    saved_ranks: set[int],
    deep_ranks: set[int],
) -> list[str]:
    return [
        "## 🔥 Must Read Today",
        "",
        *render_must_read_today(ranked_papers[:2], saved_ranks=saved_ranks),
        "",
        "## 📈 Today\u0027s Research Trends",
        "",
        *render_research_trends(ranked_papers),
        "",
        "## 🏆 Editor\u0027s Pick",
        "",
        *render_editors_pick(ranked_papers, saved_ranks=saved_ranks),
        "",
        "## 📚 Recommended Reading Order",
        "",
        *render_recommended_reading_order(ranked_papers[:5], saved_ranks=saved_ranks),
        "",
        "## 💡 Today\u0027s Project",
        "",
        *render_todays_project(ranked_papers[:5]),
        "",
        "## 💡 Today\u0027s Research Ideas",
        "",
        *render_todays_research_ideas(ranked_papers[:5], saved_ranks=saved_ranks, deep_ranks=deep_ranks),
    ]


def render_must_read_today(
    ranked_papers: Sequence[RankedPaper],
    *,
    saved_ranks: set[int],
) -> list[str]:
    if not ranked_papers:
        return ["_No candidate papers available._"]

    lines: list[str] = []
    for index, ranked in enumerate(ranked_papers, start=1):
        analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
        lines.extend(
            [
                f"{index}. {daily_index_title(ranked, saved_ranks=saved_ranks)}",
                f"   - Reason: {must_read_reason(ranked, analysis)}",
                f"   - Why it matters: {newspaper_text(analysis.why_important, 'Useful reference for Agent and Coding Agent research.')}",
                "",
            ]
        )
    return lines[:-1] if lines else lines


def must_read_reason(ranked: RankedPaper, analysis: PaperAnalysis) -> str:
    contribution = contribution_summary(analysis)
    text = paper_topic_text(ranked.paper, analysis)
    if contains_any(text, ["benchmark", "evaluation"]):
        return f"{contribution} 특히 평가 기준과 비교 실험 관점이 강해 오늘 먼저 읽을 가치가 있습니다."
    if contains_any(text, ["agent", "tool", "planning"]):
        return f"{contribution} Agent 설계나 도구 사용 흐름에 바로 연결됩니다."
    if contains_any(text, ["rag", "retrieval", "memory", "long context"]):
        return f"{contribution} 지식 검색, 메모리, 긴 문맥 처리 개선에 연결됩니다."
    return f"{contribution} 유사도 {ranked.score:.3f}로 오늘 관심사와 강하게 맞닿아 있습니다."


def render_research_trends(ranked_papers: Sequence[RankedPaper]) -> list[str]:
    trend_counts = collect_trend_counts(ranked_papers)
    if not trend_counts:
        return ["_No clear trends detected._"]
    return [
        f"- {label}: {count} papers — {describe_trend(label)}"
        for label, count in trend_counts[:5]
    ]


def collect_trend_counts(ranked_papers: Sequence[RankedPaper]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    display_labels: dict[str, str] = {}
    for ranked in ranked_papers:
        analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
        labels = [
            *normalize_related_topics(analysis.related_topics or infer_related_topics(ranked.paper)),
            *normalize_tags(analysis.tags or infer_dynamic_tags(ranked.paper), max_tags=6),
        ]
        displays = unique_preserving_order(display_trend_label(label) for label in labels)
        for display in displays:
            key = display.lower()
            counts[key] = counts.get(key, 0) + 1
            display_labels.setdefault(key, display)

    sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [(display_labels[key], count) for key, count in sorted_counts]


def display_trend_label(value: str) -> str:
    cleaned = str(value).strip().replace("_", "-").replace(".", "-")
    canonical = {
        "cs-ai": "AI",
        "ai": "AI",
        "cs-cl": "NLP",
        "nlp": "NLP",
        "cs-lg": "Machine Learning",
        "machine-learning": "Machine Learning",
        "cs-se": "Software Engineering",
        "software-engineering": "Software Engineering",
        "cs-ir": "Information Retrieval",
        "information-retrieval": "Information Retrieval",
        "agent": "Agent",
        "agents": "Agent",
        "rag": "RAG",
        "retrieval": "RAG",
        "reasoning": "Reasoning",
        "coding-agent": "Coding Agent",
        "code": "Coding Agent",
        "programming": "Coding Agent",
        "multi-agent": "Multi Agent",
        "long-context": "Long Context",
        "memory": "Memory",
        "llm-memory": "Memory",
        "robotics": "Robotics",
        "benchmark": "Benchmark",
        "evaluation": "Evaluation",
        "dataset": "Dataset",
    }
    key = cleaned.lower()
    if key in canonical:
        return canonical[key]
    return " ".join(part.capitalize() for part in cleaned.replace("-", " ").split())


def describe_trend(label: str) -> str:
    descriptions = {
        "AI": "cs.AI 계열의 일반 AI 방법론과 응용 연구가 반복적으로 등장합니다.",
        "NLP": "언어 이해, 생성, 평가와 관련된 연구 흐름이 이어집니다.",
        "Machine Learning": "학습 방법, 모델 개선, 일반화 성능 문제가 함께 다뤄집니다.",
        "Software Engineering": "코딩 에이전트와 AI 개발 자동화로 확장될 가능성이 큽니다.",
        "Information Retrieval": "검색, 랭킹, RAG 기반 지식 활용과 맞닿아 있습니다.",
        "Agent": "도구 사용, 계획, 자동화 흐름과 연결된 연구가 강하게 보입니다.",
        "RAG": "검색 기반 문맥 확장과 지식 활용 문제가 계속 중요한 축입니다.",
        "Reasoning": "복잡한 추론, 계획, 평가 방법 개선 흐름을 보여줍니다.",
        "Coding Agent": "소프트웨어 개발 자동화와 코드 이해/생성 응용으로 이어질 수 있습니다.",
        "Multi Agent": "여러 모델 또는 에이전트 간 협업과 조정 문제가 부각됩니다.",
        "Long Context": "긴 입력 처리와 메모리 설계가 핵심 병목으로 다뤄집니다.",
        "Memory": "지속 기억, 개인화, 장기 작업 맥락 관리와 관련됩니다.",
        "Robotics": "언어 모델을 실제 행동 계획과 물리 환경으로 확장하는 흐름입니다.",
        "Benchmark": "모델 능력을 더 정확히 재기 위한 평가 기준이 늘고 있습니다.",
        "Evaluation": "성능 비교와 신뢰 가능한 측정 방법이 주요 관심사입니다.",
        "Dataset": "새 데이터 구축이 태스크 정의와 평가 품질을 좌우합니다.",
    }
    return descriptions.get(label, "오늘 Top20에서 반복적으로 등장한 주제입니다.")


def render_editors_pick(
    ranked_papers: Sequence[RankedPaper],
    *,
    saved_ranks: set[int],
) -> list[str]:
    if not ranked_papers:
        return ["- Paper: _No candidate papers available._"]
    pick = max(
        ranked_papers,
        key=lambda ranked: (
            rating_value(novelty_rating(ranked)),
            ranked.score,
            -ranked.rank,
        ),
    )
    analysis = pick.analysis or build_analysis_fallback(pick.paper)
    return [
        f"- Paper: {daily_index_title(pick, saved_ranks=saved_ranks)}",
        f"- Why selected: rank {pick.rank}, similarity {pick.score:.6f}이며 novelty와 프로젝트 전환 가능성이 높습니다. {newspaper_text(analysis.why_important, 'Useful reference for Agent and Coding Agent research.')}",
        f"- Novelty: {novelty_rating(pick)} — {editor_evaluation_reason(pick, analysis, 'novelty')}",
        f"- Impact: {interest_level(pick.score)} — {editor_evaluation_reason(pick, analysis, 'impact')}",
        f"- Research Value: {research_value_rating(pick)} — {editor_evaluation_reason(pick, analysis, 'research')}",
        f"- Project Potential: {project_potential_rating(pick)} — {editor_evaluation_reason(pick, analysis, 'project')}",
    ]


def rating_value(stars: str) -> int:
    return stars.count("⭐")


def research_value_rating(ranked: RankedPaper) -> str:
    analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
    text = paper_topic_text(ranked.paper, analysis)
    score = ranked.score
    if contains_any(text, ["benchmark", "dataset", "evaluation", "empirical", "taxonomy", "method", "ablation"]):
        score = max(score, 0.60)
    return interest_level(score)


def project_potential_rating(ranked: RankedPaper) -> str:
    analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
    text = paper_topic_text(ranked.paper, analysis)
    score = ranked.score
    if contains_any(text, ["agent", "coding", "rag", "tool", "system", "dataset", "benchmark", "github"]):
        score = max(score, 0.60)
    return interest_level(score)


def editor_evaluation_reason(ranked: RankedPaper, analysis: PaperAnalysis, kind: str) -> str:
    text = paper_topic_text(ranked.paper, analysis)
    if kind == "novelty":
        if contains_any(text, ["novel", "first", "new", "taxonomy"]):
            return "새 문제 설정이나 정리 체계가 보여 오늘 후보군에서 신선도가 높습니다."
        if contains_any(text, ["benchmark", "dataset"]):
            return "새 평가 기준이나 데이터 구성이 후속 연구의 기준점이 될 수 있습니다."
        return "초록 기준으로 기존 흐름과 연결되지만 원문에서 차별점을 확인할 가치가 있습니다."
    if kind == "impact":
        return "관심사 embedding과의 유사도가 높아 Agent/Coding Agent 연구 흐름에 직접 연결됩니다."
    if kind == "research":
        if contains_any(text, ["evaluation", "benchmark", "empirical", "ablation"]):
            return "평가 설계와 실험 비교를 통해 후속 연구의 근거 자료로 쓰기 좋습니다."
        return "핵심 아이디어를 관련 연구 맥락 안에서 비교해볼 가치가 있습니다."
    if contains_any(text, ["agent", "coding", "rag", "tool", "system"]):
        return "작은 프로토타입이나 Obsidian 연구 자동화 기능으로 전환하기 쉽습니다."
    return "아이디어를 미니 프로젝트로 축소해 검증할 여지가 있습니다."


def render_recommended_reading_order(
    ranked_papers: Sequence[RankedPaper],
    *,
    saved_ranks: set[int],
) -> list[str]:
    if not ranked_papers:
        return ["_No saved papers available._"]

    lines: list[str] = []
    ordered = sorted(ranked_papers, key=lambda ranked: (reading_stage_order(reading_stage(ranked)), ranked.rank))
    for index, ranked in enumerate(ordered, start=1):
        analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
        stage = reading_stage(ranked)
        lines.append(
            f"{index}. {daily_index_title(ranked, saved_ranks=saved_ranks)} — "
            f"{stage}: {reading_stage_reason(stage, analysis)} {analysis.one_sentence_summary}"
        )
    return lines


def reading_stage(ranked: RankedPaper) -> str:
    analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
    text = paper_topic_text(ranked.paper, analysis)
    if contains_any(text, ["survey", "review", "taxonomy", "overview"]):
        return "Survey"
    if contains_any(text, ["benchmark", "evaluation", "leaderboard", "metric", "dataset"]):
        return "Benchmark"
    if contains_any(text, ["method", "framework", "algorithm", "architecture", "training", "optimization", "reasoning"]):
        return "Method"
    if contains_any(text, ["application", "system", "tool", "coding", "software", "robot", "automation"]):
        return "Application"
    return "Foundation"


def reading_stage_order(stage: str) -> int:
    order = {
        "Foundation": 0,
        "Benchmark": 1,
        "Method": 2,
        "Application": 3,
        "Survey": 4,
    }
    return order.get(stage, 99)


def reading_stage_reason(stage: str, analysis: PaperAnalysis) -> str:
    reasons = {
        "Foundation": "먼저 문제 설정과 기본 개념을 잡기 좋습니다.",
        "Benchmark": "그다음 평가 기준과 비교 지표를 확인하기 좋습니다.",
        "Method": "평가 기준을 이해한 뒤 방법론을 읽기 좋습니다.",
        "Application": "방법론을 실제 시스템이나 프로젝트로 연결하기 좋습니다.",
        "Survey": "마지막에 전체 지형을 정리하며 빠진 연결을 확인하기 좋습니다.",
    }
    return reasons.get(stage, contribution_summary(analysis))


def short_title(ranked: RankedPaper) -> str:
    title = str(ranked.paper.get("title", "Untitled Paper")).strip()
    words = title.split()
    return " ".join(words[:5]) if len(words) > 5 else title


def render_todays_project(ranked_papers: Sequence[RankedPaper]) -> list[str]:
    if not ranked_papers:
        return ["- Project: _No project generated._"]

    base = ranked_papers[0]
    analysis = base.analysis or build_analysis_fallback(base.paper)
    build_plan = analysis.can_i_build_it
    project = analysis.project_idea or build_plan.suggested_mini_project
    based_on = ", ".join(
        daily_index_title(ranked, saved_ranks={ranked.rank})
        for ranked in ranked_papers[:3]
    )
    return [
        f"- Project: {project}",
        f"- Based on: {based_on}",
        f"- Difficulty: {build_plan.difficulty}",
        f"- Time Estimate: {build_plan.time_estimate}",
        "- Tech Stack: Python, OpenAI API, arXiv metadata, Obsidian Markdown",
        f"- First Step: {build_plan.suggested_mini_project}",
    ]


def render_todays_research_ideas(
    ranked_papers: Sequence[RankedPaper],
    *,
    saved_ranks: set[int],
    deep_ranks: set[int],
) -> list[str]:
    sources = list(ranked_papers[:5])
    if not sources:
        return [
            "1. Idea: 오늘은 연구 아이디어를 만들 Top5 논문이 부족합니다.",
            "   Based on: _No saved papers available._",
            "   Why promising: 후보 논문이 쌓이면 반복 주제와 Deep Read 근거를 연결해 연구 질문을 생성합니다.",
            "   Difficulty: ⭐⭐☆☆☆",
            "   First experiment: 내일 Daily Note에서 Top5 후보와 Deep Read 노트가 생성되는지 확인합니다.",
        ]

    deep_sources = [ranked for ranked in sources if ranked.rank in deep_ranks or ranked.deep_analysis.strip()]
    lines: list[str] = []
    for index in range(3):
        primary = sources[index % len(sources)]
        secondary_pool = deep_sources or sources
        secondary = secondary_pool[index % len(secondary_pool)]
        if len(sources) > 1 and secondary.rank == primary.rank:
            secondary = sources[(index + 1) % len(sources)]
        analysis = primary.analysis or build_analysis_fallback(primary.paper)
        question = research_question_for_sources(primary, secondary, index)
        links = research_idea_links(primary, secondary, saved_ranks=saved_ranks, deep_ranks=deep_ranks)
        lines.extend(
            [
                f"{index + 1}. Idea: {question}",
                f"   Based on: {links}",
                f"   Why promising: {research_idea_promise(primary, secondary)}",
                f"   Difficulty: {research_idea_difficulty(primary, secondary)}",
                f"   First experiment: {research_idea_first_experiment(primary, analysis)}",
                "",
            ]
        )
    return lines[:-1]


def research_question_for_sources(primary: RankedPaper, secondary: RankedPaper, index: int) -> str:
    analysis = primary.analysis or build_analysis_fallback(primary.paper)
    text = paper_topic_text(primary.paper, analysis)
    topic = compact_topic_label(primary)
    if contains_any(text, ["benchmark", "evaluation", "metric"]):
        return f"Can a small failure-focused benchmark for {topic} predict real task success better than aggregate scores?"
    if contains_any(text, ["rag", "retrieval", "memory", "long context"]):
        return f"Can adaptive retrieval and memory selection reduce context noise in {topic} tasks?"
    if contains_any(text, ["agent", "tool", "planning", "coding", "software"]):
        return f"Can separating planning traces from tool-use logs improve {topic} agent reliability?"
    if contains_any(text, ["dataset", "corpus", "annotation"]):
        return f"Can coverage-guided filtering create a smaller but more useful {topic} evaluation dataset?"
    if index == 1:
        return f"Which minimal ablation best explains performance differences between {compact_topic_label(primary)} and {compact_topic_label(secondary)}?"
    return f"Can a one-week reproducible study turn {topic} into a measurable research benchmark?"


def research_idea_links(
    primary: RankedPaper,
    secondary: RankedPaper,
    *,
    saved_ranks: set[int],
    deep_ranks: set[int],
) -> str:
    links = [daily_index_title(primary, saved_ranks=saved_ranks)]
    if primary.rank in deep_ranks or primary.deep_analysis.strip():
        links.append(deep_index_title(primary))
    if secondary.rank != primary.rank:
        links.append(daily_index_title(secondary, saved_ranks=saved_ranks))
        if secondary.rank in deep_ranks or secondary.deep_analysis.strip():
            links.append(deep_index_title(secondary))
    return ", ".join(unique_preserving_order(links))


def deep_index_title(ranked: RankedPaper) -> str:
    title = str(ranked.paper.get("title", "Untitled Paper"))
    return f"[[Deep/{paper_note_filename(ranked.paper)}|{escape_wikilink_label(title)} Deep]]"


def research_idea_promise(primary: RankedPaper, secondary: RankedPaper) -> str:
    primary_analysis = primary.analysis or build_analysis_fallback(primary.paper)
    secondary_analysis = secondary.analysis or build_analysis_fallback(secondary.paper)
    shared = sorted(
        set(normalize_tags(primary_analysis.tags or infer_dynamic_tags(primary.paper), max_tags=6))
        & set(normalize_tags(secondary_analysis.tags or infer_dynamic_tags(secondary.paper), max_tags=6))
    )
    if shared:
        shared_text = ", ".join(shared[:2])
        return f"Top5 안에서 {shared_text} 축이 반복되어 비교 실험으로 논문화하기 좋습니다."
    if primary.deep_analysis.strip() or secondary.deep_analysis.strip():
        return "Deep Read 근거가 있어 초록 수준 아이디어를 방법론과 한계 분석까지 연결할 수 있습니다."
    return f"{contribution_summary(primary_analysis)} 흐름을 다른 Top5 논문과 비교해 작은 실험 질문으로 좁힐 수 있습니다."


def research_idea_difficulty(primary: RankedPaper, secondary: RankedPaper) -> str:
    combined = f"{paper_topic_text(primary.paper, primary.analysis or build_analysis_fallback(primary.paper))} {paper_topic_text(secondary.paper, secondary.analysis or build_analysis_fallback(secondary.paper))}"
    if contains_any(combined, ["fine-tuning", "training", "robot", "robotics"]):
        return "⭐⭐⭐⭐☆"
    if contains_any(combined, ["benchmark", "dataset", "evaluation", "rag", "memory"]):
        return "⭐⭐⭐☆☆"
    return "⭐⭐☆☆☆"


def research_idea_first_experiment(primary: RankedPaper, analysis: PaperAnalysis) -> str:
    text = paper_topic_text(primary.paper, analysis)
    if contains_any(text, ["benchmark", "evaluation", "metric"]):
        return "1주일 안에 20개 미니 태스크를 만들고 기존 점수와 실패 유형 점수의 상관을 비교합니다."
    if contains_any(text, ["rag", "retrieval", "memory", "long context"]):
        return "논문 10개 QA 세트를 만들고 고정 retrieval과 adaptive retrieval의 정답률과 토큰 비용을 비교합니다."
    if contains_any(text, ["agent", "tool", "planning", "coding", "software"]):
        return "간단한 코딩 태스크 10개에서 planner 로그 분리 전후의 성공률과 재시도 횟수를 측정합니다."
    if contains_any(text, ["dataset", "corpus", "annotation"]):
        return "샘플 100개를 난이도와 coverage로 태깅한 뒤 작은 평가셋이 전체셋 순위를 보존하는지 확인합니다."
    return "Top5 논문 중 2편을 골라 같은 입력, 같은 모델, 같은 지표로 최소 재현 실험을 설계합니다."


def compact_topic_label(ranked: RankedPaper) -> str:
    analysis = ranked.analysis or build_analysis_fallback(ranked.paper)
    labels = normalize_related_topics(analysis.related_topics or infer_related_topics(ranked.paper))
    if labels:
        return labels[0]
    tags = normalize_tags(analysis.tags or infer_dynamic_tags(ranked.paper), max_tags=1)
    if tags:
        return display_trend_label(tags[0])
    title = str(ranked.paper.get("title", "this paper")).split()
    return " ".join(title[:4]) if title else "this paper"


def daily_index_title(ranked: RankedPaper, *, saved_ranks: set[int]) -> str:
    title = str(ranked.paper.get("title", "Untitled Paper"))
    escaped_title = escape_wikilink_label(title)
    if ranked.rank in saved_ranks:
        return f"[[Papers/{paper_note_filename(ranked.paper)}|{escaped_title}]]"
    return escaped_title


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def get_interest_sentence(config: dict[str, Any]) -> str:
    cleaned = get_interest_topics(config)
    if cleaned:
        return build_interest_sentence(cleaned)

    interests = config.get("interests", [])
    if isinstance(interests, dict):
        sentence = str(interests.get("sentence", "")).strip()
        if sentence:
            return sentence
    raise ValueError("config.yaml must define interests as a non-empty list.")


def build_interest_sentence(interests: Sequence[str]) -> str:
    interest_text = ", ".join(str(interest).strip() for interest in interests if str(interest).strip())
    return (
        "I am interested in research papers about "
        f"{interest_text}. "
        "Prioritize papers that are practical for building AI systems, developer tools, "
        "research automation workflows, and projects that can become useful software products."
    )


def get_interest_topics(config: dict[str, Any]) -> list[str]:
    interests = config.get("interests", [])
    if isinstance(interests, list):
        return [str(interest).strip() for interest in interests if str(interest).strip()]
    return []


def get_obsidian_vault_path(obsidian_config: dict[str, Any]) -> Path:
    env_path = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
    configured_path = obsidian_config.get("vault_path", DEFAULT_VAULT_DIR)
    return Path(env_path or configured_path).expanduser()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full daily paper-agent pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--date", type=date.fromisoformat, help="UTC arXiv submission date.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug logs and saved file paths.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "DEBUG" if args.verbose else args.log_level.upper()
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    stats = PipelineStats()
    try:
        saved_paths = run_daily(args.config, today=args.date, stats=stats)
    except Exception as exc:
        logging.error("Daily pipeline failed: %s", exc)
        return 1

    if args.verbose:
        for path in saved_paths:
            logging.debug("Saved file: %s", path)
    logging.info(format_pipeline_summary(stats))
    return 0


def format_pipeline_summary(stats: PipelineStats) -> str:
    return "\n".join(
        [
            "✅ Daily pipeline completed",
            f"- Fetched papers: {stats.fetched_papers}",
            f"- Candidate papers: {stats.candidate_papers}",
            f"- Saved daily notes: {stats.saved_daily_notes}",
            f"- Saved paper notes: {stats.saved_paper_notes}",
            f"- Deep read papers: {stats.saved_deep_notes}",
            f"- Embedding calls: {stats.embedding_calls}",
            f"- Response calls: {stats.response_calls}",
            f"- Cache hits: {stats.cache_hits}",
            f"- Cache misses: {stats.cache_misses}",
            f"- Runtime: {stats.runtime_seconds:.1f}s",
            f"- Output: {stats.output_dir or ''}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
