from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from deep_read import deep_read_pdf
from embedding import create_embeddings
from fetch import build_query_url, fetch_feed, parse_feed
from rank import cosine_similarity
from save import render_front_matter, safe_filename
from summarize import summarize_abstract


DEFAULT_CATEGORY = "cs.AI"
DEFAULT_MAX_RESULTS = 100
DEFAULT_TOP_K = 20
DEFAULT_DEEP_READ_COUNT = 3
DEFAULT_VAULT_DIR = Path("obsidian")
DEFAULT_FOLDER = "papers"


@dataclass(frozen=True)
class RankedPaper:
    rank: int
    score: float
    paper: dict[str, Any]
    short_summary: str = ""
    deep_analysis: str = ""


def run_daily(config_path: Path, *, today: date | None = None) -> list[Path]:
    config = load_config(config_path)
    fetch_config = config.get("fetch", {})
    ranking_config = config.get("ranking", {})
    obsidian_config = config.get("obsidian", {})
    models_config = config.get("models", {})

    target_date = today or datetime.now(timezone.utc).date()
    category = str(fetch_config.get("category", DEFAULT_CATEGORY))
    max_results = int(fetch_config.get("max_results", DEFAULT_MAX_RESULTS))
    top_k = int(ranking_config.get("top_k", DEFAULT_TOP_K))
    deep_read_count = int(ranking_config.get("deep_read_k", DEFAULT_DEEP_READ_COUNT))
    interest = get_interest_sentence(config)
    vault_dir = get_obsidian_vault_path(obsidian_config)
    folder = str(obsidian_config.get("folder", DEFAULT_FOLDER))
    work_dir = Path(".paper_agent_cache") / target_date.isoformat()
    embedding_model = str(models_config.get("embedding", "text-embedding-3-small"))
    summary_model = str(models_config.get("summary", "gpt-5.5"))
    deep_read_model = str(models_config.get("deep_read", summary_model))

    papers = fetch_papers(category, target_date, max_results)
    logging.info("Fetched papers: %s", len(papers))
    logging.info("Embedding candidate papers: %s", count_embedding_candidates(papers))

    top_papers = rank_papers_by_interest(papers, interest, embedding_model=embedding_model, top_k=top_k)
    logging.info("Ranking result top_papers: %s", len(top_papers))
    if not top_papers:
        raise ValueError("Ranking produced no top papers. Check fetch results, embeddings, and config.")

    enriched = enrich_ranked_papers(
        top_papers,
        work_dir,
        summary_model=summary_model,
        deep_read_model=deep_read_model,
        deep_read_count=deep_read_count,
    )
    return save_daily_markdown(
        enriched,
        vault_dir,
        folder=folder,
        target_date=target_date,
        interests=get_interest_topics(config),
    )


def fetch_papers(category: str, target_date: date, max_results: int) -> list[dict[str, Any]]:
    url = build_query_url(category, target_date, max_results)
    papers = parse_feed(fetch_feed(url, timeout=30))
    return [normalize_paper(paper) for paper in papers]


def rank_papers_by_interest(
    papers: Sequence[dict[str, Any]],
    interest: str,
    *,
    embedding_model: str,
    top_k: int,
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

    logging.info("Embedding batch size: %s", len(candidates) + 1)
    try:
        embedding_results = create_embeddings(
            [interest, *[embedding_input for _, embedding_input in candidates]],
            model=embedding_model,
        )
    except Exception as exc:
        logging.warning("Embedding batch failed: %s", exc)
        return []

    interest_embedding = embedding_results[0].embedding
    paper_embeddings = embedding_results[1:]
    scored: list[tuple[float, dict[str, Any]]] = []

    for (paper, _), embedding_result in zip(candidates, paper_embeddings):
        title = str(paper.get("title", "")).strip()
        try:
            score = cosine_similarity(interest_embedding, embedding_result.embedding)
        except Exception as exc:
            logging.warning("Skipping paper after ranking failure: %s (%s)", title, exc)
            continue

        scored.append((score, paper))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        RankedPaper(rank=rank, score=score, paper=paper)
        for rank, (score, paper) in enumerate(scored[:top_k], start=1)
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
) -> list[RankedPaper]:
    enriched = []
    for ranked in ranked_papers:
        paper = ranked.paper
        title = str(paper.get("title", "Untitled Paper"))
        abstract = str(paper.get("summary", ""))
        short_summary = ""
        deep_analysis = ""

        try:
            short_summary = summarize_abstract(title, abstract, model=summary_model)
        except Exception as exc:
            logging.warning("Using fallback summary after abstract summary failure: %s (%s)", title, exc)
            short_summary = build_summary_fallback(paper)

        if ranked.rank <= deep_read_count:
            pdf_url = paper.get("pdf_url")
            if pdf_url:
                try:
                    deep_analysis = deep_read_pdf(
                        title,
                        str(pdf_url),
                        work_dir / safe_filename(str(paper.get("arxiv_id", title))),
                        model=deep_read_model,
                    )
                except Exception as exc:
                    logging.warning("Using fallback deep analysis after deep read failure: %s (%s)", title, exc)
                    deep_analysis = build_deep_read_fallback(paper, exc)
            else:
                logging.warning("Using fallback deep analysis for %s: no pdf_url", title)
                deep_analysis = build_deep_read_fallback(paper, "no pdf_url")

        enriched.append(
            RankedPaper(
                rank=ranked.rank,
                score=ranked.score,
                paper=paper,
                short_summary=short_summary,
                deep_analysis=deep_analysis,
            )
        )
    return enriched


def build_summary_fallback(paper: dict[str, Any]) -> str:
    title = str(paper.get("title", "Untitled Paper")).strip()
    abstract = str(paper.get("summary", "")).strip()
    entry_url = str(paper.get("entry_url", "")).strip()
    pdf_url = str(paper.get("pdf_url", "")).strip()
    return "\n".join(
        [
            "## 한 줄 요약",
            "",
            "_OpenAI 요약 생성에 실패했습니다. 아래 원문 초록을 확인하세요._",
            "",
            "## 3줄 요약",
            "",
            abstract or "_초록이 없습니다._",
            "",
            "## 왜 중요한가",
            "",
            "_자동 생성 실패로 판단할 수 없습니다._",
            "",
            "## 기존 연구와 차이",
            "",
            "_자동 생성 실패로 판단할 수 없습니다._",
            "",
            "## 내 프로젝트 적용 아이디어",
            "",
            "_자동 생성 실패로 판단할 수 없습니다._",
            "",
            "## Links",
            "",
            f"- Title: {title}",
            f"- arXiv: {entry_url}",
            f"- PDF: {pdf_url}",
        ]
    )


def build_deep_read_fallback(paper: dict[str, Any], reason: object) -> str:
    entry_url = str(paper.get("entry_url", "")).strip()
    pdf_url = str(paper.get("pdf_url", "")).strip()
    return "\n".join(
        [
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
            "## Links",
            "",
            f"- arXiv: {entry_url}",
            f"- PDF: {pdf_url}",
        ]
    )


def save_daily_markdown(
    ranked_papers: Sequence[RankedPaper],
    vault_dir: Path,
    *,
    folder: str,
    target_date: date,
    interests: Sequence[str] | None = None,
) -> list[Path]:
    if not ranked_papers:
        raise ValueError("No top papers to save. Refusing to write an empty Top20.md.")

    output_dir = vault_dir / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for ranked in ranked_papers:
        title = str(ranked.paper.get("title", "Untitled Paper"))
        output_path = output_dir / f"{target_date.isoformat()}-{safe_filename(title)}.md"
        output_path.write_text(
            render_daily_paper_markdown(ranked, target_date, interests=interests or []),
            encoding="utf-8",
        )
        saved_paths.append(output_path)

    index_path = output_dir / f"{target_date.isoformat()}-Top20.md"
    index_path.write_text(render_daily_index(ranked_papers, target_date), encoding="utf-8")
    saved_paths.append(index_path)
    return saved_paths


def render_daily_paper_markdown(
    ranked: RankedPaper,
    target_date: date,
    *,
    interests: Sequence[str] | None = None,
) -> str:
    paper = ranked.paper
    title = str(paper.get("title", "Untitled Paper"))
    authors = paper.get("authors") if isinstance(paper.get("authors"), list) else []
    categories = paper.get("categories") if isinstance(paper.get("categories"), list) else []
    front_matter = {
        "title": title,
        "date": target_date.isoformat(),
        "rank": ranked.rank,
        "score": f"{ranked.score:.6f}",
        "arxiv_id": paper.get("arxiv_id", ""),
        "arxiv_url": paper.get("entry_url", ""),
        "pdf_url": paper.get("pdf_url", ""),
        "tags": ["paper-agent", "arxiv", "daily"],
    }
    lines = [
        render_front_matter(front_matter),
        f"# {title}",
        "",
        "## Ranking",
        "",
        f"- Rank: {ranked.rank}",
        f"- Similarity: {ranked.score:.6f}",
        "",
        "## Metadata",
        "",
        f"- Authors: {', '.join(str(author) for author in authors) if authors else 'Unknown'}",
        f"- arXiv ID: {paper.get('arxiv_id', '')}",
        f"- Published: {paper.get('published', '')}",
        f"- Categories: {', '.join(str(category) for category in categories)}",
        f"- arXiv: {paper.get('entry_url', '')}",
        f"- PDF: {paper.get('pdf_url', '')}",
        f"- Similarity Score: {ranked.score:.6f}",
        "",
        "## Short Summary",
        "",
        ranked.short_summary or "_Summary failed or unavailable._",
        "",
        "## Abstract",
        "",
        str(paper.get("summary", "")).strip(),
        "",
        render_insight_sections(ranked, interests=interests or []),
        "",
    ]
    if ranked.deep_analysis:
        lines.extend(["## Deep Analysis", "", ranked.deep_analysis, ""])
    return "\n".join(lines)


def render_insight_sections(ranked: RankedPaper, *, interests: Sequence[str]) -> str:
    topics = infer_related_topics(ranked.paper, interests)
    return "\n".join(
        [
            "## My Insight",
            "",
            "- 왜 중요한가? 이 논문은 관심 주제와 유사도가 높아 후속 리서치 후보로 볼 만합니다.",
            f"- Similarity score: {ranked.score:.6f}",
            "",
            "## Startup Idea",
            "",
            "- 논문의 핵심 방법을 특정 산업 문서, 사내 지식, 연구 자동화 워크플로우에 맞춘 SaaS 기능으로 바꿀 수 있는지 검토합니다.",
            "",
            "## Project Idea",
            "",
            "- 초록의 방법을 작은 데이터셋이나 공개 벤치마크로 재현하고, 기존 RAG/Agent 파이프라인에 붙이는 미니 프로젝트를 설계합니다.",
            "",
            "## Related Topics",
            "",
            *[f"- [[{topic}]]" for topic in topics],
        ]
    )


def infer_related_topics(paper: dict[str, Any], interests: Sequence[str]) -> list[str]:
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

    topics = [str(interest).strip() for interest in interests if str(interest).strip()]
    for topic, keywords in topic_keywords.items():
        if any(keyword in text for keyword in keywords):
            topics.append(topic)

    if isinstance(categories, list):
        topics.extend(str(category) for category in categories if category)

    return unique_preserving_order(topics)[:8]


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


def render_daily_index(ranked_papers: Sequence[RankedPaper], target_date: date) -> str:
    front_matter = {
        "title": f"{target_date.isoformat()} Top20 Papers",
        "date": target_date.isoformat(),
        "tags": ["paper-agent", "ranking", "daily"],
    }
    lines = [
        render_front_matter(front_matter),
        f"# {target_date.isoformat()} Top20 Papers",
        "",
        "| Rank | Score | Paper |",
        "| ---: | ---: | --- |",
    ]
    for ranked in ranked_papers:
        title = str(ranked.paper.get("title", "Untitled Paper"))
        filename = f"{target_date.isoformat()}-{safe_filename(title)}"
        lines.append(f"| {ranked.rank} | {ranked.score:.6f} | [[{filename}|{title}]] |")
    lines.append("")
    return "\n".join(lines)


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    try:
        saved_paths = run_daily(args.config, today=args.date)
    except Exception as exc:
        logging.error("Daily pipeline failed: %s", exc)
        return 1

    for path in saved_paths:
        print(path)
    logging.info("Daily pipeline saved %s Markdown files.", len(saved_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
