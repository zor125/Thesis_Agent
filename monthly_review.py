from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import yaml

from daily import get_interest_topics, get_obsidian_vault_path, load_config
from save import render_front_matter
from weekly_review import (
    DailyNoteSummary,
    DailyPaperRow,
    parse_daily_note,
    render_project_ideas,
    render_reading_statistics,
    render_top_papers,
)


def run_monthly_review(config_path: Path, *, month: date | None = None) -> Path:
    config = load_config(config_path)
    obsidian_config = config.get("obsidian", {})
    vault_dir = get_obsidian_vault_path(obsidian_config)
    folder = str(obsidian_config.get("folder", "papers"))
    base_dir = vault_dir / folder
    target_month = month or date.today()
    notes = load_monthly_daily_notes(base_dir, target_month)
    previous_notes = load_previous_monthly_daily_notes(base_dir, target_month)
    interests = load_interest_topics(config_path, config)
    markdown = render_monthly_report(notes, target_month, previous_notes=previous_notes, interests=interests)
    output_dir = base_dir / "Monthly"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target_month.strftime('%Y-%m')}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def load_monthly_daily_notes(base_dir: Path, month: date) -> list[DailyNoteSummary]:
    return load_recent_daily_notes(base_dir, month, days=30)


def load_previous_monthly_daily_notes(base_dir: Path, month: date) -> list[DailyNoteSummary]:
    return load_recent_daily_notes(base_dir, month - timedelta(days=30), days=30)


def load_recent_daily_notes(base_dir: Path, end_date: date, *, days: int) -> list[DailyNoteSummary]:
    daily_dir = base_dir / "Daily"
    if not daily_dir.exists():
        return []
    start_date = end_date - timedelta(days=days - 1)
    notes: list[DailyNoteSummary] = []
    for path in sorted(daily_dir.glob("*.md")):
        try:
            note_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if start_date <= note_date <= end_date:
            notes.append(parse_daily_note(path, base_dir=base_dir, note_date=note_date))
    return notes


def load_interest_topics(config_path: Path, config: dict) -> list[str]:
    interest_path = config_path.with_name("interest.yaml")
    if interest_path.exists():
        data = yaml.safe_load(interest_path.read_text(encoding="utf-8")) or {}
        interests = data.get("interests") if isinstance(data, dict) else data
        if isinstance(interests, list):
            return [str(item).strip() for item in interests if str(item).strip()]
        if isinstance(interests, dict):
            sentence = str(interests.get("sentence", "")).strip()
            return [sentence] if sentence else []
    return get_interest_topics(config)


def render_monthly_report(
    notes: Sequence[DailyNoteSummary],
    month: date,
    *,
    previous_notes: Sequence[DailyNoteSummary] | None = None,
    interests: Sequence[str] | None = None,
) -> str:
    previous_notes = previous_notes or []
    interests = interests or []
    period_start = month - timedelta(days=29)
    period_end = month
    month_label = month.strftime("%Y-%m")
    all_papers = [paper for note in notes for paper in note.papers]
    previous_papers = [paper for note in previous_notes for paper in note.papers]
    topic_counts = topic_frequency(notes, all_papers)
    previous_counts = topic_frequency(previous_notes, previous_papers)
    front_matter = {
        "title": f"{month_label} Monthly AI Trend Report",
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "tags": ["paper-agent", "monthly", "trend-report"],
    }
    return "\n".join(
        [
            render_front_matter(front_matter),
            "# Monthly AI Research Trend Report",
            "",
            "## Major Trends",
            "",
            *render_major_trends(topic_counts, previous_counts),
            "",
            "## Rising Topics",
            "",
            *render_rising_topics(topic_counts, previous_counts),
            "",
            "## Declining Topics",
            "",
            *render_declining_topics(topic_counts, previous_counts),
            "",
            "## Important Papers",
            "",
            *render_top_papers(all_papers, limit=10),
            "",
            "## Research Gaps",
            "",
            *render_research_gaps(topic_counts, all_papers),
            "",
            "## Project Opportunities",
            "",
            *render_project_opportunities(notes, topic_counts),
            "",
            "## Startup Opportunities",
            "",
            *render_startup_opportunities(notes, topic_counts),
            "",
            "## My Research Direction",
            "",
            *render_my_research_direction(topic_counts, interests),
            "",
            "## Reading Statistics",
            "",
            *render_monthly_reading_statistics(notes, all_papers, period_start, period_end),
            "",
        ]
    )


def topic_frequency(notes: Sequence[DailyNoteSummary], papers: Sequence[DailyPaperRow]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for note in notes:
        counts.update(normalize_topic(topic) for topic in note.trends)
    for paper in papers:
        counts.update(normalize_topic(tag) for tag in paper.tags)
    counts.pop("", None)
    return counts


def normalize_topic(value: str) -> str:
    cleaned = str(value).strip().replace("_", "-")
    canonical = {
        "agent": "Agent",
        "agents": "Agent",
        "rag": "RAG",
        "retrieval": "RAG",
        "benchmark": "Benchmark",
        "evaluation": "Evaluation",
        "dataset": "Dataset",
        "memory": "Memory",
        "reasoning": "Reasoning",
        "coding-agent": "Coding Agent",
        "software-engineering": "Software Engineering",
        "long-context": "Long Context",
        "multi-agent": "Multi Agent",
        "robotics": "Robotics",
        "mcp": "MCP",
        "tool-use": "Tool Use",
    }
    key = cleaned.lower()
    return canonical.get(key, " ".join(part.capitalize() for part in cleaned.replace("-", " ").split()))


def render_major_trends(current: Counter[str], previous: Counter[str], limit: int = 8) -> list[str]:
    if not current:
        return ["No Daily notes found for the last 30 days."]
    lines = []
    for topic, count in current.most_common(limit):
        delta = count - previous.get(topic, 0)
        lines.append(f"- {topic}: {count} mentions ({format_delta(delta)})")
    return lines


def render_rising_topics(current: Counter[str], previous: Counter[str], limit: int = 8) -> list[str]:
    rising = sorted(
        ((topic, count - previous.get(topic, 0), count) for topic, count in current.items()),
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    rising = [item for item in rising if item[1] > 0]
    if not rising:
        return ["No clear rising topics compared with the previous 30 days."]
    return [f"- {topic}: +{delta} mentions, now {count}" for topic, delta, count in rising[:limit]]


def render_declining_topics(current: Counter[str], previous: Counter[str], limit: int = 8) -> list[str]:
    declining = sorted(
        ((topic, previous_count - current.get(topic, 0), current.get(topic, 0)) for topic, previous_count in previous.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    declining = [item for item in declining if item[1] > 0]
    if not declining:
        return ["No clear declining topics compared with the previous 30 days."]
    return [f"- {topic}: -{drop} mentions, now {count}" for topic, drop, count in declining[:limit]]


def render_research_gaps(topic_counts: Counter[str], papers: Sequence[DailyPaperRow], limit: int = 5) -> list[str]:
    topics = [topic for topic, _count in topic_counts.most_common(8)]
    if not topics:
        return ["1. Gap: No topic data yet. Collect more Daily notes before identifying research gaps."]
    templates = [
        "How can {a} systems be evaluated with failure modes rather than aggregate scores?",
        "What minimal benchmark would expose the gap between {a} and real developer workflows?",
        "Can {a} methods be combined with {b} without increasing context noise or tool failures?",
        "Which ablation best explains when {a} helps and when it hurts?",
        "Can a one-week reproducibility protocol make {a} results easier to compare across papers?",
    ]
    gaps: list[str] = []
    for index, template in enumerate(templates[:limit]):
        a = topics[index % len(topics)]
        b = topics[(index + 1) % len(topics)] if len(topics) > 1 else "Evaluation"
        evidence = strongest_paper_for_topic(papers, a)
        based_on = f" Based on: {evidence}." if evidence else ""
        gaps.append(f"{index + 1}. Gap: {template.format(a=a, b=b)}{based_on}")
    return gaps


def strongest_paper_for_topic(papers: Sequence[DailyPaperRow], topic: str) -> str:
    topic_key = topic.lower()
    matches = [paper for paper in papers if any(normalize_topic(tag).lower() == topic_key for tag in paper.tags)]
    if not matches:
        return ""
    best = max(matches, key=lambda paper: (paper.score, paper.deep_read, paper.saved))
    return f"[[{best.link}|{best.title}]]" if best.link else best.title


def render_project_opportunities(notes: Sequence[DailyNoteSummary], topic_counts: Counter[str]) -> list[str]:
    ideas = render_project_ideas(notes)
    if ideas and not ideas[0].startswith("No project"):
        return ideas[:8]
    return [f"- Build a small {topic} evaluation dashboard from this month’s saved papers." for topic, _count in topic_counts.most_common(5)] or ["No project opportunities detected yet."]


def render_startup_opportunities(notes: Sequence[DailyNoteSummary], topic_counts: Counter[str]) -> list[str]:
    ideas: list[str] = []
    for note in notes:
        ideas.extend(note.startup_ideas)
    if ideas:
        return [f"- {idea}" for idea in unique_preserving_order(ideas)[:8]]
    if topic_counts:
        topic = topic_counts.most_common(1)[0][0]
        return [f"- Target AI teams that need practical {topic} monitoring, evaluation, or workflow automation."]
    return ["No startup opportunities detected yet."]


def render_my_research_direction(topic_counts: Counter[str], interests: Sequence[str]) -> list[str]:
    top_topics = [topic for topic, _count in topic_counts.most_common(8)]
    matched = []
    for interest in interests:
        interest_key = str(interest).lower()
        if any(interest_key in topic.lower() or topic.lower() in interest_key for topic in top_topics):
            matched.append(str(interest))
    focus = matched[:3] or top_topics[:3] or list(interests)[:3]
    if not focus:
        return ["- Focus: collect more Daily notes before choosing next month’s direction."]
    lines = [f"- Next month focus: {', '.join(focus)}"]
    if top_topics:
        lines.append(f"- Why: these interests overlap with the strongest monthly topics: {', '.join(top_topics[:5])}.")
    lines.append("- Action: choose one research gap above and run a one-week reproducibility or benchmark experiment.")
    return lines


def format_delta(delta: int) -> str:
    if delta > 0:
        return f"+{delta} vs previous 30 days"
    if delta < 0:
        return f"{delta} vs previous 30 days"
    return "no change vs previous 30 days"


def unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def render_monthly_reading_statistics(
    notes: Sequence[DailyNoteSummary],
    papers: Sequence[DailyPaperRow],
    month_start: date,
    month_end: date,
) -> list[str]:
    lines = render_reading_statistics(notes, papers, month_start, month_end)
    return [line.replace("- Week:", "- Period:") for line in lines]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a monthly AI trend report from recent Daily notes.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--month", type=date.fromisoformat, default=date.today(), help="Report end date; output uses YYYY-MM from this date.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = run_monthly_review(args.config, month=args.month)
    print(f"Saved monthly report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
