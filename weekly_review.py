from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from daily import get_obsidian_vault_path, load_config
from save import render_front_matter


WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


@dataclass(frozen=True)
class DailyPaperRow:
    rank: int
    score: float
    interest_level: str
    title: str
    link: str
    tags: list[str]
    saved: bool
    deep_read: bool
    source_date: date


@dataclass
class DailyNoteSummary:
    path: Path
    note_date: date
    papers: list[DailyPaperRow] = field(default_factory=list)
    trends: list[str] = field(default_factory=list)
    hidden_gems: list[str] = field(default_factory=list)
    project_ideas: list[str] = field(default_factory=list)
    startup_ideas: list[str] = field(default_factory=list)


def run_weekly_review(config_path: Path, *, week_ending: date | None = None) -> Path:
    config = load_config(config_path)
    obsidian_config = config.get("obsidian", {})
    vault_dir = get_obsidian_vault_path(obsidian_config)
    folder = str(obsidian_config.get("folder", "papers"))
    base_dir = vault_dir / folder
    target_end = week_ending or date.today()
    notes = load_weekly_daily_notes(base_dir, target_end)
    markdown = render_weekly_report(notes, target_end)
    output_dir = base_dir / "Weekly"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target_end.strftime('%G-W%V')}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def load_weekly_daily_notes(base_dir: Path, week_ending: date) -> list[DailyNoteSummary]:
    daily_dir = base_dir / "Daily"
    if not daily_dir.exists():
        return []
    week_start = week_ending - timedelta(days=6)
    notes: list[DailyNoteSummary] = []
    for path in sorted(daily_dir.glob("*.md")):
        try:
            note_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if week_start <= note_date <= week_ending:
            notes.append(parse_daily_note(path, base_dir=base_dir, note_date=note_date))
    return notes


def parse_daily_note(path: Path, *, base_dir: Path, note_date: date) -> DailyNoteSummary:
    text = path.read_text(encoding="utf-8")
    summary = DailyNoteSummary(
        path=path,
        note_date=note_date,
        papers=parse_top20_rows(text, note_date),
        trends=parse_trend_labels(text),
        hidden_gems=parse_hidden_gems(text),
        project_ideas=parse_project_ideas(text),
    )
    summary.startup_ideas = parse_startup_ideas_from_linked_papers(text, base_dir)
    return summary


def parse_top20_rows(text: str, note_date: date) -> list[DailyPaperRow]:
    rows: list[DailyPaperRow] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "## Top20 Candidates":
            in_table = True
            continue
        if in_table and stripped.startswith("## "):
            break
        if not in_table or not stripped.startswith("|") or "---" in stripped or "Rank" in stripped:
            continue
        cells = split_markdown_table_row(stripped)
        if len(cells) < 7:
            continue
        try:
            rank = int(cells[0])
            score = float(cells[1])
        except ValueError:
            continue
        title, link = parse_title_cell(cells[3])
        rows.append(
            DailyPaperRow(
                rank=rank,
                score=score,
                interest_level=cells[2],
                title=title,
                link=link,
                tags=parse_tags_cell(cells[4]),
                saved=cells[5].lower() == "yes",
                deep_read=cells[6].lower() == "yes",
                source_date=note_date,
            )
        )
    return rows


def split_markdown_table_row(line: str) -> list[str]:
    content = line.strip().strip("|")
    cells: list[str] = []
    current: list[str] = []
    in_wikilink = False
    index = 0
    while index < len(content):
        if content.startswith("[[", index):
            in_wikilink = True
            current.append("[[")
            index += 2
            continue
        if content.startswith("]]", index):
            in_wikilink = False
            current.append("]]")
            index += 2
            continue
        char = content[index]
        if char == "|" and not in_wikilink:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return cells


def parse_title_cell(value: str) -> tuple[str, str]:
    match = WIKILINK_PATTERN.search(value)
    if match:
        link = match.group(1).strip()
        title = (match.group(2) or match.group(1)).strip()
        return title, link
    return strip_markdown(value), ""


def parse_tags_cell(value: str) -> list[str]:
    tags = re.findall(r"`([^`]+)`", value)
    if tags:
        return tags
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def parse_trend_labels(text: str) -> list[str]:
    section = extract_section(text, "## 📈 Today's Research Trends")
    labels = []
    for line in section.splitlines():
        match = re.match(r"-\s*([^:]+):", line.strip())
        if match:
            labels.append(match.group(1).strip())
    return labels


def parse_hidden_gems(text: str) -> list[str]:
    section = extract_section(text, "# 💎 Hidden Gem")
    return compact_bullets(section)


def parse_project_ideas(text: str) -> list[str]:
    sections = [
        extract_section(text, "## 💡 Today's Project"),
        extract_section(text, "# 🚀 This Week Build"),
    ]
    ideas = []
    for section in sections:
        for line in section.splitlines():
            stripped = line.strip()
            if stripped.startswith("- Project:"):
                ideas.append(stripped.removeprefix("- Project:").strip())
    return unique_preserving_order(ideas)


def parse_startup_ideas_from_linked_papers(text: str, base_dir: Path) -> list[str]:
    ideas: list[str] = []
    for link, _label in WIKILINK_PATTERN.findall(text):
        if not link.startswith("Papers/"):
            continue
        note_path = base_dir / f"{link}.md"
        if not note_path.exists():
            continue
        note_text = note_path.read_text(encoding="utf-8")
        ideas.extend(parse_startup_ideas(note_text))
    return unique_preserving_order(ideas)


def parse_startup_ideas(text: str) -> list[str]:
    ideas = []
    for heading in ("## Better Startup Idea", "## Startup Idea"):
        section = extract_section(text, heading)
        if not section:
            continue
        lines = compact_bullets(section)
        if lines:
            ideas.append(" ".join(lines[:5]))
    return ideas


def render_weekly_report(notes: Sequence[DailyNoteSummary], week_ending: date) -> str:
    week_start = week_ending - timedelta(days=6)
    all_papers = [paper for note in notes for paper in note.papers]
    front_matter = {
        "title": f"{week_ending.strftime('%G-W%V')} Weekly AI Research Report",
        "week_start": week_start.isoformat(),
        "week_end": week_ending.isoformat(),
        "tags": ["paper-agent", "weekly", "research-report"],
    }
    lines = [
        render_front_matter(front_matter),
        f"# {week_ending.strftime('%G-W%V')} Weekly AI Research Report",
        "",
        "## Reading Statistics",
        "",
        *render_reading_statistics(notes, all_papers, week_start, week_ending),
        "",
        "## Top Papers",
        "",
        *render_top_papers(all_papers),
        "",
        "## Top Trends",
        "",
        *render_top_trends(notes, all_papers),
        "",
        "## Emerging Topics",
        "",
        *render_emerging_topics(notes, all_papers),
        "",
        "## Hidden Gems",
        "",
        *render_hidden_gems(notes),
        "",
        "## Project Ideas",
        "",
        *render_project_ideas(notes),
        "",
        "## Startup Ideas",
        "",
        *render_startup_ideas(notes),
        "",
    ]
    return "\n".join(lines)


def render_reading_statistics(
    notes: Sequence[DailyNoteSummary],
    papers: Sequence[DailyPaperRow],
    week_start: date,
    week_ending: date,
) -> list[str]:
    saved_count = sum(1 for paper in papers if paper.saved)
    deep_count = sum(1 for paper in papers if paper.deep_read)
    return [
        f"- Week: {week_start.isoformat()} to {week_ending.isoformat()}",
        f"- Daily notes: {len(notes)}",
        f"- Candidate papers: {len(papers)}",
        f"- Saved paper notes: {saved_count}",
        f"- Deep reads: {deep_count}",
        f"- Average candidates per day: {len(papers) / len(notes):.1f}" if notes else "- Average candidates per day: 0.0",
    ]


def render_top_papers(papers: Sequence[DailyPaperRow], limit: int = 10) -> list[str]:
    if not papers:
        return ["No Daily notes found for this week."]
    sorted_papers = sorted(papers, key=lambda paper: (paper.score, paper.deep_read, paper.saved), reverse=True)
    return [
        f"{index}. {paper_link(paper)} — {paper.score:.3f}, {paper.interest_level}, {paper.source_date.isoformat()}"
        for index, paper in enumerate(sorted_papers[:limit], start=1)
    ]


def render_top_trends(notes: Sequence[DailyNoteSummary], papers: Sequence[DailyPaperRow]) -> list[str]:
    counts = Counter()
    for note in notes:
        counts.update(note.trends)
    for paper in papers:
        counts.update(paper.tags)
    if not counts:
        return ["No recurring trends detected."]
    return [f"- {topic}: {count}" for topic, count in counts.most_common(10)]


def render_emerging_topics(notes: Sequence[DailyNoteSummary], papers: Sequence[DailyPaperRow]) -> list[str]:
    counts = Counter()
    latest_date = max((note.note_date for note in notes), default=None)
    latest_topics = Counter()
    for paper in papers:
        counts.update(paper.tags)
        if latest_date and paper.source_date == latest_date:
            latest_topics.update(paper.tags)
    emerging = [
        topic
        for topic, latest_count in latest_topics.items()
        if latest_count > 0 and counts[topic] <= 3
    ]
    if not emerging:
        return ["No clear emerging topics detected this week."]
    return [f"- {topic}: appeared late in the week with {counts[topic]} total mentions." for topic in emerging[:10]]


def render_hidden_gems(notes: Sequence[DailyNoteSummary]) -> list[str]:
    gems = unique_preserving_order(gem for note in notes for gem in note.hidden_gems)
    if not gems:
        return ["No hidden gems recorded."]
    return [f"- {gem}" for gem in gems[:10]]


def render_project_ideas(notes: Sequence[DailyNoteSummary]) -> list[str]:
    ideas = unique_preserving_order(idea for note in notes for idea in note.project_ideas)
    if not ideas:
        return ["No project ideas recorded."]
    return [f"- {idea}" for idea in ideas[:10]]


def render_startup_ideas(notes: Sequence[DailyNoteSummary]) -> list[str]:
    ideas = unique_preserving_order(idea for note in notes for idea in note.startup_ideas)
    if not ideas:
        return ["No startup ideas found in linked Paper notes."]
    return [f"- {idea}" for idea in ideas[:10]]


def paper_link(paper: DailyPaperRow) -> str:
    if paper.link:
        return f"[[{paper.link}|{paper.title}]]"
    return paper.title


def extract_section(text: str, heading: str) -> str:
    if heading not in text:
        return ""
    after = text.split(heading, 1)[1]
    return after.split("\n## ", 1)[0].split("\n# ", 1)[0].strip()


def compact_bullets(section: str) -> list[str]:
    lines = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            lines.append(strip_markdown(stripped[2:]))
        elif stripped.startswith("* "):
            lines.append(strip_markdown(stripped[2:]))
    return lines


def strip_markdown(value: str) -> str:
    return re.sub(r"`([^`]+)`", r"\1", value).strip()


def unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a weekly AI research report from Daily notes.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--week-ending", type=date.fromisoformat, default=date.today())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = run_weekly_review(args.config, week_ending=args.week_ending)
    print(f"Saved weekly report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
