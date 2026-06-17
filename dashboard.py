from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from daily import get_obsidian_vault_path, load_config
from save import render_front_matter
from weekly_review import DailyNoteSummary, DailyPaperRow, parse_daily_note, paper_link


@dataclass(frozen=True)
class NoteCard:
    title: str
    link: str
    note_type: str
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    project_idea: str = ""
    startup_idea: str = ""
    paper_type: str = ""


def run_dashboard(config_path: Path, *, target_date: date | None = None) -> Path:
    config = load_config(config_path)
    obsidian_config = config.get("obsidian", {})
    vault_dir = get_obsidian_vault_path(obsidian_config)
    folder = str(obsidian_config.get("folder", "papers"))
    base_dir = vault_dir / folder
    dashboard_date = target_date or date.today()
    daily_note = load_dashboard_daily_note(base_dir, dashboard_date)
    papers = load_note_cards(base_dir, "Papers")
    deep_notes = load_note_cards(base_dir, "Deep")
    markdown = render_dashboard(daily_note, papers, deep_notes, dashboard_date)
    output_path = base_dir / "Dashboard.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def load_dashboard_daily_note(base_dir: Path, target_date: date) -> DailyNoteSummary | None:
    daily_dir = base_dir / "Daily"
    exact_path = daily_dir / f"{target_date.isoformat()}.md"
    if exact_path.exists():
        return parse_daily_note(exact_path, base_dir=base_dir, note_date=target_date)
    candidates = []
    if daily_dir.exists():
        for path in daily_dir.glob("*.md"):
            try:
                note_date = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if note_date <= target_date:
                candidates.append((note_date, path))
    if not candidates:
        return None
    note_date, path = max(candidates, key=lambda item: item[0])
    return parse_daily_note(path, base_dir=base_dir, note_date=note_date)


def load_note_cards(base_dir: Path, folder: str) -> list[NoteCard]:
    note_dir = base_dir / folder
    if not note_dir.exists():
        return []
    cards = []
    for path in sorted(note_dir.glob("*.md")):
        cards.append(parse_note_card(path, base_dir=base_dir, note_type=folder.rstrip("s")))
    return cards


def parse_note_card(path: Path, *, base_dir: Path, note_type: str) -> NoteCard:
    text = path.read_text(encoding="utf-8")
    title = extract_title(text) or path.stem
    tags = extract_frontmatter_tags(text)
    paper_type = extract_frontmatter_value(text, "paper_type")
    summary = first_non_empty_section(text, ["One Sentence Summary", "TL;DR", "For Me", "Research Position"])
    project_idea = first_non_empty_section(text, ["Better Project Idea", "Project Idea", "Can I Build It?"])
    startup_idea = first_non_empty_section(text, ["Better Startup Idea", "Startup Idea"])
    return NoteCard(
        title=title.replace("Deep Read - ", ""),
        link=str(path.relative_to(base_dir).with_suffix("")),
        note_type=note_type,
        tags=tags,
        summary=compact_text(summary),
        project_idea=compact_text(project_idea),
        startup_idea=compact_text(startup_idea),
        paper_type=paper_type,
    )


def render_dashboard(
    daily_note: DailyNoteSummary | None,
    papers: list[NoteCard],
    deep_notes: list[NoteCard],
    target_date: date,
) -> str:
    front_matter = {
        "title": "Research Dashboard",
        "date": target_date.isoformat(),
        "tags": ["paper-agent", "dashboard", "research"],
    }
    lines = [
        render_front_matter(front_matter),
        "# Research Dashboard",
        "",
        "## Today",
        "",
        f"- Date: {target_date.isoformat()}",
        f"- Daily Note: {daily_note_link(daily_note) if daily_note else 'No Daily note found.'}",
        f"- Papers indexed: {len(papers)}",
        f"- Deep notes indexed: {len(deep_notes)}",
        "",
        "## Must Read",
        "",
        *render_must_read_queue(daily_note),
        "",
        "## Deep Read Queue",
        "",
        *render_deep_read_queue(daily_note, deep_notes),
        "",
        "## Project Queue",
        "",
        *render_project_queue(daily_note, papers),
        "",
        "## Startup Queue",
        "",
        *render_startup_queue(papers),
        "",
        "## Idea Queue",
        "",
        *render_idea_queue(daily_note, papers),
        "",
        "## Explore Papers",
        "",
        *render_note_table(papers),
        "",
        "## Explore Deep Reads",
        "",
        *render_note_table(deep_notes),
        "",
    ]
    return "\n".join(lines)


def render_must_read_queue(daily_note: DailyNoteSummary | None, limit: int = 5) -> list[str]:
    if not daily_note or not daily_note.papers:
        return ["No papers found for today's dashboard."]
    papers = sorted(daily_note.papers, key=lambda paper: (paper.deep_read, paper.saved, paper.score), reverse=True)
    return [
        f"- [ ] {paper_link(paper)} — {paper.score:.3f}, {paper.interest_level}"
        for paper in papers[:limit]
    ]


def render_deep_read_queue(daily_note: DailyNoteSummary | None, deep_notes: list[NoteCard], limit: int = 10) -> list[str]:
    deep_slugs = {card.link.split("/", 1)[-1] for card in deep_notes}
    queued: list[DailyPaperRow] = []
    if daily_note:
        queued = [
            paper for paper in daily_note.papers
            if paper.saved and not paper.deep_read and (paper.link or "").split("/", 1)[-1] not in deep_slugs
        ]
    if not queued:
        return ["No pending deep reads."]
    return [f"- [ ] {paper_link(paper)} — score {paper.score:.3f}" for paper in queued[:limit]]


def render_project_queue(daily_note: DailyNoteSummary | None, papers: list[NoteCard], limit: int = 10) -> list[str]:
    ideas = []
    if daily_note:
        ideas.extend(daily_note.project_ideas)
    ideas.extend(card.project_idea for card in papers if card.project_idea)
    unique = unique_preserving_order(ideas)
    if not unique:
        return ["No project ideas queued."]
    return [f"- [ ] {truncate_line(idea)}" for idea in unique[:limit]]


def render_startup_queue(papers: list[NoteCard], limit: int = 10) -> list[str]:
    ideas = unique_preserving_order(card.startup_idea for card in papers if card.startup_idea)
    if not ideas:
        return ["No startup ideas queued."]
    return [f"- [ ] {truncate_line(idea)}" for idea in ideas[:limit]]


def render_idea_queue(daily_note: DailyNoteSummary | None, papers: list[NoteCard], limit: int = 10) -> list[str]:
    ideas = []
    if daily_note:
        ideas.extend(daily_note.hidden_gems)
        ideas.extend(f"Explore trend: {trend}" for trend in daily_note.trends)
    ideas.extend(card.summary for card in papers if card.summary)
    unique = unique_preserving_order(ideas)
    if not unique:
        return ["No ideas queued."]
    return [f"- [ ] {truncate_line(idea)}" for idea in unique[:limit]]


def render_note_table(cards: list[NoteCard]) -> list[str]:
    if not cards:
        return ["No notes found."]
    lines = [
        "| Note | Type | Tags |",
        "| --- | --- | --- |",
    ]
    for card in cards:
        tags = ", ".join(f"`{tag}`" for tag in card.tags[:6]) if card.tags else ""
        lines.append(f"| [[{card.link}|{escape_table(card.title)}]] | {card.paper_type or card.note_type} | {tags} |")
    return lines


def daily_note_link(daily_note: DailyNoteSummary | None) -> str:
    if daily_note is None:
        return ""
    return f"[[Daily/{daily_note.note_date.isoformat()}|{daily_note.note_date.isoformat()}]]"


def extract_title(text: str) -> str:
    frontmatter_title = extract_frontmatter_value(text, "title")
    if frontmatter_title:
        return frontmatter_title
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def extract_frontmatter_value(text: str, key: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip('"')
    return ""


def extract_frontmatter_tags(text: str) -> list[str]:
    tags = []
    in_tags = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "tags:":
            in_tags = True
            continue
        if in_tags:
            if stripped.startswith("- "):
                tags.append(stripped[2:].strip().strip('"'))
            elif stripped and not line.startswith(" "):
                break
    return tags


def first_non_empty_section(text: str, headings: list[str]) -> str:
    for heading in headings:
        section = extract_section(text, f"## {heading}")
        if section:
            return section
    return ""


def extract_section(text: str, heading: str) -> str:
    if heading not in text:
        return ""
    after = text.split(heading, 1)[1]
    return after.split("\n## ", 1)[0].strip()


def compact_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*")):
            stripped = stripped[1:].strip()
        lines.append(stripped)
    return " ".join(lines)


def truncate_line(text: str, limit: int = 220) -> str:
    cleaned = compact_text(text)
    return cleaned if len(cleaned) <= limit else f"{cleaned[:limit].rstrip()}..."


def unique_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def escape_table(value: str) -> str:
    return value.replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Obsidian Research Dashboard.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = run_dashboard(args.config, target_date=args.date)
    print(f"Saved research dashboard: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
