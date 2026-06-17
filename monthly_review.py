from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from daily import get_obsidian_vault_path, load_config
from save import render_front_matter
from weekly_review import (
    DailyNoteSummary,
    parse_daily_note,
    render_emerging_topics,
    render_hidden_gems,
    render_project_ideas,
    render_reading_statistics,
    render_startup_ideas,
    render_top_papers,
    render_top_trends,
)


def run_monthly_review(config_path: Path, *, month: date | None = None) -> Path:
    config = load_config(config_path)
    obsidian_config = config.get("obsidian", {})
    vault_dir = get_obsidian_vault_path(obsidian_config)
    folder = str(obsidian_config.get("folder", "papers"))
    base_dir = vault_dir / folder
    target_month = month or date.today()
    notes = load_monthly_daily_notes(base_dir, target_month)
    markdown = render_monthly_report(notes, target_month)
    output_dir = base_dir / "Monthly"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target_month.strftime('%Y-%m')}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def load_monthly_daily_notes(base_dir: Path, month: date) -> list[DailyNoteSummary]:
    daily_dir = base_dir / "Daily"
    if not daily_dir.exists():
        return []
    notes: list[DailyNoteSummary] = []
    for path in sorted(daily_dir.glob("*.md")):
        try:
            note_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if note_date.year == month.year and note_date.month == month.month:
            notes.append(parse_daily_note(path, base_dir=base_dir, note_date=note_date))
    return notes


def render_monthly_report(notes: list[DailyNoteSummary], month: date) -> str:
    month_start = month.replace(day=1)
    month_end = max((note.note_date for note in notes), default=month)
    month_label = month.strftime("%Y-%m")
    all_papers = [paper for note in notes for paper in note.papers]
    front_matter = {
        "title": f"{month_label} Monthly AI Research Report",
        "month_start": month_start.isoformat(),
        "month_end": month_end.isoformat(),
        "tags": ["paper-agent", "monthly", "research-report"],
    }
    return "\n".join(
        [
            render_front_matter(front_matter),
            f"# {month_label} Monthly AI Research Report",
            "",
            "## Reading Statistics",
            "",
            *render_monthly_reading_statistics(notes, all_papers, month_start, month_end),
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
    )


def render_monthly_reading_statistics(
    notes: list[DailyNoteSummary],
    papers: list,
    month_start: date,
    month_end: date,
) -> list[str]:
    lines = render_reading_statistics(notes, papers, month_start, month_end)
    return [line.replace("- Week:", "- Month:") for line in lines]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a monthly AI research report from Daily notes.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--month", type=date.fromisoformat, default=date.today(), help="Any date inside the target month.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = run_monthly_review(args.config, month=args.month)
    print(f"Saved monthly report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
