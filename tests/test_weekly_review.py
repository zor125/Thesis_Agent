from datetime import date

from weekly_review import load_weekly_daily_notes, parse_top20_rows, run_weekly_review


def test_weekly_review_generates_report_from_daily_notes_and_linked_papers(tmp_path, monkeypatch):
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    base_dir = tmp_path / "Vault" / "AI Papers"
    daily_dir = base_dir / "Daily"
    papers_dir = base_dir / "Papers"
    daily_dir.mkdir(parents=True)
    papers_dir.mkdir(parents=True)
    (papers_dir / "agent-paper.md").write_text(
        """---
title: "Agent Paper"
---
# Agent Paper

## Better Startup Idea

* Target Customer: AI engineering teams
* Pain Point: Agents are hard to evaluate
* MVP: Agent evaluation dashboard
* Revenue Model: SaaS
* Competitive Advantage: Research-backed benchmark
""",
        encoding="utf-8",
    )
    (daily_dir / "2026-06-15.md").write_text(
        """---
title: "Daily"
---
# 2026-06-15 Daily Paper Candidates

## 📈 Today's Research Trends

- Agent: 3 papers — agent trend
- RAG: 2 papers — rag trend

# 💎 Hidden Gem

- Paper: [[Papers/agent-paper|Agent Paper]]
- Reason: Novel benchmark angle.

## 💡 Today's Project

- Project: Build an agent benchmark dashboard.

## 💡 Today's Research Ideas

1. Idea: Can agent benchmark failures predict real deployment regressions?
   Based on: [[Papers/agent-paper|Agent Paper]]
   Why promising: Agent evaluation appeared repeatedly this week.
   Difficulty: ⭐⭐⭐☆☆
   First experiment: Label 20 agent failures and compare them with task success.

## Top20 Candidates

| Rank | Score | Interest Level | Title | Tags | Saved | Deep Read |
| ---: | ---: | :---: | --- | --- | :---: | :---: |
| 1 | 0.900000 | ⭐⭐⭐⭐⭐ | [[Papers/agent-paper|Agent Paper]] | `agent`, `benchmark` | Yes | Yes |
| 2 | 0.700000 | ⭐⭐⭐⭐☆ | RAG Paper | `rag` | No | No |
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
obsidian:
  vault_path: "{tmp_path / 'Vault'}"
  folder: "AI Papers"
""",
        encoding="utf-8",
    )

    output_path = run_weekly_review(config_path, week_ending=date(2026, 6, 17))
    markdown = output_path.read_text(encoding="utf-8")

    assert output_path == base_dir / "Weekly" / "2026-W25.md"
    assert "# Weekly AI Research Review" in markdown
    assert "## Top Papers" in markdown
    assert "[[Papers/agent-paper|Agent Paper]]" in markdown
    assert "## Top Trends" in markdown
    assert "- Agent:" in markdown
    assert "## Emerging Topics" in markdown
    assert "## Hidden Gems" in markdown
    assert "Novel benchmark angle." in markdown
    assert "## Research Ideas" in markdown
    assert "Can agent benchmark failures predict real deployment regressions?" in markdown
    assert "First experiment:" in markdown
    assert "## Project Ideas" in markdown
    assert "Build an agent benchmark dashboard." in markdown
    assert "## What I Should Read Next" in markdown
    assert "## Weekly Summary" in markdown
    assert "## Reading Statistics" in markdown
    assert "- Candidate papers: 2" in markdown


def test_load_weekly_daily_notes_filters_by_week(tmp_path):
    base_dir = tmp_path / "AI Papers"
    daily_dir = base_dir / "Daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "2026-06-10.md").write_text("# Old", encoding="utf-8")
    (daily_dir / "2026-06-17.md").write_text("# Current", encoding="utf-8")

    notes = load_weekly_daily_notes(base_dir, date(2026, 6, 17))

    assert [note.note_date for note in notes] == [date(2026, 6, 17)]


def test_parse_top20_rows_handles_wikilink_pipe_in_title_cell():
    rows = parse_top20_rows(
        """## Top20 Candidates

| Rank | Score | Interest Level | Title | Tags | Saved | Deep Read |
| ---: | ---: | :---: | --- | --- | :---: | :---: |
| 1 | 0.910000 | ⭐⭐⭐⭐⭐ | [[Papers/agent-paper|Agent Paper]] | `agent`, `benchmark` | Yes | No |
""",
        date(2026, 6, 17),
    )

    assert rows[0].title == "Agent Paper"
    assert rows[0].link == "Papers/agent-paper"
    assert rows[0].tags == ["agent", "benchmark"]
    assert rows[0].saved is True
    assert rows[0].deep_read is False
