from datetime import date

from monthly_review import load_monthly_daily_notes, run_monthly_review


def test_monthly_review_generates_trend_report_from_recent_daily_notes(tmp_path, monkeypatch):
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    base_dir = tmp_path / "Vault" / "AI Papers"
    daily_dir = base_dir / "Daily"
    papers_dir = base_dir / "Papers"
    daily_dir.mkdir(parents=True)
    papers_dir.mkdir(parents=True)
    (papers_dir / "agent-paper.md").write_text(
        """# Agent Paper

## Better Startup Idea

* Target Customer: AI engineering teams
* MVP: Agent evaluation dashboard
""",
        encoding="utf-8",
    )
    (daily_dir / "2026-05-25.md").write_text(
        """# Previous Window

## 📈 Today's Research Trends

- RAG: 5 papers — rag trend

## Top20 Candidates

| Rank | Score | Interest Level | Title | Tags | Saved | Deep Read |
| ---: | ---: | :---: | --- | --- | :---: | :---: |
| 1 | 0.810000 | ⭐⭐⭐⭐☆ | Old RAG Paper | `rag` | No | No |
""",
        encoding="utf-8",
    )
    (daily_dir / "2026-06-03.md").write_text(
        """# 2026-06-03 Daily Paper Candidates

## 📈 Today's Research Trends

- Agent: 4 papers — agent trend
- Benchmark: 2 papers — benchmark trend

# 💎 Hidden Gem

- Paper: [[Papers/agent-paper|Agent Paper]]
- Reason: Strong benchmark angle.

## 💡 Today's Project

- Project: Build a monthly agent benchmark map.

## Top20 Candidates

| Rank | Score | Interest Level | Title | Tags | Saved | Deep Read |
| ---: | ---: | :---: | --- | --- | :---: | :---: |
| 1 | 0.920000 | ⭐⭐⭐⭐⭐ | [[Papers/agent-paper|Agent Paper]] | `agent`, `benchmark` | Yes | Yes |
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
interests:
  - Agent
  - RAG
obsidian:
  vault_path: "{tmp_path / 'Vault'}"
  folder: "AI Papers"
""",
        encoding="utf-8",
    )

    output_path = run_monthly_review(config_path, month=date(2026, 6, 17))
    markdown = output_path.read_text(encoding="utf-8")

    assert output_path == base_dir / "Monthly" / "2026-06.md"
    assert "# Monthly AI Research Trend Report" in markdown
    assert 'period_start: "2026-05-19"' in markdown
    assert 'period_end: "2026-06-17"' in markdown
    assert "## Major Trends" in markdown
    assert "- Agent:" in markdown
    assert "## Rising Topics" in markdown
    assert "+" in markdown
    assert "## Declining Topics" in markdown
    assert "RAG" in markdown
    assert "## Important Papers" in markdown
    assert "[[Papers/agent-paper|Agent Paper]]" in markdown
    assert "## Research Gaps" in markdown
    assert "1. Gap:" in markdown
    assert "## Project Opportunities" in markdown
    assert "Build a monthly agent benchmark map." in markdown
    assert "## Startup Opportunities" in markdown
    assert "Agent evaluation dashboard" in markdown
    assert "## My Research Direction" in markdown
    assert "Agent" in markdown
    assert "## Reading Statistics" in markdown
    assert "- Candidate papers: 2" in markdown


def test_load_monthly_daily_notes_filters_recent_30_days(tmp_path):
    base_dir = tmp_path / "AI Papers"
    daily_dir = base_dir / "Daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "2026-05-18.md").write_text("# Too Old", encoding="utf-8")
    (daily_dir / "2026-05-19.md").write_text("# Included Start", encoding="utf-8")
    (daily_dir / "2026-06-17.md").write_text("# Included End", encoding="utf-8")
    (daily_dir / "2026-06-18.md").write_text("# Future", encoding="utf-8")

    notes = load_monthly_daily_notes(base_dir, date(2026, 6, 17))

    assert [note.note_date for note in notes] == [date(2026, 5, 19), date(2026, 6, 17)]
