from datetime import date

from monthly_review import load_monthly_daily_notes, run_monthly_review


def test_monthly_review_generates_report_from_daily_notes(tmp_path, monkeypatch):
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
    (daily_dir / "2026-06-03.md").write_text(
        """# 2026-06-03 Daily Paper Candidates

## 📈 Today's Research Trends

- Agent: 4 papers — agent trend

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
    (daily_dir / "2026-07-01.md").write_text("# July", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
obsidian:
  vault_path: "{tmp_path / 'Vault'}"
  folder: "AI Papers"
""",
        encoding="utf-8",
    )

    output_path = run_monthly_review(config_path, month=date(2026, 6, 17))
    markdown = output_path.read_text(encoding="utf-8")

    assert output_path == base_dir / "Monthly" / "2026-06.md"
    assert "# 2026-06 Monthly AI Research Report" in markdown
    assert "month_start: \"2026-06-01\"" in markdown
    assert "month_end: \"2026-06-03\"" in markdown
    assert "## Reading Statistics" in markdown
    assert "- Month: 2026-06-01 to 2026-06-03" in markdown
    assert "- Candidate papers: 1" in markdown
    assert "## Top Papers" in markdown
    assert "[[Papers/agent-paper|Agent Paper]]" in markdown
    assert "## Top Trends" in markdown
    assert "- Agent:" in markdown
    assert "## Hidden Gems" in markdown
    assert "Strong benchmark angle." in markdown
    assert "## Project Ideas" in markdown
    assert "Build a monthly agent benchmark map." in markdown
    assert "## Startup Ideas" in markdown
    assert "Agent evaluation dashboard" in markdown
    assert "July" not in markdown


def test_load_monthly_daily_notes_filters_by_month(tmp_path):
    base_dir = tmp_path / "AI Papers"
    daily_dir = base_dir / "Daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "2026-05-31.md").write_text("# May", encoding="utf-8")
    (daily_dir / "2026-06-01.md").write_text("# June", encoding="utf-8")
    (daily_dir / "2026-07-01.md").write_text("# July", encoding="utf-8")

    notes = load_monthly_daily_notes(base_dir, date(2026, 6, 17))

    assert [note.note_date for note in notes] == [date(2026, 6, 1)]
