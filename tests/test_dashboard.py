from datetime import date

from dashboard import run_dashboard


def test_dashboard_generates_queues_and_exploration_tables(tmp_path, monkeypatch):
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    base_dir = tmp_path / "Vault" / "AI Papers"
    daily_dir = base_dir / "Daily"
    papers_dir = base_dir / "Papers"
    deep_dir = base_dir / "Deep"
    daily_dir.mkdir(parents=True)
    papers_dir.mkdir(parents=True)
    deep_dir.mkdir(parents=True)
    (daily_dir / "2026-06-17.md").write_text(
        """# 2026-06-17 Daily Paper Candidates

## 📈 Today's Research Trends

- Agent: 5 papers — agent trend

# 💎 Hidden Gem

- Paper: [[Papers/agent-paper|Agent Paper]]
- Reason: Novel agent benchmark.

## 💡 Today's Project

- Project: Build a tool-use evaluation harness.

## Top20 Candidates

| Rank | Score | Interest Level | Title | Tags | Saved | Deep Read |
| ---: | ---: | :---: | --- | --- | :---: | :---: |
| 1 | 0.910000 | ⭐⭐⭐⭐⭐ | [[Papers/agent-paper|Agent Paper]] | `agent`, `benchmark` | Yes | No |
| 2 | 0.880000 | ⭐⭐⭐⭐⭐ | [[Papers/deep-agent|Deep Agent]] | `agent`, `planning` | Yes | Yes |
""",
        encoding="utf-8",
    )
    (papers_dir / "agent-paper.md").write_text(
        """---
title: "Agent Paper"
tags:
  - "agent"
  - "benchmark"
---
# Agent Paper

## One Sentence Summary

Agent benchmark summary.

## Better Project Idea

* Beginner: Build a small agent benchmark.

## Better Startup Idea

* MVP: Agent monitoring dashboard
""",
        encoding="utf-8",
    )
    (papers_dir / "deep-agent.md").write_text(
        """---
title: "Deep Agent"
tags:
  - "agent"
  - "planning"
---
# Deep Agent

## One Sentence Summary

Deep agent summary.
""",
        encoding="utf-8",
    )
    (deep_dir / "deep-agent.md").write_text(
        """---
title: "Deep Read - Deep Agent"
paper_type: "Research"
tags:
  - "deep-read"
  - "agent"
---
# Deep Read - Deep Agent
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

    output_path = run_dashboard(config_path, target_date=date(2026, 6, 17))
    markdown = output_path.read_text(encoding="utf-8")

    assert output_path == base_dir / "Dashboard.md"
    assert "# Research Dashboard" in markdown
    assert "## Must Read" in markdown
    assert "[[Papers/agent-paper|Agent Paper]]" in markdown
    assert "## Deep Read Queue" in markdown
    assert "- [ ] [[Papers/agent-paper|Agent Paper]]" in markdown
    assert "## Project Queue" in markdown
    assert "Build a tool-use evaluation harness." in markdown
    assert "Build a small agent benchmark." in markdown
    assert "## Startup Queue" in markdown
    assert "Agent monitoring dashboard" in markdown
    assert "## Idea Queue" in markdown
    assert "Novel agent benchmark." in markdown
    assert "## Explore Papers" in markdown
    assert "| [[Papers/agent-paper|Agent Paper]]" in markdown
    assert "## Explore Deep Reads" in markdown
    assert "| [[Deep/deep-agent|Deep Agent]]" in markdown
