# Thesis Agent

An Obsidian-based AI research assistant for arXiv paper ranking, summarization, and deep reading.

Thesis Agent is a Level 2 paper research pipeline. It fetches new arXiv papers,
ranks them against your research interests, summarizes the Top20, deeply reads
the Top2 PDFs, and saves Obsidian-ready Markdown notes.

## GitHub About

Recommended repository description:

```text
Obsidian-based AI research assistant for arXiv paper ranking, summarization, and deep reading.
```

Recommended topics:

```text
arxiv, openai, obsidian, research-agent, paper-ranking, literature-review, rag, llm-agents, ai-research, python
```

## Examples

Sample generated notes are available in `examples/`:

```text
examples/sample-daily.md
examples/sample-paper.md
examples/sample-deep.md
```

They contain no API keys or personal information.

## Project Structure

```text
paper-agent/
  src/
    paper_agent/
  tests/
  docker/
  examples/
  .github/workflows/
  README.md
  requirements.txt
  config.yaml
  .env.example
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Daily Workflow

When you log in to macOS:

1. LaunchAgent automatically starts Thesis Agent.
2. Yesterday's arXiv papers are fetched to avoid timezone and update delays.
3. Top20 papers are ranked.
4. Top5 Paper notes are generated.
5. Top2 Deep notes are generated.
6. Daily note is written to the Obsidian vault.

Recommended daily routine:

- Read Daily note: 2-3 min
- Read one Paper note
- Read one Deep note
- Write your own:
  - My Idea
  - Question
  - Criticism

The goal is not to read many papers, but to accumulate research insights.

## Configuration

Copy `.env.example` to `.env` and set your OpenAI API key.

```bash
cp .env.example .env
```

```text
OPENAI_API_KEY=sk-...
```

Do NOT wrap the key in unmatched quotes. Keep `.env` simple: one key-value pair per line.

Configure interests, ranking size, model names, and Obsidian output in
`config.yaml`.

```yaml
interests:
  - LLM Agents
  - Coding Agents
  - RAG
  - Memory
  - Reasoning
  - Multimodal
  - Robotics
  - Software Engineering
  - Long Context
  - Planning
  - MCP
  - Tool Use
  - AI Automation

fetch:
  category: cs.AI
  max_results: 100

ranking:
  candidate_k: 20
  save_k: 5
  deep_read_k: 2

obsidian:
  vault_path: "/Users/yourname/Documents/ObsidianVault"
  folder: "AI Papers"

models:
  embedding: text-embedding-3-small
  summary: gpt-4.1-mini
  deep_read: gpt-4.1-mini
```

## Run Daily Pipeline

Local run:

```bash
python daily.py --date YYYY-MM-DD
```

The pipeline:

1. Fetches today's `cs.AI` arXiv papers.
2. Embeds each paper's `title + abstract`.
3. Embeds your configured interests as a natural-language research profile.
4. Selects the Top20 candidates by cosine similarity.
5. Summarizes and saves only the Top5 paper notes.
6. Downloads and deeply analyzes only the Top2 PDFs.
7. Adds `My Insight`, `Can I Build It?`, `Startup Idea`, `Project Idea`, and `Related Topics`.
8. Saves a daily index plus selected paper/deep-read notes in your Obsidian vault.

Output structure:

```text
AI Papers/
  Daily/
    2025-06-16.md
  Papers/
    2506.xxxx-short-title.md
  Deep/
    2506.xxxx-short-title.md
  Weekly/
    2026-W25.md
  Monthly/
    2026-06.md
```

Each note includes dynamic YAML tags, an interest-level star score, buildability
fields, and paper-specific `Related Topics` written as Obsidian wikilinks such
as `[[NLP Evaluation]]` or `[[Educational AI]]`.

## Weekly Review

Generate a weekly AI research report from the last 7 days of `Daily/`, `Papers/`, and `Deep/` notes.

```bash
python weekly_review.py
```

Use an explicit week ending date when needed.

```bash
python weekly_review.py --week-ending YYYY-MM-DD
```

The report is saved to the Obsidian vault as:

```text
AI Papers/Weekly/YYYY-Www.md
```

## Weekly Workflow

Recommended:

- Read Weekly Summary
- Check Top Trends
- Check Top Deep Reads
- Review your ideas from this week

## Monthly Review

Generate a 30-day AI trend report from recent `Daily/`, `Papers/`, and `Deep/` notes.

```bash
python monthly_review.py
```

The report is saved to the Obsidian vault as:

```text
AI Papers/Monthly/YYYY-MM.md
```

## Monthly Workflow

Review:

- Major trends
- Emerging topics
- Research gaps
- My future direction

## Research Dashboard

Generate `AI Papers/Dashboard.md` from today's Daily note plus all saved
`Papers/` and `Deep/` notes.

```bash
python dashboard.py --config config.yaml
```

Use an explicit dashboard date when needed.

```bash
python dashboard.py --config config.yaml --date 2026-06-17
```

The dashboard includes `Must Read`, `Deep Read Queue`, `Project Queue`,
`Startup Queue`, `Idea Queue`, and exploration tables for all Paper and Deep
notes.

## Tests

```bash
PYTHONPATH=src:. pytest
```

CI:

- pytest only, no OpenAI API calls
- `python daily.py` is not executed in GitHub Actions
- `OPENAI_API_KEY` is only required for local pipeline runs
- tests use temporary paths instead of the local Obsidian vault path in `config.yaml`

## Fetch Today's arXiv Papers

Fetch up to 100 papers submitted today in `cs.AI` and save them as JSON.

```bash
python fetch.py
```

Use an explicit arXiv submission date when needed.

```bash
python fetch.py --date 2026-06-16 --max-results 100 --output papers_today.json
```

## Create Embeddings

Set `OPENAI_API_KEY` in `.env`, then create embeddings with the OpenAI
Embedding API.

```bash
python embedding.py --text "Graph neural networks for paper recommendation"
```

Embed multiple texts or write the result to stdout.

```bash
python embedding.py --text "first paper" --text "second paper" --output -
```

## Rank Embeddings

Calculate cosine similarity and return the top 20 closest embeddings.

```bash
python rank.py --embeddings embeddings.json --query-embedding-file query_embedding.json
```

You can also pass a query vector directly.

```bash
python rank.py --query-embedding "[0.1, 0.2, 0.3]" --top-k 20
```

## Save to Obsidian

Save fetched papers as Markdown notes in an Obsidian vault.

```bash
python save.py --input papers_today.json --vault ~/Documents/ObsidianVault
```

Save ranking output as a single Markdown note.

```bash
python save.py --input top20.json --mode ranking --vault ~/Documents/ObsidianVault
```

## GitHub Actions CI

GitHub Actions runs tests only. It does not execute the daily pipeline and does
not call the OpenAI API.

Workflow:

1. checkout
2. setup-python
3. `pip install -r requirements.txt`
4. `pip install pytest`
5. `pytest`

Run the real daily pipeline locally when you want to fetch papers, call OpenAI,
and write Obsidian Markdown:

```bash
python daily.py --date YYYY-MM-DD
```

## macOS Login Auto Run

Paper_Agent can run once per day when you log in to macOS. This uses a local
LaunchAgent and is not used by GitHub Actions.

The LaunchAgent is configured to run Paper_Agent from `/Users/zor125/Projects/Paper_Agent`. If you move the project directory later, update the scripts or reinstall after adjusting that path.

The automatic macOS launcher uses yesterday's date by default to avoid arXiv indexing delays.

Example:

```text
Login on:      2026-06-17
Pipeline date: 2026-06-16
```

The launcher:

- enters the project root
- waits 30 seconds for network readiness
- uses `.venv/bin/python` when it exists
- reads only `OPENAI_API_KEY` from `.env` or the shell environment
- runs `daily.py` with yesterday's date
- skips execution when `logs/last_run_YYYY-MM-DD` already exists
- writes stdout/stderr logs under `logs/`
- opens Obsidian after a successful run

Install:

```bash
chmod +x scripts/*.sh
./scripts/install_launchd.sh
```

Manual test:

```bash
launchctl start com.zor125.paperagent
```

Check logs:

```bash
cat logs/launchd.out.log
cat logs/launchd.err.log
cat logs/run_daily_$(date +%F).log
cat logs/run_daily_$(date -v-1d +%F).log
```

Remove:

```bash
./scripts/uninstall_launchd.sh
```

## Troubleshooting

If automatic execution does not work:

1. Check the LaunchAgent state.

```bash
launchctl print gui/$(id -u)/com.zor125.paperagent
```

2. Check generated logs.

```bash
ls logs/
```

3. Read the daily run log for the pipeline date.

```bash
cat logs/run_daily_YYYY-MM-DD.log
cat logs/run_daily_$(date -v-1d +%F).log
```

4. Check that:

- `.venv` exists
- `OPENAI_API_KEY` is valid
- `config.yaml` paths are correct
- the Obsidian vault path exists

## Philosophy

Thesis Agent is not designed to maximize the number of papers you read.

It is designed to maximize long-term research understanding.

The recommended workflow is:

```text
Daily
→ Papers
→ Deep
→ Weekly
→ Monthly
→ Personal Insights
```

Your own ideas, questions, and criticisms are the most valuable output of the system.

## Docker

```bash
docker build -f docker/Dockerfile -t paper-agent .
docker run --rm --env-file .env paper-agent
```
