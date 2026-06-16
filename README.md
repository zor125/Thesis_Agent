# Thesis Agent

Thesis Agent is a Level 2 paper research pipeline. It fetches new arXiv papers,
ranks them against your research interests, summarizes the Top20, deeply reads
the Top3 PDFs, and saves Obsidian-ready Markdown notes.

## Project Structure

```text
paper-agent/
  src/
    paper_agent/
  tests/
  docker/
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

## Configuration

Copy `.env.example` to `.env` and set your OpenAI API key.

```bash
cp .env.example .env
```

```text
OPENAI_API_KEY=your_api_key_here
```

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
  top_k: 20
  deep_read_k: 3

obsidian:
  vault_path: "/Users/yourname/Documents/ObsidianVault"
  folder: "AI Papers"

models:
  embedding: text-embedding-3-small
  summary: gpt-4.1-mini
  deep_read: gpt-4.1-mini
```

## Run Daily Pipeline

```bash
python daily.py
```

The pipeline:

1. Fetches today's `cs.AI` arXiv papers.
2. Embeds each paper's `title + abstract`.
3. Embeds your configured interests as a natural-language research profile.
4. Selects the Top20 by cosine similarity.
5. Summarizes the Top20 from abstracts.
6. Downloads and deeply analyzes only the Top3 PDFs.
7. Adds `My Insight`, `Startup Idea`, `Project Idea`, and `Related Topics`.
8. Saves Markdown notes as `YYYY-MM-DD-title.md` in your Obsidian vault.

`Related Topics` are written as Obsidian wikilinks such as `[[RAG]]`,
`[[Agent]]`, and `[[Robotics]]`, so the graph view becomes useful over time.

## Tests

```bash
PYTHONPATH=src:. pytest
```

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

## Daily GitHub Action

The `Daily Paper Agent` workflow runs every day at 07:00 KST. It runs the full
pipeline in `daily.py` and uploads the generated Obsidian Markdown files as an
artifact. Add `OPENAI_API_KEY` as a GitHub Actions secret before enabling it.
It can also be started manually from the GitHub Actions tab.

In GitHub Actions, `OBSIDIAN_VAULT_PATH` is set to `obsidian` so the generated
Markdown can be uploaded as an artifact. Locally, `config.yaml` controls the
Obsidian vault path unless you set `OBSIDIAN_VAULT_PATH` yourself.

## Docker

```bash
docker build -f docker/Dockerfile -t paper-agent .
docker run --rm --env-file .env paper-agent
```
