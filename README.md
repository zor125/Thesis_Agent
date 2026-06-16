# paper-agent

Paper Agent is a Python project scaffold for building an agent that can search,
read, summarize, and organize research papers.

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
python -m paper_agent
```

## Configuration

Copy `.env.example` to `.env` and fill in any required API keys.

```bash
cp .env.example .env
```

Application defaults live in `config.yaml`.

## Tests

```bash
pytest
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

The `Daily Paper Fetch` workflow runs every day at 07:00 KST, fetches up to
100 `cs.AI` papers, and uploads `papers_today.json` as an artifact. It can also
be started manually from the GitHub Actions tab.

## Docker

```bash
docker build -f docker/Dockerfile -t paper-agent .
docker run --rm --env-file .env paper-agent
```
