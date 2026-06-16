from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence


DEFAULT_VAULT_DIR = Path("obsidian")
DEFAULT_FOLDER = "papers"


def save_papers(
    papers: Sequence[dict[str, Any]],
    vault_dir: Path,
    *,
    folder: str = DEFAULT_FOLDER,
) -> list[Path]:
    output_dir = vault_dir / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for paper in papers:
        title = str(paper.get("title", "Untitled Paper"))
        arxiv_id = str(paper.get("arxiv_id", "")).strip()
        filename = safe_filename(f"{arxiv_id} {title}" if arxiv_id else title)
        output_path = output_dir / f"{filename}.md"
        output_path.write_text(render_paper_markdown(paper), encoding="utf-8")
        saved_paths.append(output_path)

    return saved_paths


def save_ranking(
    ranked_items: Sequence[dict[str, Any]],
    vault_dir: Path,
    *,
    folder: str = DEFAULT_FOLDER,
    title: str = "Top 20 Papers",
) -> Path:
    output_dir = vault_dir / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{safe_filename(title)}.md"
    output_path.write_text(render_ranking_markdown(ranked_items, title), encoding="utf-8")
    return output_path


def render_paper_markdown(paper: dict[str, Any]) -> str:
    title = str(paper.get("title", "Untitled Paper"))
    authors = as_string_list(paper.get("authors"))
    categories = as_string_list(paper.get("categories"))

    front_matter = {
        "title": title,
        "arxiv_id": paper.get("arxiv_id", ""),
        "published": paper.get("published", ""),
        "updated": paper.get("updated", ""),
        "categories": categories,
        "tags": ["paper", "arxiv", *[category.replace(".", "-") for category in categories]],
    }

    lines = [
        render_front_matter(front_matter),
        f"# {title}",
        "",
        "## Metadata",
        "",
        f"- Authors: {', '.join(authors) if authors else 'Unknown'}",
        f"- arXiv ID: {paper.get('arxiv_id', '')}",
        f"- Published: {paper.get('published', '')}",
        f"- Updated: {paper.get('updated', '')}",
        f"- Categories: {', '.join(categories)}",
        f"- Entry: {paper.get('entry_url', '')}",
        f"- PDF: {paper.get('pdf_url', '')}",
        "",
        "## Summary",
        "",
        str(paper.get("summary", "")).strip(),
        "",
    ]
    return "\n".join(lines)


def render_ranking_markdown(ranked_items: Sequence[dict[str, Any]], title: str) -> str:
    front_matter = {
        "title": title,
        "created": date.today().isoformat(),
        "tags": ["paper-agent", "ranking"],
    }
    lines = [
        render_front_matter(front_matter),
        f"# {title}",
        "",
        "| Rank | Score | Paper |",
        "| ---: | ---: | --- |",
    ]

    for item in ranked_items:
        rank = item.get("rank", "")
        score = item.get("score", "")
        score_text = f"{float(score):.6f}" if isinstance(score, int | float) else str(score)
        text = escape_markdown_table(str(item.get("text", "")))
        lines.append(f"| {rank} | {score_text} | {text} |")

    lines.append("")
    return "\n".join(lines)


def render_front_matter(values: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in values.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {yaml_scalar(str(item))}")
        else:
            lines.append(f"{key}: {yaml_scalar(str(value))}")
    lines.append("---")
    return "\n".join(lines)


def yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def safe_filename(value: str, *, max_length: int = 120) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|#\[\]]+", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(".")
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:max_length].rstrip()


def as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def looks_like_paper(value: dict[str, Any]) -> bool:
    return "arxiv_id" in value or "summary" in value or "entry_url" in value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save paper-agent JSON output as Markdown files for Obsidian.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file from fetch.py or rank.py.",
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=DEFAULT_VAULT_DIR,
        help="Obsidian vault directory. Defaults to ./obsidian.",
    )
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLDER,
        help="Folder inside the vault. Defaults to papers.",
    )
    parser.add_argument(
        "--title",
        default="Top 20 Papers",
        help="Title for a ranking note.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "papers", "ranking"],
        default="auto",
        help="How to interpret the input JSON. Defaults to auto.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = load_json(args.input)
        if not isinstance(data, list):
            raise ValueError("Input JSON must be an array.")

        mode = args.mode
        if mode == "auto":
            first_item = data[0] if data else {}
            mode = "papers" if isinstance(first_item, dict) and looks_like_paper(first_item) else "ranking"

        if mode == "papers":
            saved_paths = save_papers(data, args.vault, folder=args.folder)
        else:
            saved_paths = [save_ranking(data, args.vault, folder=args.folder, title=args.title)]
    except Exception as exc:
        print(f"Save failed: {exc}", file=sys.stderr)
        return 1

    for path in saved_paths:
        print(path)
    print(f"Saved {len(saved_paths)} Markdown file(s) to {args.vault / args.folder}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
