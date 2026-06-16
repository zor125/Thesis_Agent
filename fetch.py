from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    updated: str
    categories: list[str]
    pdf_url: str | None
    entry_url: str


def build_query_url(category: str, target_date: date, max_results: int) -> str:
    start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(target_date, time.max, tzinfo=timezone.utc)
    submitted_range = f"{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}"
    params = {
        "search_query": f"cat:{category} AND submittedDate:[{submitted_range}]",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"


def fetch_feed(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "paper-agent/0.1.0 (mailto:example@example.com)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_feed(feed_xml: bytes) -> list[Paper]:
    root = ElementTree.fromstring(feed_xml)
    return [parse_entry(entry) for entry in root.findall("atom:entry", ATOM_NS)]


def parse_entry(entry: ElementTree.Element) -> Paper:
    entry_url = text(entry, "atom:id")
    pdf_url = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = link.attrib.get("href")
            break

    return Paper(
        arxiv_id=entry_url.rsplit("/", 1)[-1],
        title=normalize_space(text(entry, "atom:title")),
        authors=[
            normalize_space(text(author, "atom:name"))
            for author in entry.findall("atom:author", ATOM_NS)
        ],
        summary=normalize_space(text(entry, "atom:summary")),
        published=text(entry, "atom:published"),
        updated=text(entry, "atom:updated"),
        categories=[
            category.attrib["term"]
            for category in entry.findall("atom:category", ATOM_NS)
            if "term" in category.attrib
        ],
        pdf_url=pdf_url,
        entry_url=entry_url,
    )


def text(element: ElementTree.Element, path: str) -> str:
    found = element.find(path, ATOM_NS)
    return "" if found is None or found.text is None else found.text


def normalize_space(value: str) -> str:
    return " ".join(value.split())


def write_json(papers: Iterable[Paper], output_path: Path | None) -> None:
    payload = [asdict(paper) for paper in papers]
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)

    if output_path is None:
        print(json_text)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch today's cs.AI papers from the arXiv API.",
    )
    parser.add_argument("--category", default="cs.AI", help="arXiv category to fetch.")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=datetime.now(timezone.utc).date(),
        help="Submission date in YYYY-MM-DD format. Defaults to today's UTC date.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum number of papers to fetch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("papers_today.json"),
        help="Output JSON path. Use '-' to print to stdout.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_results < 1:
        print("--max-results must be greater than 0.", file=sys.stderr)
        return 2

    output_path = None if str(args.output) == "-" else args.output
    url = build_query_url(args.category, args.date, args.max_results)
    papers = parse_feed(fetch_feed(url, args.timeout))
    write_json(papers, output_path)

    destination = "stdout" if output_path is None else output_path
    print(f"Fetched {len(papers)} papers for {args.category} on {args.date} -> {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
