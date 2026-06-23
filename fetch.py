from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
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


def build_query_url(category: str, max_results: int, *, sort_by: str = "submittedDate") -> str:
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    return f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"


def expanded_max_results(max_results: int) -> int:
    return max(max_results * 5, 500)


def filter_by_published_date(papers: Iterable[Paper], target_date: date) -> list[Paper]:
    return [paper for paper in papers if published_utc_date(paper.published) == target_date]


def published_utc_date(value: str) -> date | None:
    try:
        cleaned = value.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).astimezone(timezone.utc).date()
    except ValueError:
        return None


def debug_feed(url: str, raw_papers: list[Paper], filtered_papers: list[Paper], target_date: date) -> None:
    print(f"arXiv query URL: {url}", file=sys.stderr)
    print(f"raw feed entry count: {len(raw_papers)}", file=sys.stderr)
    print(f"filtered count: {len(filtered_papers)}", file=sys.stderr)
    for index, paper in enumerate(raw_papers[:10], start=1):
        print(
            f"entry {index}: title={paper.title} published={paper.published} updated={paper.updated}",
            file=sys.stderr,
        )
    if not filtered_papers:
        print(f"Fetched 0 papers after filtering by published date {target_date.isoformat()}", file=sys.stderr)


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
    parser.add_argument(
        "--sort-by",
        choices=["submittedDate", "lastUpdatedDate"],
        default="submittedDate",
        help="arXiv API sort field. Use lastUpdatedDate if submittedDate is unstable.",
    )
    parser.add_argument("--debug", action="store_true", help="Print arXiv URL and feed filtering details.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_results < 1:
        print("--max-results must be greater than 0.", file=sys.stderr)
        return 2

    output_path = None if str(args.output) == "-" else args.output
    raw_max_results = expanded_max_results(args.max_results)
    url = build_query_url(args.category, raw_max_results, sort_by=args.sort_by)
    raw_papers = parse_feed(fetch_feed(url, args.timeout))
    papers = filter_by_published_date(raw_papers, args.date)[: args.max_results]
    if args.debug:
        debug_feed(url, raw_papers, papers, args.date)
    elif not papers:
        print(f"Fetched 0 papers after filtering by published date {args.date.isoformat()}", file=sys.stderr)
    write_json(papers, output_path)

    destination = "stdout" if output_path is None else output_path
    print(f"Fetched {len(papers)} papers for {args.category} on {args.date} -> {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
