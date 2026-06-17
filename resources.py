from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence
from urllib.parse import urlparse


URL_PATTERN = re.compile(r"https?://[^\s<>)\]}\"']+")
TRAILING_PUNCTUATION = ".,;:!?)\"]}'"
DATASET_DOMAINS = ("huggingface.co/datasets", "kaggle.com", "zenodo.org", "figshare.com")
PROJECT_CONTEXT = ("project page", "project website", "homepage", "demo", "website")
DATASET_CONTEXT = ("dataset", "data set", "corpus", "benchmark", "data")


@dataclass(frozen=True)
class CodeResources:
    official_code: list[str] = field(default_factory=list)
    related_implementation: list[str] = field(default_factory=list)
    project_page: list[str] = field(default_factory=list)
    huggingface: list[str] = field(default_factory=list)
    dataset: list[str] = field(default_factory=list)

    def has_links(self) -> bool:
        return any([self.official_code, self.related_implementation, self.project_page, self.huggingface, self.dataset])


def extract_code_resources(*texts: str) -> CodeResources:
    github_candidates: list[tuple[str, str]] = []
    project_page: list[str] = []
    huggingface: list[str] = []
    dataset: list[str] = []

    for text in texts:
        if not text:
            continue
        for match in URL_PATTERN.finditer(text):
            url = clean_url(match.group(0))
            if not url:
                continue
            context = surrounding_context(text, match.start(), match.end()).lower()
            lowered_url = url.lower()

            if "github.com" in lowered_url:
                if is_valid_github_repo_url(url):
                    github_candidates.append((url, context))
                continue
            if "huggingface.co" in lowered_url:
                huggingface.append(url)
                if "huggingface.co/datasets" in lowered_url:
                    dataset.append(url)
                continue
            if is_project_page(lowered_url, context):
                project_page.append(url)
                continue
            if is_dataset_url(lowered_url, context):
                dataset.append(url)
                continue

    github_urls = unique_urls([url for url, _ in github_candidates])
    github_context_by_url = {url: context for url, context in github_candidates}
    official_code: list[str] = []
    related_implementation: list[str] = []
    if github_urls:
        official = max(
            github_urls,
            key=lambda candidate: github_official_score(candidate, github_context_by_url.get(candidate, "")),
        )
        official_code = [official]
        related_implementation = [url for url in github_urls if url != official]

    return CodeResources(
        official_code=official_code,
        related_implementation=related_implementation,
        project_page=unique_urls(project_page),
        huggingface=unique_urls(huggingface),
        dataset=unique_urls(dataset),
    )


def render_code_resources(resources: CodeResources, *, include_empty: bool = False) -> list[str]:
    if not resources.has_links() and not include_empty:
        return []

    return [
        "## Code / Resources",
        "",
        f"- Official Code: {format_resource_links(resources.official_code) if resources.official_code else 'No official code found.'}",
        f"- Related Implementation: {format_resource_links(resources.related_implementation) if resources.related_implementation else 'Not found.'}",
        f"- Project Page: {format_resource_links(resources.project_page) if resources.project_page else 'Not found.'}",
        f"- HuggingFace: {format_resource_links(resources.huggingface) if resources.huggingface else 'Not found.'}",
        f"- Dataset: {format_resource_links(resources.dataset) if resources.dataset else 'Not found.'}",
        "",
    ]


def clean_url(url: str) -> str:
    return url.strip().rstrip(TRAILING_PUNCTUATION)


def surrounding_context(text: str, start: int, end: int, window: int = 90) -> str:
    return text[max(0, start - window): min(len(text), end + window)]


def is_dataset_url(lowered_url: str, context: str) -> bool:
    return any(domain in lowered_url for domain in DATASET_DOMAINS) or any(keyword in context for keyword in DATASET_CONTEXT)


def is_project_page(lowered_url: str, context: str) -> bool:
    if any(domain in lowered_url for domain in ("arxiv.org", "doi.org")):
        return False
    return any(keyword in context for keyword in PROJECT_CONTEXT)


def is_valid_github_repo_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return False
    owner, repo = parts[0], parts[1]
    if owner.lower() in {"topics", "collections", "features", "marketplace", "explore"}:
        return False
    if repo.lower() in {"topics", "collections", "features", "marketplace", "explore"}:
        return False
    return True


def github_official_score(url: str, context: str) -> int:
    lowered_context = context.lower()
    lowered_url = url.lower()
    score = 0
    if "official code" in lowered_context or "code is available" in lowered_context:
        score += 8
    if "official implementation" in lowered_context or "official repository" in lowered_context:
        score += 8
    if "github" in lowered_context:
        score += 2
    if "implementation" in lowered_context or "repo" in lowered_context or "repository" in lowered_context:
        score += 2
    if "baseline" in lowered_context or "unofficial" in lowered_context or "reimplementation" in lowered_context:
        score -= 4
    if any(segment in lowered_url for segment in ("/paper", "/code", "/official", "/project")):
        score += 1
    return score


def format_resource_links(urls: Sequence[str]) -> str:
    return ", ".join(urls)


def unique_urls(urls: Sequence[str]) -> list[str]:
    seen = set()
    unique = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique
