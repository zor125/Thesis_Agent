from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from rank import cosine_similarity


@dataclass(frozen=True)
class MemoryEntry:
    title: str
    tags: list[str]
    embedding: list[float]
    summary: str
    paper_type: str
    project_idea: str
    link: str
    source_path: str
    note_type: str


@dataclass(frozen=True)
class MemoryRecommendation:
    title: str
    link: str
    score: float
    tags: list[str]
    paper_type: str
    project_idea: str


def load_memory_db(path: Path) -> list[MemoryEntry]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    entries = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        entries.append(
            MemoryEntry(
                title=str(item.get("title", "")),
                tags=[str(tag) for tag in item.get("tags", []) if str(tag).strip()],
                embedding=[float(value) for value in embedding],
                summary=str(item.get("summary", "")),
                paper_type=str(item.get("paper_type", "")),
                project_idea=str(item.get("project_idea", "")),
                link=str(item.get("link", "")),
                source_path=str(item.get("source_path", "")),
                note_type=str(item.get("note_type", "")),
            )
        )
    return entries


def save_memory_db(path: Path, entries: Sequence[MemoryEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(entry) for entry in entries]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def upsert_memory_entries(path: Path, new_entries: Sequence[MemoryEntry]) -> list[MemoryEntry]:
    entries_by_path = {entry.source_path: entry for entry in load_memory_db(path)}
    for entry in new_entries:
        if not entry.source_path or not entry.embedding:
            continue
        entries_by_path[entry.source_path] = entry
    entries = sorted(entries_by_path.values(), key=lambda entry: (entry.note_type, entry.title, entry.source_path))
    save_memory_db(path, entries)
    return entries


def recommend_memory_entries(
    entries: Sequence[MemoryEntry],
    query_embedding: Sequence[float] | None,
    *,
    exclude_paths: set[str] | None = None,
    limit: int = 5,
    min_score: float = 0.20,
) -> list[MemoryRecommendation]:
    if not query_embedding:
        return []
    exclude_paths = exclude_paths or set()
    scored = []
    for entry in entries:
        if entry.source_path in exclude_paths:
            continue
        try:
            score = cosine_similarity(list(query_embedding), entry.embedding)
        except Exception:
            continue
        if score < min_score:
            continue
        scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        MemoryRecommendation(
            title=entry.title,
            link=entry.link,
            score=score,
            tags=entry.tags,
            paper_type=entry.paper_type,
            project_idea=entry.project_idea,
        )
        for score, entry in scored[:limit]
    ]
