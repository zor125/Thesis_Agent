from __future__ import annotations

import urllib.request
from pathlib import Path


def download_pdf(url: str, output_path: Path, *, timeout: int = 60) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "paper-agent/0.1.0 (mailto:example@example.com)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        output_path.write_bytes(response.read())
    return output_path


def extract_text_from_pdf(pdf_path: Path, *, max_pages: int | None = None) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "The pypdf package is required. Install it with: pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    text_parts = []
    for page in pages:
        text_parts.append(page.extract_text() or "")
    return "\n\n".join(part.strip() for part in text_parts if part.strip())


def chunk_text(text: str, *, max_chars: int = 12000, overlap: int = 500) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return []
    if max_chars < 1:
        raise ValueError("max_chars must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars.")

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = end - overlap
    return chunks
