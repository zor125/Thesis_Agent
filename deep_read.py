from __future__ import annotations

from pathlib import Path
from typing import Any

from pdf_utils import chunk_text, download_pdf, extract_text_from_pdf
from summarize import build_client


DEFAULT_DEEP_READ_MODEL = "gpt-4.1-mini"
MAX_ANALYSIS_CHARS = 90000
PDF_CHUNK_CHARS = 12000


def analyze_full_text(
    title: str,
    full_text: str,
    *,
    model: str = DEFAULT_DEEP_READ_MODEL,
    client: Any | None = None,
) -> str:
    openai_client = client if client is not None else build_client()
    chunks = chunk_text(full_text, max_chars=PDF_CHUNK_CHARS)
    clipped_text = "\n\n".join(chunks)[:MAX_ANALYSIS_CHARS]
    response = openai_client.responses.create(
        model=model,
        instructions=(
            "You deeply analyze AI research papers for a Korean researcher. "
            "Return Korean Markdown with exactly these headings: "
            "## 핵심 기여, ## 방법론, ## 실험 결과, ## 한계점, "
            "## 구현 난이도, ## 대학생 프로젝트 아이디어, ## 스타트업 아이디어. "
            "Be concrete and distinguish evidence from inference."
        ),
        input=f"Title: {title}\n\nFull text chunks:\n{clipped_text}",
    )
    return response.output_text.strip()


def deep_read_pdf(
    title: str,
    pdf_url: str,
    work_dir: Path,
    *,
    model: str = DEFAULT_DEEP_READ_MODEL,
    client: Any | None = None,
    max_pages: int | None = None,
) -> str:
    pdf_path = download_pdf(pdf_url, work_dir / "paper.pdf")
    full_text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
    if not full_text.strip():
        raise ValueError("No extractable text found in PDF.")
    return analyze_full_text(title, full_text, model=model, client=client)
