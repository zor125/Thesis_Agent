from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pdf_utils import chunk_text, download_pdf, extract_text_from_pdf
from resources import extract_code_resources, render_code_resources
from summarize import build_client


DEFAULT_DEEP_READ_MODEL = "gpt-4.1-mini"
MAX_ANALYSIS_CHARS = 90000
PDF_CHUNK_CHARS = 12000
PAPER_TYPES = {"Survey", "Research", "Benchmark", "Dataset", "System", "Position"}


@dataclass(frozen=True)
class DeepReadResult:
    paper_type: str
    markdown: str


def analyze_full_text(
    title: str,
    full_text: str,
    *,
    abstract: str = "",
    model: str = DEFAULT_DEEP_READ_MODEL,
    client: Any | None = None,
) -> DeepReadResult:
    openai_client = client if client is not None else build_client()
    chunks = chunk_text(full_text, max_chars=PDF_CHUNK_CHARS)
    clipped_text = "\n\n".join(chunks)[:MAX_ANALYSIS_CHARS]
    response = openai_client.responses.create(
        model=model,
        instructions=(
            "You deeply analyze AI research papers for a Korean researcher. "
            "First classify paper_type as exactly one of: Survey, Research, Benchmark, Dataset, System, Position. "
            "Use the title, abstract, and PDF text. "
            "Return Korean Markdown with these headings in this order: "
            "## Paper Type, then the type-specific analysis headings, then "
            "## For Me, ## Can I Build It?, ## Key Figure / Core Diagram, "
            "## Reading Path, ## 대학생 프로젝트 아이디어, ## 스타트업 아이디어, "
            "## Evaluation, ## Next Action, then the Research Notebook sections: "
            "## Research Position, ## Comparison Table, ## If I Were Building This, "
            "## Open Questions, ## Future Work Ideas. "
            "In ## Paper Type, write '- Type: <paper_type>' and one Korean sentence explaining why. "
            "Use the type-specific headings: "
            "Survey: ## Taxonomy, ## Covered Areas, ## Key Trends, ## Open Problems, ## Recommended Reading Path. "
            "Research: ## Method, ## Experiments, ## Ablation, ## Results, ## Limitations. "
            "Benchmark: ## Task Definition, ## Dataset/Benchmark Design, ## Metrics, ## Baselines, ## Limitations. "
            "Dataset: ## Data Construction, ## Annotation, ## Evaluation, ## Possible Uses, ## Bias/Risks. "
            "System: ## Architecture, ## Components, ## Deployment, ## Scalability, ## Failure Modes. "
            "Position: ## Main Claim, ## Arguments, ## Counterarguments, ## Implications. "
            "In ## For Me, include exactly these bullets: Relevance to My Interests, "
            "Why it matters to me, How it connects to Agent/RAG/Reasoning/Coding Agent, "
            "Should I revisit this later. "
            "In ## Can I Build It?, include exactly these bullets: Difficulty, Time Estimate, "
            "Need GPU, Need Dataset, Need Fine-tuning, Solo Developer Possible, "
            "Recommended Tech Stack, Beginner Version, Advanced Version. "
            "Use 1-5 star ratings for Relevance to My Interests and Difficulty. "
            "In ## Key Figure / Core Diagram, summarize the core architecture or research flow "
            "as a compact text diagram using arrows. "
            "In ## Reading Path, recommend 3-5 concepts, papers, or benchmarks to read first. "
            "In ## Evaluation, include exactly these bullets: Novelty, Impact, "
            "Practicality, Reproducibility, Worth Reading. "
            "Each bullet must use a 1-5 star rating like ⭐⭐⭐☆☆ and add one sentence of evidence. "
            "If the paper is a survey, evaluate Novelty by the value of its organization, taxonomy, "
            "coverage, and synthesis rather than by whether it proposes a new method. "
            "If the paper is a research paper, evaluate based on methodology and experimental results. "
            "In ## Next Action, include exactly these checklist items: "
            "- [ ] Read full paper, - [ ] Find GitHub implementation, "
            "- [ ] Search related papers, - [ ] Build mini prototype, "
            "- [ ] Write blog summary, - [ ] Skip. "
            "Mark exactly 1-2 recommended actions with [x], leave the rest as [ ]. "
            "After the checklist, add one bullet starting with '- 추천 이유:' and explain in one Korean sentence. "
            "Prefer actions a university student or solo developer can complete within one week. "
            "In ## Research Position, explain where this paper sits in the broader research trajectory. "
            "In ## Comparison Table, compare with exactly 3 representative related papers in a Markdown table "
            "with columns: Paper, Planning, Memory, Tool, Benchmark. "
            "In ## If I Were Building This, propose concrete design improvements or implementation changes. "
            "In ## Open Questions, list 3-5 research questions that remain after reading. "
            "In ## Future Work Ideas, list concrete follow-up research directions. "
            "Be concrete and distinguish evidence from inference."
        ),
        input=f"Title: {title}\n\nAbstract:\n{abstract}\n\nFull text chunks:\n{clipped_text}",
    )
    markdown = ensure_paper_type_section(response.output_text.strip(), title=title, abstract=abstract, full_text=full_text)
    resources = extract_code_resources(abstract, full_text)
    if resources.has_links() and "## Code / Resources" not in markdown:
        markdown = f"{markdown.rstrip()}\n\n" + "\n".join(render_code_resources(resources)).rstrip()
    return DeepReadResult(paper_type=extract_paper_type(markdown, title=title, abstract=abstract, full_text=full_text), markdown=markdown)


def deep_read_pdf(
    title: str,
    pdf_url: str,
    work_dir: Path,
    *,
    abstract: str = "",
    model: str = DEFAULT_DEEP_READ_MODEL,
    client: Any | None = None,
    max_pages: int | None = None,
) -> DeepReadResult:
    pdf_path = download_pdf(pdf_url, work_dir / "paper.pdf")
    full_text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
    if not full_text.strip():
        raise ValueError("No extractable text found in PDF.")
    return analyze_full_text(title, full_text, abstract=abstract, model=model, client=client)


def extract_paper_type(markdown: str, *, title: str = "", abstract: str = "", full_text: str = "") -> str:
    patterns = [
        r"Type:\s*(Survey|Research|Benchmark|Dataset|System|Position)",
        r"paper_type:\s*(Survey|Research|Benchmark|Dataset|System|Position)",
    ]
    for pattern in patterns:
        match = re.search(pattern, markdown, flags=re.IGNORECASE)
        if match:
            return canonical_paper_type(match.group(1))
    return classify_paper_type_heuristic(title=title, abstract=abstract, full_text=full_text)


def ensure_paper_type_section(markdown: str, *, title: str, abstract: str, full_text: str) -> str:
    if "## Paper Type" in markdown:
        return markdown
    paper_type = classify_paper_type_heuristic(title=title, abstract=abstract, full_text=full_text)
    return f"## Paper Type\n\n- Type: {paper_type}\n- 근거: 제목, 초록, 본문 단서에 기반한 fallback 분류입니다.\n\n{markdown}"


def classify_paper_type_heuristic(*, title: str = "", abstract: str = "", full_text: str = "") -> str:
    text = f"{title}\n{abstract}\n{full_text[:4000]}".lower()
    if any(keyword in text for keyword in ["survey", "review", "taxonomy", "overview"]):
        return "Survey"
    if any(keyword in text for keyword in ["benchmark", "leaderboard", "baseline", "metrics"]):
        return "Benchmark"
    if any(keyword in text for keyword in ["dataset", "corpus", "annotation", "annotated"]):
        return "Dataset"
    if any(keyword in text for keyword in ["system", "architecture", "deployment", "pipeline"]):
        return "System"
    if any(keyword in text for keyword in ["position", "perspective", "vision", "manifesto"]):
        return "Position"
    return "Research"


def canonical_paper_type(value: str) -> str:
    normalized = value.strip().lower()
    for paper_type in PAPER_TYPES:
        if paper_type.lower() == normalized:
            return paper_type
    return "Research"
