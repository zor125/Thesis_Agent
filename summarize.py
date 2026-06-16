from __future__ import annotations

from typing import Any

from dotenv import load_dotenv


DEFAULT_SUMMARY_MODEL = "gpt-4.1-mini"


def summarize_abstract(
    title: str,
    abstract: str,
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    client: Any | None = None,
) -> str:
    openai_client = client if client is not None else build_client()
    response = openai_client.responses.create(
        model=model,
        instructions=(
            "You summarize AI research papers for a Korean researcher. "
            "Return concise Korean Markdown with exactly these headings: "
            "## 한 줄 요약, ## 3줄 요약, ## 왜 중요한가, "
            "## 기존 연구와 차이, ## 내 프로젝트 적용 아이디어, "
            "## My Insight, ## Startup Idea, ## Project Idea, ## Related Topics. "
            "For Related Topics, return 3-7 concise bullet items such as RAG, Agent, RL, Robotics. "
            "Keep the answer practical and based only on the given abstract."
        ),
        input=f"Title: {title}\n\nAbstract:\n{abstract}",
    )
    return response.output_text.strip()


def build_client() -> Any:
    load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Install it with: pip install -r requirements.txt"
        ) from exc

    return OpenAI()
