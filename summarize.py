from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from save import normalize_tags


DEFAULT_SUMMARY_MODEL = "gpt-4.1-mini"


@dataclass(frozen=True)
class BuildPlan:
    difficulty: str = "⭐⭐⭐☆☆"
    time_estimate: str = "Unknown"
    need_gpu: str = "Unknown"
    need_dataset: str = "Unknown"
    undergraduate_friendly: str = "Unknown"
    suggested_mini_project: str = "Reproduce the core idea on a small public dataset."


@dataclass(frozen=True)
class PaperAnalysis:
    one_sentence_summary: str
    tldr: str
    key_contributions: str
    why_important: str
    difference_from_previous_work: str
    limitations: str
    my_insight: str
    can_i_build_it: BuildPlan = field(default_factory=BuildPlan)
    startup_idea: str = "No startup idea generated."
    project_idea: str = "No project idea generated."
    related_topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def summarize_abstract(
    title: str,
    abstract: str,
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    client: Any | None = None,
) -> str:
    return render_analysis_markdown(
        analyze_abstract(title, abstract, model=model, client=client)
    )


def analyze_abstract(
    title: str,
    abstract: str,
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    client: Any | None = None,
) -> PaperAnalysis:
    openai_client = client if client is not None else build_client()
    response = openai_client.responses.create(
        model=model,
        instructions=(
            "You summarize AI research papers for a Korean researcher. "
            "Return only strict JSON. Do not wrap it in Markdown. "
            "Required keys: one_sentence_summary, tldr, key_contributions, "
            "why_important, difference_from_previous_work, limitations, my_insight, "
            "can_i_build_it, startup_idea, project_idea, related_topics, tags. "
            "can_i_build_it must contain difficulty, time_estimate, need_gpu, "
            "need_dataset, undergraduate_friendly, suggested_mini_project. "
            "tags must be 3-6 lowercase topic tags. related_topics must be 3-6 "
            "paper-specific Obsidian topic names, not generic fixed interests. "
            "Write all human-readable analysis fields in Korean: "
            "one_sentence_summary, tldr, key_contributions, why_important, "
            "difference_from_previous_work, limitations, my_insight, all can_i_build_it values, "
            "startup_idea, and project_idea. "
            "Keep tags in lowercase kebab-case English. Related topics may be concise English "
            "or Korean noun phrases suitable for Obsidian wiki links. "
            "Keep all analysis practical and based only on the given title and abstract."
        ),
        input=f"Title: {title}\n\nAbstract:\n{abstract}",
    )
    return parse_analysis_json(response.output_text)


def parse_analysis_json(value: str) -> PaperAnalysis:
    payload = json.loads(strip_json_fence(value))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object from summary model.")

    build_plan = payload.get("can_i_build_it", {})
    if not isinstance(build_plan, dict):
        build_plan = {}

    return PaperAnalysis(
        one_sentence_summary=str(payload.get("one_sentence_summary", "")).strip(),
        tldr=str(payload.get("tldr", "")).strip(),
        key_contributions=str(payload.get("key_contributions", "")).strip(),
        why_important=str(payload.get("why_important", "")).strip(),
        difference_from_previous_work=str(payload.get("difference_from_previous_work", "")).strip(),
        limitations=str(payload.get("limitations", "")).strip(),
        my_insight=str(payload.get("my_insight", "")).strip(),
        can_i_build_it=BuildPlan(
            difficulty=str(build_plan.get("difficulty", "⭐⭐⭐☆☆")).strip(),
            time_estimate=str(build_plan.get("time_estimate", "Unknown")).strip(),
            need_gpu=str(build_plan.get("need_gpu", "Unknown")).strip(),
            need_dataset=str(build_plan.get("need_dataset", "Unknown")).strip(),
            undergraduate_friendly=str(build_plan.get("undergraduate_friendly", "Unknown")).strip(),
            suggested_mini_project=str(
                build_plan.get(
                    "suggested_mini_project",
                    "Reproduce the core idea on a small public dataset.",
                )
            ).strip(),
        ),
        startup_idea=str(payload.get("startup_idea", "")).strip(),
        project_idea=str(payload.get("project_idea", "")).strip(),
        related_topics=[
            str(topic).strip()
            for topic in payload.get("related_topics", [])
            if str(topic).strip()
        ][:6],
        tags=normalize_tags(
            [str(tag) for tag in payload.get("tags", []) if str(tag).strip()],
            max_tags=6,
        ),
    )


def strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def render_analysis_markdown(analysis: PaperAnalysis) -> str:
    return "\n\n".join(
        [
            "## One Sentence Summary\n\n" + analysis.one_sentence_summary,
            "## TL;DR\n\n" + analysis.tldr,
            "## Key Contributions\n\n" + analysis.key_contributions,
            "## Why Important\n\n" + analysis.why_important,
            "## Difference From Previous Work\n\n" + analysis.difference_from_previous_work,
            "## Limitations\n\n" + analysis.limitations,
            "## My Insight\n\n" + analysis.my_insight,
            "## Startup Idea\n\n" + analysis.startup_idea,
            "## Project Idea\n\n" + analysis.project_idea,
        ]
    )


def build_client() -> Any:
    load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Install it with: pip install -r requirements.txt"
        ) from exc

    return OpenAI()
