import json

from save import (
    normalize_paper_type,
    paper_filename,
    render_paper_markdown,
    render_ranking_markdown,
    safe_filename,
    save_papers,
    save_ranking,
)


def test_safe_filename_removes_obsidian_unfriendly_characters():
    assert safe_filename('2606.12345 A/B: "Test" [Draft]') == "2606.12345 A B Test Draft"


def test_paper_filename_uses_arxiv_id_without_version_and_short_slug():
    assert (
        paper_filename(
            "Unified Software Engineering Agent as AI Software Engineer: A Very Long Study",
            "2506.14683v2",
        )
        == "2506.14683-unified-software-engineering-agent"
    )


def test_normalize_paper_type_keeps_known_types_and_defaults_to_research():
    assert normalize_paper_type("survey") == "Survey"
    assert normalize_paper_type("Benchmark") == "Benchmark"
    assert normalize_paper_type("unknown") == "Research"


def test_render_paper_markdown_includes_front_matter_and_links():
    markdown = render_paper_markdown(
        {
            "arxiv_id": "2606.12345v1",
            "title": "Useful AI Paper",
            "authors": ["Ada Lovelace"],
            "summary": "A concise summary.",
            "published": "2026-06-16T00:00:00Z",
            "updated": "2026-06-16T00:00:00Z",
            "categories": ["cs.AI"],
            "entry_url": "http://arxiv.org/abs/2606.12345v1",
            "pdf_url": "http://arxiv.org/pdf/2606.12345v1",
        }
    )

    assert 'title: "Useful AI Paper"' in markdown
    assert "# Useful AI Paper" in markdown
    assert "- Authors: Ada Lovelace" in markdown
    assert "A concise summary." in markdown


def test_render_ranking_markdown_creates_table():
    markdown = render_ranking_markdown(
        [{"rank": 1, "score": 0.987654321, "text": "Best | paper"}],
        "Top 20 Papers",
    )

    assert "# Top 20 Papers" in markdown
    assert "| 1 | 0.987654 | Best \\| paper |" in markdown


def test_save_papers_writes_markdown_files(tmp_path):
    saved_paths = save_papers(
        [
            {
                "arxiv_id": "2606.12345v1",
                "title": "Useful AI Paper",
                "authors": [],
                "summary": "Summary",
                "categories": ["cs.AI"],
            }
        ],
        tmp_path,
    )

    assert len(saved_paths) == 1
    assert saved_paths[0].exists()
    assert saved_paths[0].suffix == ".md"
    assert saved_paths[0].name == "2606.12345-useful-ai-paper.md"


def test_save_ranking_writes_single_note(tmp_path):
    output_path = save_ranking(
        [{"rank": 1, "score": 1.0, "text": "Best paper"}],
        tmp_path,
        title="Ranking",
    )

    assert output_path == tmp_path / "papers" / "Ranking.md"
    assert json.dumps("Best paper")[1:-1] in output_path.read_text(encoding="utf-8")
