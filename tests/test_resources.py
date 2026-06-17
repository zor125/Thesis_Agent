from resources import extract_code_resources, render_code_resources


def test_extract_code_resources_classifies_common_research_links():
    text = (
        "Official code is available at https://github.com/example/paper-code. "
        "A related implementation is https://github.com/example/reimplementation. "
        "Project page: https://example.github.io/paper. "
        "We release the dataset at https://huggingface.co/datasets/example/data "
        "and model at https://huggingface.co/example/model."
    )

    resources = extract_code_resources(text)

    assert resources.official_code == ["https://github.com/example/paper-code"]
    assert resources.related_implementation == ["https://github.com/example/reimplementation"]
    assert resources.project_page == ["https://example.github.io/paper"]
    assert "https://huggingface.co/datasets/example/data" in resources.huggingface
    assert "https://huggingface.co/example/model" in resources.huggingface
    assert resources.dataset == ["https://huggingface.co/datasets/example/data"]


def test_render_code_resources_can_omit_empty_section():
    resources = extract_code_resources("No code is mentioned here.")

    assert render_code_resources(resources) == []


def test_extract_code_resources_filters_incomplete_github_links_and_renders_related_implementation():
    resources = extract_code_resources(
        "Code: https://github.com/ and org page https://github.com/example. "
        "Official repository: https://github.com/example/official-code. "
        "Unofficial reimplementation: https://github.com/other/reimpl."
    )

    assert resources.official_code == ["https://github.com/example/official-code"]
    assert resources.related_implementation == ["https://github.com/other/reimpl"]

    markdown = "\n".join(render_code_resources(resources))

    assert "- Official Code: https://github.com/example/official-code" in markdown
    assert "- Related Implementation: https://github.com/other/reimpl" in markdown
    assert "https://github.com/" not in resources.official_code
    assert "https://github.com/example" not in resources.related_implementation
