from datetime import date
from urllib.parse import parse_qs, urlparse

from fetch import build_query_url, parse_feed


def test_build_query_url_filters_category_and_date():
    url = build_query_url("cs.AI", date(2026, 6, 16), 100)
    params = parse_qs(urlparse(url).query)

    assert params["max_results"] == ["100"]
    assert params["sortBy"] == ["submittedDate"]
    assert params["sortOrder"] == ["descending"]
    assert params["search_query"] == [
        "cat:cs.AI AND submittedDate:[202606160000 TO 202606162359]"
    ]


def test_parse_feed_extracts_paper_fields():
    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2606.12345v1</id>
        <updated>2026-06-16T12:34:56Z</updated>
        <published>2026-06-16T12:34:56Z</published>
        <title> A   Useful AI Paper </title>
        <summary> This paper  does things. </summary>
        <author><name>Ada Lovelace</name></author>
        <author><name>Alan Turing</name></author>
        <category term="cs.AI" />
        <link href="http://arxiv.org/abs/2606.12345v1" rel="alternate" type="text/html" />
        <link title="pdf" href="http://arxiv.org/pdf/2606.12345v1" rel="related" type="application/pdf" />
      </entry>
    </feed>
    """

    papers = parse_feed(feed)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2606.12345v1"
    assert papers[0].title == "A Useful AI Paper"
    assert papers[0].authors == ["Ada Lovelace", "Alan Turing"]
    assert papers[0].categories == ["cs.AI"]
    assert papers[0].pdf_url == "http://arxiv.org/pdf/2606.12345v1"
