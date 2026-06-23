from datetime import date
from urllib.parse import parse_qs, urlparse

from fetch import build_query_url, expanded_max_results, filter_by_published_date, parse_feed


def test_build_query_url_uses_category_only_query():
    url = build_query_url("cs.AI", 500)
    params = parse_qs(urlparse(url).query)

    assert params["max_results"] == ["500"]
    assert params["sortBy"] == ["submittedDate"]
    assert params["sortOrder"] == ["descending"]
    assert params["search_query"] == ["cat:cs.AI"]
    assert "submittedDate" not in params["search_query"][0]


def test_build_query_url_can_use_last_updated_sort():
    url = build_query_url("cs.AI", 500, sort_by="lastUpdatedDate")
    params = parse_qs(urlparse(url).query)

    assert params["sortBy"] == ["lastUpdatedDate"]


def test_expanded_max_results_fetches_extra_before_filtering():
    assert expanded_max_results(20) == 500
    assert expanded_max_results(200) == 1000


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


def test_filter_by_published_date_uses_utc_date():
    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2606.11111v1</id>
        <updated>2026-06-17T01:00:00Z</updated>
        <published>2026-06-16T23:59:00Z</published>
        <title>Included</title>
        <summary>Summary</summary>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2606.22222v1</id>
        <updated>2026-06-16T01:00:00Z</updated>
        <published>2026-06-15T23:59:00Z</published>
        <title>Excluded</title>
        <summary>Summary</summary>
      </entry>
    </feed>
    """

    papers = filter_by_published_date(parse_feed(feed), date(2026, 6, 16))

    assert [paper.title for paper in papers] == ["Included"]
