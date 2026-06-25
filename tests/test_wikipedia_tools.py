from unittest.mock import patch

from tools import (
    _should_return_raw_response,
    parse_studio_album_rows,
    wikipedia_studio_albums,
)

SAMPLE_WIKITEXT = """
== Discography ==
=== Studio albums ===
{| class="wikitable"
|-
|2005
|''Corazón Libre''
|-
|2009
|''[[Cantora, un Viaje Íntimo|Cantora 1]]''
|-
|2009
|''Cantora 2''
|}

=== Live albums ===
|-
|2002
|''Acústico''
"""


def test_should_return_raw_response_for_wikipedia_api():
    assert _should_return_raw_response(
        "https://en.wikipedia.org/w/api.php?action=parse&format=json",
        "text/html",
    )
    assert _should_return_raw_response(
        "https://example.com/data.json",
        "text/plain",
    )
    assert not _should_return_raw_response(
        "https://en.wikipedia.org/wiki/Mercedes_Sosa",
        "text/html",
    )


def test_parse_studio_album_rows_filters_by_section():
    rows = parse_studio_album_rows(SAMPLE_WIKITEXT)
    years = [year for year, _album in rows]
    assert years == [2005, 2009, 2009]
    assert "Corazón Libre" in rows[0][1]
    assert "Cantora 1" in rows[1][1]


def test_wikipedia_studio_albums_counts_in_range():
    with patch("tools._fetch_wikipedia_wikitext", return_value=SAMPLE_WIKITEXT):
        result = wikipedia_studio_albums("Mercedes Sosa", 2000, 2009)

    assert "between 2000 and 2009 inclusive: 3" in result
    assert "2005: Corazón Libre" in result
    assert "2009: Cantora 1" in result
