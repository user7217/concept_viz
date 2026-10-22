import json
import unittest

from atlas.reading import _title_from, cite, external_links


def _api(links):
    payload = {"query": {"pages": {"1": {"extlinks": [{"*": u} for u in links]}}}}
    return lambda url, timeout=None: json.dumps(payload)


class TestFurtherReading(unittest.TestCase):
    def test_teaching_material_comes_before_papers(self) -> None:
        out = external_links("X", transport=_api([
            "https://doi.org/10.1/abc",
            "https://www.cs.unc.edu/~welch/kalman/kalmanPaper.html",
        ]))
        self.assertEqual(out[0].kind, "course notes")

    def test_kinds_are_interleaved_not_exhausted(self) -> None:
        # taking the best domain first filled every slot with arXiv ids
        out = external_links("X", transport=_api([
            "https://arxiv.org/abs/1", "https://arxiv.org/abs/2",
            "https://arxiv.org/abs/3", "https://arxiv.org/abs/4",
            "https://mit.edu/notes.pdf",
        ]), limit=3)
        self.assertIn("course notes", [r.kind for r in out])

    def test_archives_and_shops_are_dropped(self) -> None:
        out = external_links("X", transport=_api([
            "https://web.archive.org/web/1/http://mit.edu/x.pdf",
            "https://books.google.com/books?id=1",
            "https://www.amazon.com/dp/123",
        ]))
        self.assertEqual(out, [])

    def test_the_same_document_is_not_listed_twice(self) -> None:
        out = external_links("X", transport=_api([
            "http://mit.edu/notes.pdf", "https://mit.edu/notes.pdf",
        ]))
        self.assertEqual(len(out), 1)

    def test_a_missing_article_is_empty_not_an_error(self) -> None:
        self.assertEqual(external_links("X", transport=lambda *a, **k: None), [])
        self.assertEqual(
            external_links("X", transport=lambda *a, **k: "not json"), [])

    def test_a_url_becomes_a_readable_label(self) -> None:
        self.assertIn("kalmanPaper",
                      _title_from("http://cs.unc.edu/welch/kalmanPaper.html"))
        self.assertIn("cs.unc.edu",
                      _title_from("http://cs.unc.edu/welch/kalmanPaper.html"))


ARXIV_FEED = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<title>Kalman Filtering
  on Manifolds</title><published>2016-03-14T00:00:00Z</published>
<author><name>Ross Hartley</name></author><author><name>Maani Ghaffari</name></author>
</entry></feed>"""


def _serve(routes):
    """A transport that answers by URL prefix and records what was asked."""
    asked = []

    def transport(url, *a, **k):
        asked.append(url)
        for prefix, body in routes.items():
            if url.startswith(prefix):
                return body
        raise AssertionError(f"unexpected fetch {url}")
    transport.asked = asked
    return transport


class TestRealTitles(unittest.TestCase):
    """The panel showed "node7" and "10.2307%2F2371328" for real papers."""

    def test_an_arxiv_link_is_named_by_arxiv(self) -> None:
        got = cite("https://arxiv.org/abs/1603.04166v2",
                   _serve({"http://export.arxiv.org/api/query?id_list=1603.04166": ARXIV_FEED}))
        self.assertEqual(got.title, "Kalman Filtering on Manifolds")
        self.assertEqual(got.byline, "Hartley, Ghaffari · 2016")

    def test_an_escaped_doi_is_unescaped_before_crossref(self) -> None:
        body = json.dumps({"message": {"title": ["Relations <i>between</i> two sets"],
                                       "author": [{"family": "Hotelling"}],
                                       "issued": {"date-parts": [[1936]]}}}).encode()
        serve = _serve({"https://api.crossref.org/works/10.2307/2333955": body})
        got = cite("https://doi.org/10.2307%2F2333955", serve)
        self.assertEqual((got.title, got.byline, got.via),
                         ("Relations between two sets", "Hotelling · 1936", "crossref"))

    def test_a_book_is_named_by_its_isbn(self) -> None:
        body = json.dumps({"title": "Computational Statistics",
                           "publish_date": "2009"}).encode()
        got = cite("https://www.springer.com/x/book/978-3-642-01688-2",
                   _serve({"https://openlibrary.org/isbn/9783642016882.json": body}))
        self.assertEqual((got.title, got.byline), ("Computational Statistics", "2009"))

    def test_a_course_page_is_named_by_its_title(self) -> None:
        page = b"<html><head><title>  Gaussian &amp; conditioning\n</title></head></html>"
        got = cite("http://fourier.eng.hmc.edu/e161/node7.html",
                   _serve({"http://fourier": page}))
        self.assertEqual(got.title, "Gaussian & conditioning")

    def test_a_title_naming_the_server_state_is_not_a_title(self) -> None:
        for junk in (b"<title>404 Not Found</title>", b"<title>Just a moment...</title>"):
            self.assertIsNone(cite("http://x.edu/a.html", _serve({"http://x.edu": junk})))

    def test_a_bot_wall_or_a_hostname_is_not_a_title(self) -> None:
        for junk in (b"<title>Client Challenge</title>",
                     b"<title>pubmed.ncbi.nlm.nih.gov</title>"):
            self.assertIsNone(cite("http://x.edu/a.html", _serve({"http://x.edu": junk})))

    def test_a_real_title_starting_with_a_section_word_survives(self) -> None:
        page = b"<title>Search Algorithms for Planning</title>"
        self.assertEqual(cite("http://x.edu/a.html", _serve({"http://x.edu": page})).title,
                         "Search Algorithms for Planning")

    def test_a_jstor_link_is_named_by_crossref(self) -> None:
        body = json.dumps({"message": {"title": ["On Estimation"]}}).encode()
        got = cite("https://www.jstor.org/stable/2984505",
                   _serve({"https://api.crossref.org/works/10.2307/2984505": body}))
        self.assertEqual(got.title, "On Estimation")

    def test_the_site_name_is_trimmed_off(self) -> None:
        page = (b"<title>Lecture 3: Multiplication and inverse matrices | Linear "
                b"Algebra | MIT OpenCourseWare</title>")
        got = cite("https://ocw.mit.edu/x", _serve({"https://ocw": page}))
        self.assertEqual(got.title, "Lecture 3: Multiplication and inverse matrices")
        page = b"<title>Norm -- from Wolfram MathWorld</title>"
        got = cite("https://mathworld.wolfram.com/Norm.html", _serve({"https://math": page}))
        self.assertEqual(got.title, "Norm -- from Wolfram MathWorld")   # "Norm" is too short to stand alone

    def test_a_pdf_is_not_fetched(self) -> None:
        serve = _serve({})
        self.assertIsNone(cite("http://web.mit.edu/6.041/lec22.pdf", serve))
        self.assertEqual(serve.asked, [])

    def test_an_unreachable_link_keeps_its_url_label(self) -> None:
        def down(url, *a, **k):
            raise TimeoutError("timed out")
        self.assertIsNone(cite("http://x.edu/a.html", down))


if __name__ == "__main__":
    unittest.main()
