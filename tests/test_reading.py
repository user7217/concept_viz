import json
import unittest

from atlas.reading import _title_from, external_links


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


if __name__ == "__main__":
    unittest.main()
