"""Tests for source retrieval. No network: transports are injected."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.retrieval import (
    RetrievalError,
    Source,
    aliases_for,
    cited_but_not_retrieved,
    fetch_wikipedia,
    grounding_report,
    retrieve,
    retrieve_node,
    search_arxiv,
    title_variants,
)

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/1234.5678v1</id>
    <title>A Paper On Filtering</title><summary>We derive a filter.</summary></entry>
  <entry><id>http://arxiv.org/abs/9999.0001v1</id>
    <title>Another</title><summary>More.</summary></entry>
</feed>"""


def wiki(title: str, extract: str) -> bytes:
    return json.dumps(
        {"query": {"pages": {"42": {"title": title, "extract": extract}}}}
    ).encode()


MISSING = json.dumps({"query": {"pages": {"-1": {"title": "Nope"}}}}).encode()


class TestTitleVariants(unittest.TestCase):
    def test_title_case_gets_a_sentence_case_variant(self) -> None:
        self.assertEqual(
            title_variants("Covariance Matrix"), ["Covariance Matrix", "Covariance matrix"]
        )

    def test_single_word_has_one_variant(self) -> None:
        self.assertEqual(title_variants("Entropy"), ["Entropy"])

    def test_already_sentence_case_is_not_duplicated(self) -> None:
        self.assertEqual(title_variants("Kalman filter"), ["Kalman filter"])


class TestWikipedia(unittest.TestCase):
    def test_extract_is_returned(self) -> None:
        source = fetch_wikipedia(
            "Kalman filter", transport=lambda u: wiki("Kalman filter", "x" * 2000)
        )
        self.assertIsNotNone(source)
        self.assertEqual(source.id, "wikipedia:Kalman_filter")
        self.assertTrue(source.is_substantive)

    def test_missing_page_returns_none(self) -> None:
        self.assertIsNone(fetch_wikipedia("Nope", transport=lambda u: MISSING))

    def test_transport_failure_is_not_mistaken_for_a_missing_page(self) -> None:
        """A rate-limited request must never look like 'no such concept'.

        Conflating the two would license a fallback to recall, which is the
        failure design doc section 9 exists to prevent.
        """

        def throttled(url: str) -> bytes:
            raise RetrievalError("HTTP 429")

        with self.assertRaises(RetrievalError):
            fetch_wikipedia("Entropy", transport=throttled)

    def test_malformed_response_raises_rather_than_returning_none(self) -> None:
        with self.assertRaises(RetrievalError):
            fetch_wikipedia("X", transport=lambda u: b"not json")

    def test_empty_extract_is_treated_as_missing(self) -> None:
        self.assertIsNone(fetch_wikipedia("X", transport=lambda u: wiki("X", "   ")))


class TestArxiv(unittest.TestCase):
    def test_entries_are_parsed(self) -> None:
        sources = search_arxiv("kalman", transport=lambda u: ATOM.encode())
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].id, "1234.5678v1")
        self.assertEqual(sources[0].kind, "paper")

    def test_malformed_feed_returns_empty(self) -> None:
        self.assertEqual(search_arxiv("x", transport=lambda u: b"<feed"), [])

    def test_abstract_alone_is_not_substantive(self) -> None:
        sources = search_arxiv("kalman", transport=lambda u: ATOM.encode())
        self.assertFalse(sources[0].is_substantive)


class TestRetrieve(unittest.TestCase):
    def test_sentence_case_variant_is_tried_after_a_miss(self) -> None:
        calls = []

        def transport(url: str) -> bytes:
            calls.append(url)
            return MISSING if len(calls) == 1 else wiki("Covariance matrix", "y" * 2000)

        sources = retrieve("Covariance Matrix", transport=transport, include_papers=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sources[0].title, "Covariance matrix")

    def test_a_hit_stops_further_variants(self) -> None:
        calls = []

        def transport(url: str) -> bytes:
            calls.append(url)
            return wiki("Covariance Matrix", "y" * 2000)

        retrieve("Covariance Matrix", transport=transport, include_papers=False)
        self.assertEqual(len(calls), 1)

    def test_duplicate_sources_are_not_repeated(self) -> None:
        sources = retrieve(
            "Kalman filter", aliases=("Kalman Filter",),
            transport=lambda u: wiki("Kalman filter", "z" * 2000),
            include_papers=False,
        )
        self.assertEqual(len(sources), 1)


class TestAliasMap(unittest.TestCase):
    def test_alias_replaces_a_generated_name(self) -> None:
        self.assertEqual(aliases_for("linear_kalman_filter"), ["Kalman filter"])

    def test_empty_alias_means_known_to_have_no_source(self) -> None:
        self.assertEqual(aliases_for("anchor_box"), [])

    def test_unlisted_node_returns_none_not_empty(self) -> None:
        """None means 'no entry'; [] means 'known absent'. They differ."""
        self.assertIsNone(aliases_for("gaussian_conditioning"))

    def test_known_absent_node_makes_no_request(self) -> None:
        def transport(url: str) -> bytes:
            raise AssertionError("should not have been called")

        self.assertEqual(retrieve_node("anchor_box", "Anchor Box", transport), [])

    def test_alias_is_preferred_over_the_node_name(self) -> None:
        seen = []

        def transport(url: str) -> bytes:
            seen.append(url)
            return wiki("Kalman filter", "k" * 2000)

        retrieve_node("linear_kalman_filter", "Linear Kalman Filter", transport)
        self.assertIn("Kalman+filter", seen[0].replace("%20", "+"))


class TestGroundingGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = [
            Source("wikipedia:Kalman_filter", "Kalman filter", "u", "encyclopedia", "x" * 2000)
        ]

    def test_a_retrieved_citation_passes(self) -> None:
        self.assertEqual(
            cited_but_not_retrieved(["wikipedia:Kalman_filter"], self.sources), set()
        )

    def test_a_citation_from_memory_is_caught(self) -> None:
        """The whole point: a plausible reference nobody fetched must not pass."""
        self.assertEqual(
            cited_but_not_retrieved(
                ["Bar-Shalom, Estimation with Applications, 2001"], self.sources
            ),
            {"Bar-Shalom, Estimation with Applications, 2001"},
        )

    def test_no_sources_means_every_citation_is_ungrounded(self) -> None:
        self.assertEqual(cited_but_not_retrieved(["anything"], []), {"anything"})


class TestGroundingReport(unittest.TestCase):
    def test_abstract_only_is_not_grounded(self) -> None:
        thin = [Source("a", "t", "u", "paper", "short")]
        self.assertFalse(grounding_report("x", thin)["grounded"])

    def test_substantive_source_grounds_the_concept(self) -> None:
        full = [Source("a", "t", "u", "encyclopedia", "x" * 2000)]
        report = grounding_report("x", full)
        self.assertTrue(report["grounded"])
        self.assertEqual(report["best"], "a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
