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
    consumers_of,
    extract_passages,
    ground_from_consumers,
    technique_terms,
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
        """None means 'no entry'; [] means 'known absent'. They differ.

        Uses a synthetic id: any real node may later gain an alias entry, which
        would make this assert on a stale fixture rather than on the contract.
        """
        self.assertIsNone(aliases_for("not_a_canon_node_xyz"))
        self.assertEqual(aliases_for("anchor_box"), [])

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


class TestTechniqueTerms(unittest.TestCase):
    def test_generic_technique_nouns_are_dropped(self) -> None:
        """'derivation' is what makes it a technique name, not what to search for."""
        terms = technique_terms("importance_weight_derivation", "Importance Weight Derivation")
        self.assertIn("importance", terms)
        self.assertIn("weight", terms)
        self.assertNotIn("derivation", terms)

    def test_alias_terms_are_included(self) -> None:
        terms = technique_terms("matrix_inversion_lemma", "Matrix Inversion Lemma")
        self.assertIn("woodbury", terms)

    def test_short_tokens_are_dropped(self) -> None:
        self.assertNotIn("of", technique_terms("trace_of_a_matrix", "Trace Of A Matrix"))


class TestExtractPassages(unittest.TestCase):
    TEXT = (
        "Short line.\n"
        + "A paragraph about transpose identities that is long enough to count as a "
        "real paragraph rather than a stray fragment of text.\n"
        + "An unrelated paragraph concerning gardening which is also comfortably "
        "longer than the minimum length threshold used here.\n"
        + "Another transpose paragraph, mentioning transpose twice, and long enough "
        "to clear the minimum length bar comfortably."
    )

    def test_only_matching_paragraphs_are_returned(self) -> None:
        passages = extract_passages(self.TEXT, ["transpose"])
        self.assertEqual(len(passages), 2)
        self.assertTrue(all("transpose" in p.lower() for p in passages))

    def test_more_hits_ranks_higher(self) -> None:
        passages = extract_passages(self.TEXT, ["transpose"])
        self.assertIn("twice", passages[0])

    def test_short_fragments_are_skipped(self) -> None:
        self.assertNotIn("Short line.", extract_passages(self.TEXT, ["short"]))

    def test_no_terms_returns_nothing(self) -> None:
        self.assertEqual(extract_passages(self.TEXT, []), [])


class TestGroundFromConsumers(unittest.TestCase):
    def setUp(self) -> None:
        from atlas.bootstrap import build_graph, load

        self.graph = build_graph(load())

    def test_consumers_put_algorithms_first(self) -> None:
        """An algorithm's article is likelier to show the move performed."""
        from atlas.schema import NodeType

        consumers = consumers_of(self.graph, "transpose_identities")
        first = self.graph.nodes[consumers[0]].type
        self.assertIs(first, NodeType.ALGORITHM)

    def test_derived_source_records_its_provenance(self) -> None:
        body = (
            "Intro paragraph that is long enough to be considered a real paragraph "
            "for the purposes of this extraction test.\n"
            "Here the transpose of the Jacobian appears, and the order of "
            "multiplication is reversed accordingly in the derivation."
        )

        sources = ground_from_consumers(
            self.graph, "transpose_identities", "Transpose Identities",
            transport=lambda u: wiki("Backpropagation", body), limit=1,
        )
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertEqual(source.kind, "derived")
        self.assertEqual(source.derived_from, "wikipedia:Backpropagation")
        self.assertIsNotNone(source.via)
        self.assertIn("transpose", source.text.lower())

    def test_derived_passages_clear_a_lower_substantive_bar(self) -> None:
        """A narrowed passage is not held to whole-article length."""
        passage = Source("a#t", "t", "u", "derived", "x" * 400)
        article = Source("a", "t", "u", "encyclopedia", "x" * 400)
        self.assertTrue(passage.is_substantive)
        self.assertFalse(article.is_substantive)

    def test_no_matching_passage_yields_nothing_rather_than_a_stub(self) -> None:
        body = "A long paragraph about something else entirely, with no relevant move in it at all."
        sources = ground_from_consumers(
            self.graph, "softmax_differentiation", "Softmax Differentiation",
            transport=lambda u: wiki("Cross-entropy", body), limit=2,
        )
        self.assertEqual(sources, [])

    def test_derived_citations_still_pass_the_grounding_guard(self) -> None:
        derived = Source("wikipedia:X#t", "t", "u", "derived", "y" * 400,
                         derived_from="wikipedia:X", via="backpropagation")
        self.assertEqual(cited_but_not_retrieved(["wikipedia:X#t"], [derived]), set())
        self.assertEqual(
            cited_but_not_retrieved(["Some Textbook, 1999"], [derived]),
            {"Some Textbook, 1999"},
        )
