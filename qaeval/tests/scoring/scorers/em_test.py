from qaeval.scoring.scorers import ExactMatchScorer
from qaeval.tests.scoring.scorers.scorer_test import TestScorer


class TestExactMatchScorer(TestScorer):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = ExactMatchScorer()

    def test_keys(self):
        assert self.scorer.keys() == {'em'}

    def test_default_scores(self):
        assert self.scorer.default_scores() == {'em': 0.0}

    def test_is_answered(self):
        # the transformer library accepts "a jogger" and "the jogger" for exact match
        self.assert_expected_output(
            self.scorer,
            {'em': (1 / 3 + 1 / 1) / 2},
            [{'em': 1 / 3}, {'em': 1 / 1}],
            [[{'em': 0.0}, {'em': 1.0}, {'em': 0.0}], [{'em': 1.0}]]
        )
