from qaeval.scoring.scorers import F1Scorer
from qaeval.tests.scoring.scorers.scorer_test import TestScorer


class TestF1Scorer(TestScorer):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = F1Scorer()

    def test_keys(self):
        assert self.scorer.keys() == {'f1'}

    def test_default_scores(self):
        assert self.scorer.default_scores() == {'f1': 0.0}

    def test_is_answered(self):
        self.assert_expected_output(
            self.scorer,
            {'f1': (1 / 3 + 1 / 1) / 2},
            [{'f1': 1 / 3}, {'f1': 1 / 1}],
            [[{'f1': 0.0}, {'f1': 1.0}, {'f1': 0.0}], [{'f1': 1.0}]]
        )
