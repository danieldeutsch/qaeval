from qaeval.scoring.scorers import IsAnsweredScorer, F1Scorer, MetaScorer
from qaeval.tests.scoring.scorers.scorer_test import TestScorer


class TestMetaScorer(TestScorer):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = MetaScorer([
            IsAnsweredScorer(), F1Scorer(),
        ])

    def test_keys(self):
        assert self.scorer.keys() == {'is_answered', 'f1'}

    def test_default_scores(self):
        assert self.scorer.default_scores() == {'is_answered': 0.0, 'f1': 0.0}

    def test_is_answered(self):
        self.assert_expected_output(
            self.scorer,
            {'is_answered': (2 / 3 + 1 / 1) / 2, 'f1': (1 / 3 + 1 / 1) / 2},
            [{'is_answered': 2 / 3, 'f1': 1 / 3}, {'is_answered': 1 / 1, 'f1': 1 / 1}],
            [
                [{'is_answered': 1.0, 'f1': 0.0}, {'is_answered': 1.0, 'f1': 1.0}, {'is_answered': 0.0, 'f1': 0.0}],
                [{'is_answered': 1.0, 'f1': 1.0}]
            ]
        )
