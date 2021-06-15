import os
import pytest

from qaeval.scoring.scorers import LERCScorer
from qaeval.tests.scoring.scorers.scorer_test import TestScorer


@pytest.mark.skipif('LERC_MODEL' not in os.environ or 'LERC_PRETRAINED' not in os.environ, reason='LERC environment variables not set')
class TestLERCScorer(TestScorer):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = LERCScorer(
            model_path=os.environ['LERC_MODEL'],
            pretrained_path=os.environ['LERC_PRETRAINED'],
            cuda_device=0
        )

    def test_keys(self):
        assert self.scorer.keys() == {'lerc'}

    def test_default_scores(self):
        assert self.scorer.default_scores() == {'lerc': 0.0}

    def test_is_answered(self):
        self.assert_expected_output(
            # This is a regression test. It does not ensure these numbers are correct
            self.scorer,
            {'lerc': (2.5152266025543213 + 4.940724849700928) / 2},
            [{'lerc': 2.5152266025543213}, {'lerc': 4.940724849700928}],
            [[{'lerc': 2.5210483074188232}, {'lerc': 5.024631500244141}, {'lerc': 0.0}], [{'lerc': 4.940724849700928}]]
        )
