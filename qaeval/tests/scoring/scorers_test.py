import os
import pytest
import unittest

from qaeval.scoring.scorers import LERC, ExactMatchF1, MetaScorer


_SUMMARY = 'Somewhere in me I knew it all along , there are all those moments when he stares into my eyes and his start to sparkle while this gorgeous grin spreads across his face . When he first started to do it I would ask him \" what ? What s funny ? \" he would always say nothing and attempt to divert his attention elsewhere .'
_QUESTIONS_LIST = [
    [
        'What\'s a possible reason the guy stares into the writer\'s eyes ?',
        'Why did he stare into the eyes ?',
        'Why did he stare into the eyes ?',
    ],
    [
        'Who said they knew it all along ?',
        'Who stared into someone\'s eyes ?'
    ]
]
_ANSWERS_LIST = [
    [
        'Because he likes her a lot .',
        'He seems to like her',
        'He seems to like her',
    ],
    [
        'I',
        'He did',
    ]
]
_PREDICTIONS_LIST = [
    [
        'He is a child and it\'s a very rare thing.',
        'He seems to like her',
        'He seems to like her',
    ],
    [
        'The main character',
        'He, the main character, did',
    ]
]
_PROBABILITIES_LIST = [
    [1.0, 1.0, 0.2],
    [1.0, 1.0],
]
_NULL_PROBABILITIES_LIST = [
    [0.0, 0.0, 0.8],
    [0.0, 0.0]
]

_EXPECTED_SCORES_LIST = [
    {'em': 1/3, 'f1': 0.3846153846, 'lerc': 1.9416077733},
    {'em': 0, 'f1': 1/3, 'lerc': 3.2519777417182922},
]
_EXPECTED_AVG_SCORE_LIST = {'em': 1/6, 'f1': 0.358974359, 'lerc': 2.5967927575}
_EXPECTED_INDIVIDUAL_SCORES_LIST = [
    [
        {'em': 0.0, 'f1': 0.15384615384615385, 'lerc': 0.8436512351036072},
        {'em': 1.0, 'f1': 1.0, 'lerc': 4.98117208480835},
        {'em': 0.0, 'f1': 0.0, 'lerc': 0.0},
    ],
    [
        {'em': 0.0, 'f1': 0.0, 'lerc': 1.8995610475540161},
        {'em': 0.0, 'f1': 2/3, 'lerc': 4.604394435882568},
    ]
]


class TestEMF1(unittest.TestCase):
    def test_emf1_single_ref(self):
        metric = ExactMatchF1()
        for i in range(len(_QUESTIONS_LIST)):
            scores, individual_scores = metric.score_single_ref(_SUMMARY, _QUESTIONS_LIST[i], _ANSWERS_LIST[i],
                                                                _PREDICTIONS_LIST[i],
                                                                _PROBABILITIES_LIST[i], _NULL_PROBABILITIES_LIST[i])
            for name in ['em', 'f1']:
                self.assertAlmostEqual(_EXPECTED_SCORES_LIST[i][name], scores[name], places=4)
                for expected, actual in zip(_EXPECTED_INDIVIDUAL_SCORES_LIST[i], individual_scores):
                    self.assertAlmostEqual(expected[name], actual[name], places=4)

    def test_emf1_multi_ref(self):
        metric = ExactMatchF1()
        scores, individual_scores = metric.score_multi_ref(_SUMMARY, _QUESTIONS_LIST, _ANSWERS_LIST, _PREDICTIONS_LIST,
                                                           _PROBABILITIES_LIST, _NULL_PROBABILITIES_LIST)
        for name in ['em', 'f1']:
            self.assertAlmostEqual(_EXPECTED_AVG_SCORE_LIST[name], scores[name], places=4)


@pytest.mark.skipif('LERC_MODEL' not in os.environ or 'LERC_PRETRAINED' not in os.environ, reason='LERC environment variables not set')
class TestLERC(unittest.TestCase):
    def test_lerc_single_ref(self):
        metric = LERC(model_path=os.environ['LERC_MODEL'],
                      pretrained_path=os.environ['LERC_PRETRAINED'],
                      cuda_device=0)
        for i in range(len(_QUESTIONS_LIST)):
            scores, individual_scores = metric.score_single_ref(_SUMMARY, _QUESTIONS_LIST[i], _ANSWERS_LIST[i],
                                                                _PREDICTIONS_LIST[i],
                                                                _PROBABILITIES_LIST[i], _NULL_PROBABILITIES_LIST[i])
            for name in ['lerc']:
                self.assertAlmostEqual(_EXPECTED_SCORES_LIST[i][name], scores[name], places=4)
                for expected, actual in zip(_EXPECTED_INDIVIDUAL_SCORES_LIST[i], individual_scores):
                    self.assertAlmostEqual(expected[name], actual[name], places=4)

    def test_emf1_multi_ref(self):
        metric = LERC(model_path=os.environ['LERC_MODEL'],
                      pretrained_path=os.environ['LERC_PRETRAINED'],
                      cuda_device=0)
        scores, individual_scores = metric.score_multi_ref(_SUMMARY, _QUESTIONS_LIST, _ANSWERS_LIST, _PREDICTIONS_LIST,
                                                           _PROBABILITIES_LIST, _NULL_PROBABILITIES_LIST)
        for name in ['lerc']:
            self.assertAlmostEqual(_EXPECTED_AVG_SCORE_LIST[name], scores[name], places=4)


@pytest.mark.skipif('LERC_MODEL' not in os.environ or 'LERC_PRETRAINED' not in os.environ, reason='LERC environment variables not set')
class TestMeta(unittest.TestCase):
    def test_meta_single_ref(self):
        emf1 = ExactMatchF1()
        lerc = LERC(model_path=os.environ['LERC_MODEL'],
                    pretrained_path=os.environ['LERC_PRETRAINED'],
                    cuda_device=0)
        metric = MetaScorer([emf1, lerc])
        for i in range(len(_QUESTIONS_LIST)):
            scores, individual_scores = metric.score_single_ref(_SUMMARY, _QUESTIONS_LIST[i], _ANSWERS_LIST[i],
                                                                _PREDICTIONS_LIST[i],
                                                                _PROBABILITIES_LIST[i], _NULL_PROBABILITIES_LIST[i])
            for name in ['em', 'f1', 'lerc']:
                self.assertAlmostEqual(_EXPECTED_SCORES_LIST[i][name], scores[name], places=4)
                for expected, actual in zip(_EXPECTED_INDIVIDUAL_SCORES_LIST[i], individual_scores):
                    self.assertAlmostEqual(expected[name], actual[name], places=4)

    def test_emf1_multi_ref(self):
        emf1 = ExactMatchF1()
        lerc = LERC(model_path=os.environ['LERC_MODEL'],
                    pretrained_path=os.environ['LERC_PRETRAINED'],
                    cuda_device=0)
        metric = MetaScorer([emf1, lerc])
        scores, individual_scores = metric.score_multi_ref(_SUMMARY, _QUESTIONS_LIST, _ANSWERS_LIST, _PREDICTIONS_LIST,
                                                           _PROBABILITIES_LIST, _NULL_PROBABILITIES_LIST)
        for name in ['em', 'f1', 'lerc']:
            self.assertAlmostEqual(_EXPECTED_AVG_SCORE_LIST[name], scores[name], places=4)