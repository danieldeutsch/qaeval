import unittest
from typing import Dict, List

from qaeval.scoring.scorers import Scorer

SUMMARY = '(CNN)Singer-songwriter David Crosby hit a jogger with his car Sunday evening, a spokesman said. The accident happened in Santa Ynez, California, near where Crosby lives. Crosby was driving at approximately 50 mph when he struck the jogger, according to California Highway Patrol Spokesman Don Clotworthy. The posted speed limit was 55. The jogger suffered multiple fractures, and was airlifted to a hospital in Santa Barbara, Clotworthy said.'
REFERENCE = 'Accident happens in Santa Ynez, California, near where Crosby lives . The jogger suffered multiple fractures; his injuries are not believed to be life-threatening .'

QUESTIONS = [
    [
        'What happens in Santa Ynez, California, near where Crosby lives?',
        'Where in California does accident happen near where Crosby lives?',
        'What did the jogger suffer multiple fractures for that are not believed to be life-threatening?',
    ],
    [
        'Who suffered multiple fractures?',
    ]
]
ANSWERS = [
    ['Accident', 'Santa Ynez', 'injuries'],
    ['The jogger'],
]
ANSWER_OFFSETS = [
    [(0, 8), (20, 30), (115, 122)],
    [(70, 80)],
]
PREDICTIONS = [
    ['hit a jogger', 'Santa Ynez', 'hit'],
    ['a jogger'],
]
PREDICTION_OFFSETS = [
    [(36, 48), (121, 131), (36, 39)],
    [(40, 48)],
]
PROBABILITIES = [
    [0.8, 0.6, 0.3],
    [0.5]
]
NULL_PROBABILITIES = [
    [0.3, 0.2, 0.6],
    [0.1]
]


class TestScorer(unittest.TestCase):
    def assert_expected_output(
        self,
        scorer: Scorer,
        instance_scores: Dict[str, float],
        reference_scores_list: List[Dict[str, float]],
        question_scores_lists: List[List[Dict[str, float]]],
    ) -> None:
        for i in range(len(QUESTIONS)):
            actual_reference_scores, actual_question_scores_list = scorer.score_single_ref(
                SUMMARY,
                QUESTIONS[i],
                ANSWERS[i],
                PREDICTIONS[i],
                PROBABILITIES[i],
                NULL_PROBABILITIES[i]
            )
            for key in scorer.keys():
                self.assertAlmostEqual(reference_scores_list[i][key], actual_reference_scores[key], places=4)
                for expected, actual in zip(question_scores_lists[i], actual_question_scores_list):
                    self.assertAlmostEqual(expected[key], actual[key], places=4)

        actual_instance_scores, actual_question_scores_lists = scorer.score_multi_ref(
            SUMMARY,
            QUESTIONS,
            ANSWERS,
            PREDICTIONS,
            PROBABILITIES,
            NULL_PROBABILITIES
        )

        for key in scorer.keys():
            self.assertAlmostEqual(instance_scores[key], actual_instance_scores[key], places=4)
            for expected_list, actual_list in zip(question_scores_lists, actual_question_scores_lists):
                for expected, actual in zip(expected_list, actual_list):
                    self.assertAlmostEqual(expected[key], actual[key], places=4)
