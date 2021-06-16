from typing import List, Dict, Set, Tuple


class Scorer(object):
    def keys(self) -> Set[str]:
        raise NotImplementedError

    def default_scores(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.keys()}

    def score_single_ref(
        self,
        context: str,
        questions: List[str],
        answers: List[str],
        predictions: List[str],
        probabilities: List[float],
        null_probabilities: List[float]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        scores_dicts = self._score_single_ref(
            context,
            questions,
            answers,
            predictions,
            probabilities,
            null_probabilities
        )
        aggregated_scores = self.aggregate_scores(scores_dicts)
        return aggregated_scores, scores_dicts

    def _score_single_ref(
        self,
        context: str,
        questions: List[str],
        answers: List[str],
        predictions: List[str],
        probabilities: List[float],
        null_probabilities: List[float]
    ) -> List[Dict[str, float]]:
        raise NotImplementedError

    def score_multi_ref(
        self,
        context: str,
        questions_list: List[List[str]],
        answers_list: List[List[str]],
        predictions_list: List[List[str]],
        probabilities_list: List[List[float]],
        null_probabilities_list: List[List[float]]
    ) -> Tuple[Dict[str, float], List[List[Dict[str, float]]]]:
        # The aggregated per-reference scores
        reference_scores_list = []
        # The scores for each individual question. [i][j] will be the scores from
        # reference i and question j
        question_scores_list = []

        for i in range(len(questions_list)):
            reference_scores, question_scores = self.score_single_ref(
                context,
                questions_list[i],
                answers_list[i],
                predictions_list[i],
                probabilities_list[i],
                null_probabilities_list[i]
            )
            reference_scores_list.append(reference_scores)
            question_scores_list.append(question_scores)

        instance_scores = self.aggregate_scores(reference_scores_list)
        return instance_scores, question_scores_list

    def _ensure_expected_keys(self, expected_keys: Set[str], scores_dicts: List[Dict[str, float]]) -> None:
        for scores in scores_dicts:
            if expected_keys != scores.keys():
                raise Exception(f'Unequal keys: {expected_keys}; {scores.keys()}')

    def aggregate_scores(self, scores_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        if len(scores_dicts) == 0:
            return self.default_scores()

        expected_keys = self.keys()
        self._ensure_expected_keys(expected_keys, scores_dicts)
        sums = {key: 0.0 for key in expected_keys}
        for scores in scores_dicts:
            for key in expected_keys:
                sums[key] += scores[key]

        averages = {key: sums[key] / len(scores_dicts) for key in expected_keys}
        return averages