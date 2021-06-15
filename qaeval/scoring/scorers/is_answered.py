from typing import Dict, List, Set

from qaeval.scoring.scorers import Scorer


class IsAnsweredScorer(Scorer):
    def keys(self) -> Set[str]:
        return {'is_answered'}

    def _score_single_ref(
        self,
        context: str,
        questions: List[str],
        answers: List[str],
        predictions: List[str],
        probabilities: List[float],
        null_probabilities: List[float]
    ) -> List[Dict[str, float]]:
        scores = []
        for prob, null_prob in zip(probabilities, null_probabilities):
            if prob > null_prob:
                scores.append({'is_answered': 1.0})
            else:
                scores.append({'is_answered': 0.0})
        return scores
