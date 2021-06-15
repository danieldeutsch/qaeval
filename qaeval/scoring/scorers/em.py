from transformers.data.metrics.squad_metrics import compute_exact
from typing import Dict, List, Set

from qaeval.scoring.scorers import Scorer


class ExactMatchScorer(Scorer):
    def keys(self) -> Set[str]:
        return {'em'}

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
        for prediction, answer, prob, null_prob in zip(predictions, answers, probabilities, null_probabilities):
            if prediction is None or null_prob >= prob:
                scores.append({'em': 0.0})
            else:
                scores.append({'em': compute_exact(answer, prediction)})
        return scores
