from transformers.data.metrics.squad_metrics import compute_f1
from typing import Dict, List, Set

from qaeval.scoring.scorers import Scorer


class F1Scorer(Scorer):
    def keys(self) -> Set[str]:
        return {'f1'}

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
                scores.append({'f1': 0.0})
            else:
                scores.append({'f1': compute_f1(answer, prediction)})
        return scores
