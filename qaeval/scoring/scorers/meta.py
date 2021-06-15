from typing import Dict, List, Set

from qaeval.scoring.scorers import Scorer


class MetaScorer(Scorer):
    def __init__(self, scorers: List['Scorer']) -> None:
        self.scorers = scorers

    def _merge_dicts(self, dicts: List[Dict[str, float]]) -> Dict[str, float]:
        merged = {}
        for other in dicts:
            merged.update(other)
        return merged

    def keys(self) -> Set[str]:
        keys = set()
        for scorer in self.scorers:
            keys |= scorer.keys()
        return keys

    def _score_single_ref(
        self,
        context: str,
        questions: List[str],
        answers: List[str],
        predictions: List[str],
        probabilities: List[float],
        null_probabilities: List[float]
    ) -> List[Dict[str, float]]:
        scores_list = []
        for scorer in self.scorers:
            _, scores = scorer.score_single_ref(
                context,
                questions,
                answers,
                predictions,
                probabilities,
                null_probabilities
            )
            scores_list.append(scores)

        combined_scores = []
        for i in range(len(questions)):
            combined_scores.append(self._merge_dicts([scores[i] for scores in scores_list]))
        return combined_scores
