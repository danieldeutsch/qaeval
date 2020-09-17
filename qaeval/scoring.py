from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

from typing import List, Tuple


def _calculate_exact_match(answer: str,
                           prediction: str,
                           probability: float,
                           null_probability: float) -> float:
    if prediction is None:
        return 0.0
    else:
        if probability > null_probability:
            return compute_exact(answer, prediction)
        return 0.0


def _calculate_f1(answer: str,
                  prediction: str,
                  probability: float,
                  null_probability: float) -> float:
    if prediction is None:
        return 0.0
    else:
        if probability > null_probability:
            return compute_f1(answer, prediction)
        return 0.0


def score(answers: List[str],
          predictions: List[str],
          probabilities: List[float],
          null_probabilities: List[float],
          return_all_scores: bool = False) -> Tuple[float, float]:
    """
    Calculates the exact-match and F1 scores for a candidate summary based on answers to questions generated from
    1 reference summary.
    """
    ems = []
    f1s = []
    for answer, prediction, probability, null_probability in zip(answers, predictions, probabilities, null_probabilities):
        ems.append(_calculate_exact_match(answer, prediction, probability, null_probability))
        f1s.append(_calculate_f1(answer, prediction, probability, null_probability))

    em = sum(ems) / len(ems)
    f1 = sum(f1s) / len(f1s)
    if return_all_scores:
        return em, f1, ems, f1s
    else:
        return em, f1


def score_multiple_references(answers_list: List[List[str]],
                              predictions_list: List[List[str]],
                              probabilities_list: List[List[float]],
                              null_probabilities_list: List[List[float]],
                              return_all_scores: bool = False) -> Tuple[float, float]:
    """
    Calculates the exact-match and F1 scores for a candidate summary based on answers to questions generated from
    multiple reference summaries.
    """
    # Reference-level scores
    ems = []
    f1s = []

    # question-level score lists
    ems_list = []
    f1s_list = []

    for answers, predictions, probabilities, null_probabilities in zip(answers_list, predictions_list, probabilities_list, null_probabilities_list):
        em, f1, ref_em, ref_f1 = score(answers, predictions, probabilities, null_probabilities, return_all_scores=True)
        ems.append(em)
        f1s.append(f1)
        ems_list.append(ref_em)
        f1s_list.append(ref_f1)

    em = sum(ems) / len(ems)
    f1 = sum(f1s) / len(f1s)
    if return_all_scores:
        return em, f1, ems_list, f1s_list
    else:
        return em, f1
