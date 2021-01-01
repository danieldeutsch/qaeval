from allennlp.models.archival import load_archive
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1
from typing import Dict, List, Tuple

from qaeval.scoring.lerc.lerc_predictor import LERCPredictor


class Scorer(object):
    def score_single_ref(self,
                         context: str,
                         questions: List[str],
                         answers: List[str],
                         predictions: List[str],
                         probabilities: List[float],
                         null_probabilities: List[float]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        raise NotImplementedError

    def score_multi_ref(self,
                        context: str,
                        questions_list: List[List[str]],
                        answers_list: List[List[str]],
                        predictions_list: List[List[str]],
                        probabilities_list: List[List[float]],
                        null_probabilities_list: List[List[float]]) -> Tuple[Dict[str, float], List[List[Dict[str, float]]]]:
        raise NotImplementedError


class ExactMatchF1(Scorer):
    def _calculate_exact_match(self,
                               answer: str,
                               prediction: str,
                               probability: float,
                               null_probability: float) -> float:
        if prediction is None:
            return 0.0
        else:
            if probability > null_probability:
                return compute_exact(answer, prediction)
            return 0.0

    def _calculate_f1(self,
                      answer: str,
                      prediction: str,
                      probability: float,
                      null_probability: float) -> float:
        if prediction is None:
            return 0.0
        else:
            if probability > null_probability:
                return compute_f1(answer, prediction)
            return 0.0

    def score_single_ref(self,
                         context: str,
                         questions: List[str],
                         answers: List[str],
                         predictions: List[str],
                         probabilities: List[float],
                         null_probabilities: List[float]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        scores = []
        for answer, prediction, probability, null_probability in zip(answers, predictions, probabilities, null_probabilities):
            em = self._calculate_exact_match(answer, prediction, probability, null_probability)
            f1 = self._calculate_f1(answer, prediction, probability, null_probability)
            scores.append({'em': em, 'f1': f1})
        if len(scores) > 0:
            score = {
                'em': sum(s['em'] for s in scores) / len(scores),
                'f1': sum(s['f1'] for s in scores) / len(scores),
            }
        else:
            score = {'em': 0.0, 'f1': 0.0}
        return score, scores

    def score_multi_ref(self,
                        context: str,
                        questions_list: List[List[str]],
                        answers_list: List[List[str]],
                        predictions_list: List[List[str]],
                        probabilities_list: List[List[float]],
                        null_probabilities_list: List[List[float]]) -> Tuple[Dict[str, float], List[List[Dict[str, float]]]]:
        ref_scores, all_scores = [], []
        for questions, answers, predictions, probabilities, null_probabilities in zip(questions_list, answers_list, predictions_list, probabilities_list, null_probabilities_list):
            score, individual_scores = self.score_single_ref(context, questions, answers, predictions, probabilities, null_probabilities)
            ref_scores.append(score)
            all_scores.append(individual_scores)

        if len(ref_scores) > 0:
            score = {
                'em': sum(s['em'] for s in ref_scores) / len(ref_scores),
                'f1': sum(s['f1'] for s in ref_scores) / len(ref_scores),
            }
        else:
            score = {'em': 0.0, 'f1': 0.0}
        return score, all_scores


class LERC(Scorer):
    def __init__(self, model_path: str, pretrained_path: str, cuda_device: int, batch_size: int = 8) -> None:
        archive = load_archive(model_path, cuda_device=cuda_device, overrides='{"model.pretrained_archive_path": "' + pretrained_path + '"}')
        self.predictor = LERCPredictor.from_archive(archive, predictor_name='lerc')
        self.batch_size = batch_size

    def score_single_ref(self,
                         context: str,
                         questions: List[str],
                         answers: List[str],
                         predictions: List[str],
                         probabilities: List[float],
                         null_probabilities: List[float]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        input_dicts = []
        indices = []
        for i, (answer, question, prediction, probability, null_probability) in enumerate(zip(answers, questions, predictions,
                                                                                              probabilities, null_probabilities)):
            if probability > null_probability:
                input_dicts.append({
                    'context': context,
                    'question': question,
                    'reference': answer,
                    'candidate': prediction
                })
                indices.append(i)

        output_dicts = []
        for i in range(0, len(input_dicts), self.batch_size):
            batch = input_dicts[i:i + self.batch_size]
            output_dicts.extend(self.predictor.predict_batch_json(batch))
        assert len(output_dicts) == len(input_dicts)

        scores = [0.0] * len(questions)
        for i, output_dict in zip(indices, output_dicts):
            scores[i] = output_dict['pred_score']

        if len(scores) > 0:
            final_score = {'lerc': sum(scores) / len(scores)}
        else:
            final_score = {'lerc': 0}
        scores = [{'lerc': s} for s in scores]
        return final_score, scores

    def score_multi_ref(self,
                        context: str,
                        questions_list: List[List[str]],
                        answers_list: List[List[str]],
                        predictions_list: List[List[str]],
                        probabilities_list: List[List[float]],
                        null_probabilities_list: List[List[float]]) -> Tuple[Dict[str, float], List[List[Dict[str, float]]]]:
        ref_scores, all_scores = [], []
        for questions, answers, predictions, probabilities, null_probabilities in zip(questions_list, answers_list, predictions_list, probabilities_list, null_probabilities_list):
            score, individual_scores = self.score_single_ref(context, questions, answers, predictions, probabilities, null_probabilities)
            ref_scores.append(score)
            all_scores.append(individual_scores)

        if len(ref_scores) > 0:
            score = {'lerc': sum(s['lerc'] for s in ref_scores) / len(ref_scores)}
        else:
            score = {'lerc': 0.0}
        return score, all_scores


class MetaScorer(Scorer):
    def __init__(self, scorers: List['Scorer']) -> None:
        self.scorers = scorers

    def _merge_dicts(self, dicts: List[Dict[str, float]]) -> Dict[str, float]:
        merged = {}
        for other in dicts:
            merged.update(other)
        return merged

    def score_single_ref(self,
                         context: str,
                         questions: List[str],
                         answers: List[str],
                         predictions: List[str],
                         probabilities: List[float],
                         null_probabilities: List[float]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        instance_results = []
        individual_results = []
        for scorer in self.scorers:
            instance, individual = scorer.score_single_ref(context, questions, answers, predictions, probabilities, null_probabilities)
            instance_results.append(instance)
            individual_results.append(individual)

        instance = self._merge_dicts(instance_results)
        individual = []
        for i in range(len(questions)):
            individual.append(self._merge_dicts([results[i] for results in individual_results]))
        return instance, individual

    def score_multi_ref(self,
                        context: str,
                        questions_list: List[List[str]],
                        answers_list: List[List[str]],
                        predictions_list: List[List[str]],
                        probabilities_list: List[List[float]],
                        null_probabilities_list: List[List[float]]) -> Tuple[Dict[str, float], List[List[Dict[str, float]]]]:
        instance_results = []
        individual_results = []
        for scorer in self.scorers:
            instance, individual = scorer.score_multi_ref(context, questions_list, answers_list, predictions_list,
                                                          probabilities_list, null_probabilities_list)
            instance_results.append(instance)
            individual_results.append(individual)

        instance = self._merge_dicts(instance_results)
        individual = []
        for i in range(len(questions_list)):
            individual.append([])
            for j in range(len(questions_list[i])):
                individual[-1].append(self._merge_dicts([results[i][j] for results in individual_results]))
        return instance, individual