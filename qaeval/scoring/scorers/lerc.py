from typing import Dict, List, Set

from allennlp.models import load_archive

from qaeval.scoring.lerc.lerc_predictor import LERCPredictor
from qaeval.scoring.scorers import Scorer


class LERCScorer(Scorer):
    def __init__(self, model_path: str, pretrained_path: str, cuda_device: int, batch_size: int = 8) -> None:
        archive = load_archive(model_path, cuda_device=cuda_device, overrides='{"model.pretrained_archive_path": "' + pretrained_path + '"}')
        self.predictor = LERCPredictor.from_archive(archive, predictor_name='lerc')
        self.batch_size = batch_size

    def keys(self) -> Set[str]:
        return {'lerc'}

    def _score_single_ref(
        self,
        context: str,
        questions: List[str],
        answers: List[str],
        predictions: List[str],
        probabilities: List[float],
        null_probabilities: List[float]
    ) -> List[Dict[str, float]]:
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
        scores = [{'lerc': s} for s in scores]
        return scores