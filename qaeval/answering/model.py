# This file was edited from the run_squad.py file in the experiment repository
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadResult, SquadExample

from typing import List, Tuple, Union

from qaeval.answering.utils import compute_predictions_logits_with_null

Prediction = Union[Tuple[str, float, float], Tuple[str, float, float, Tuple[int, int]]]


class QuestionAnsweringModel(object):
    def __init__(self,
                 model_dir: str,
                 cuda_device: int = 0,
                 batch_size: int = 8,
                 silent: bool = True) -> None:
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir, config=self.config)
        if cuda_device >= 0:
            self.model.to(cuda_device)

        self.model_type = 'electra'
        self.cuda_device = cuda_device
        self.batch_size = batch_size
        self.max_seq_length = 384
        self.doc_stride = 128
        self.silent = silent

    def _to_list(self, tensor):
        return tensor.detach().cpu().tolist()

    def answer(self, question: str, context: str, return_offsets: bool = False) -> Prediction:
        return self.answer_all([(question, context)], return_offsets=return_offsets)[0]

    def answer_all(self,
                   input_data: List[Tuple[str, str]],
                   return_offsets: bool = False) -> List[Prediction]:
        # Convert all of the instances to squad examples
        examples = []
        for i, (question, context) in enumerate(input_data):
            examples.append(SquadExample(
                qas_id=str(i),
                question_text=question,
                context_text=context,
                answer_text=None,
                start_position_character=None,
                title=None,
                is_impossible=True,
                answers=[]
            ))

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            threads=1,
            tqdm_enabled=not self.silent
        )

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.batch_size)

        self.model.eval()
        all_results = []
        generator = eval_dataloader
        if not self.silent:
            generator = tqdm(generator, desc='Evaluating')

        for batch in generator:
            if self.cuda_device >= 0:
                batch = tuple(t.to(self.cuda_device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                feature_indices = batch[3]
                outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [self._to_list(output[i]) for output in outputs]
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        model_predictions = compute_predictions_logits_with_null(
            self.tokenizer,
            examples,
            features,
            all_results,
            20,
            30,
            True,
            False,
            True,
            return_offsets=return_offsets
        )

        if return_offsets:
            predictions, prediction_probs, no_answer_probs, offsets = model_predictions
        else:
            predictions, prediction_probs, no_answer_probs = model_predictions

        results = []
        for i in range(len(input_data)):
            i = str(i)
            r = (predictions[i], prediction_probs[i], no_answer_probs[i])
            if return_offsets:
                r = r + (offsets[i],)
            results.append(r)
        return results


