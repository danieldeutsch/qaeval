import collections
import edlib
from typing import Tuple

from transformers.data.metrics.squad_metrics import (
    get_final_text,
    _get_best_indexes,
    _compute_softmax,
)


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def _get_char_offsets(example, pred_start, pred_end):
    # The returned end index will be exclusive
    if pred_start is None or pred_end is None:
        # This could happen if there's an edge case with no valid predictions. See where the prediction is "empty"
        return None, None

    token_to_char_start = {}
    token_to_char_end = {}
    for char_index, token_index in enumerate(example.char_to_word_offset):
        if token_index not in token_to_char_start:
            token_to_char_start[token_index] = char_index
        token_to_char_end[token_index] = char_index

    # Any whitespace after the token is included in that token. Find the last non-whitespace character
    for token_index, end in token_to_char_end.items():
        if token_index == -1:
            # Whitespace at the beginning is mapped to token -1. We don't care about it
            continue
        while _is_whitespace(example.context_text[end]):
            end -= 1
            if end < 0:
                break
        if end < 0:
            raise Exception(f'Token end is less than 0.')
        token_to_char_end[token_index] = end + 1  # exclusive
    return token_to_char_start[pred_start], token_to_char_end[pred_end]


def compute_predictions_logits_with_null(
        tokenizer,
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        verbose_logging,
        version_2_with_negative,
        return_offsets = False
):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    all_probs = collections.OrderedDict()
    null_scores = collections.OrderedDict()
    offsets = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "doc_start", "doc_end"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                orig_doc_start = None
                orig_doc_end = None
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit,
                                          doc_start=orig_doc_start, doc_end=orig_doc_end))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit,
                                              doc_start=None, doc_end=None))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, doc_start=None,
                                                 doc_end=None))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, doc_start=None,
                                          doc_end=None))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        best_non_null_entry_index = None
        for i, entry in enumerate(nbest):
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
                    best_non_null_entry_index = i

        probs = _compute_softmax(total_scores)

        nbest_json = []
        null_prob = None
        best_prob = None
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            if entry.text == '':
                null_prob = probs[i]
            if i == best_non_null_entry_index:
                best_prob = probs[i]
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # Always predict the best non-null text
            all_predictions[example.qas_id] = best_non_null_entry.text
            all_probs[example.qas_id] = best_prob
            null_scores[example.qas_id] = null_prob
            offsets[example.qas_id] = _get_char_offsets(example, best_non_null_entry.doc_start,
                                                        best_non_null_entry.doc_end)

            # # predict "" iff the null score - the score of best non-null > threshold
            # score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            # scores_diff_json[example.qas_id] = score_diff
            # if score_diff > null_score_diff_threshold:
            #     all_predictions[example.qas_id] = ""
            # else:
            #     all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    output = (all_predictions, all_probs, null_scores)
    if return_offsets:
        output = output + (offsets,)
    return output


class SpanFixError(Exception):
    pass


def fix_answer_span(prediction: str, document_span: str, start: int, end: int) -> Tuple[int, int]:
    """
    Tries to fix the answer span of the prediction, which may include some extra whitespace or special characters.

    # Parameters
    - `prediction`: the string output by the QA model
    - `document_span`: the span in the text given by the maybe noisy offsets. See `QuestionAnsweringModel.answer()`
    documentation for more information
    - `start`: the start character offset of `document_span` in the original text
    - `end`: the *exclusive* end character offset of the `document_span` in the original text

    # Returns
    The `start` and *exclusive* `end` character offsets of fixed character offsets of `prediction` in the
    original text.
    """

    if len(prediction) > len(document_span):
        raise SpanFixError(f'Unexpected lengths: "{prediction}", "{document_span}"')

    alignment = edlib.align(prediction, document_span, mode='HW', task='path')
    locations = alignment['locations']
    if len(locations) != 1:
        raise SpanFixError(f'Unable to compute span: "{prediction}", "{document_span}"')
    align_start, align_end = locations[0]

    start += align_start
    end -= len(document_span) - align_end
    end += 1
    return start, end