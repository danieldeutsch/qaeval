import json
import os
import pytest
import unittest
from typing import List

from qaeval import QAEval, FIXTURES_ROOT


@pytest.mark.skipif(
    "GENERATION_MODEL" not in os.environ,
    reason="`GENERATION_MODEL` environment variable not set",
)
@pytest.mark.skipif(
    "ANSWERING_MODEL" not in os.environ,
    reason="`ANSWERING_MODEL` environment variable not set",
)
class TestQAEval(unittest.TestCase):
    def setUp(self) -> None:
        self.summaries = []
        self.references_list = []
        with open(f"{FIXTURES_ROOT}/multiling2011.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                summary = data["summary"]["text"]
                references = [reference["text"] for reference in data["references"]]
                self.summaries.append(summary)
                self.references_list.append(references)

    def _check_output(self, metric: QAEval, expected_output: List) -> None:
        actual_output = metric.score_batch(self.summaries, self.references_list)
        assert len(expected_output) == len(actual_output)
        for expected, actual in zip(expected_output, actual_output):
            assert len(expected) == len(actual) == 1
            expected = expected["qa-eval"]
            actual = actual["qa-eval"]
            assert len(expected) == len(actual)
            for metric in expected.keys():
                assert expected[metric] == pytest.approx(actual[metric], abs=1e-5)

    def test_qaeval(self):
        # This is a regression test, not necessarily a test for correctness
        metric = QAEval(
            generation_model_path=os.environ["GENERATION_MODEL"],
            answering_model_dir=os.environ["ANSWERING_MODEL"],
        )
        expected_output = [
            {
                "qa-eval": {
                    "is_answered": 0.2171952736318408,
                    "em": 0.03078358208955224,
                    "f1": 0.05688114487088367,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.2706778606965174,
                    "em": 0.08286691542288557,
                    "f1": 0.11367400349443259,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.4552238805970149,
                    "em": 0.05223880597014925,
                    "f1": 0.10360696517412935,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.2671408582089552,
                    "em": 0.04582555970149253,
                    "f1": 0.05402803689883914,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.17126063232225966,
                    "em": 0.025276841598459315,
                    "f1": 0.04173576561636263,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.3291829383548209,
                    "em": 0.029159756771697066,
                    "f1": 0.0543755246092705,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.34836235489220563,
                    "em": 0.05223880597014925,
                    "f1": 0.09381412591922542,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.4337987481945113,
                    "em": 0.04537794896485315,
                    "f1": 0.12145356515842792,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.44427039821776665,
                    "em": 0.06434837092731831,
                    "f1": 0.10272833079850623,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.40391255917571706,
                    "em": 0.09642160957950431,
                    "f1": 0.13482779720666102,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.5345864661654135,
                    "em": 0.12349624060150374,
                    "f1": 0.16393273976257167,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.5204365079365079,
                    "em": 0.12678571428571428,
                    "f1": 0.16151234567901235,
                }
            },
        ]
        self._check_output(metric, expected_output)

    @pytest.mark.skipif(
        "LERC_MODEL" not in os.environ,
        reason="`LERC_MODEL` environment variable not set",
    )
    @pytest.mark.skipif(
        "LERC_PRETRAINED_MODEL" not in os.environ,
        reason="`LERC_PRETRAINED_MODEL` environment variable not set",
    )
    def test_qaeval_with_lerc(self):
        # This is a regression test, not necessarily a test for correctness
        metric = QAEval(
            generation_model_path=os.environ["GENERATION_MODEL"],
            answering_model_dir=os.environ["ANSWERING_MODEL"],
            use_lerc=True,
            lerc_model_path=os.environ["LERC_MODEL"],
            lerc_pretrained_model_path=os.environ["LERC_PRETRAINED_MODEL"],
        )
        expected_output = [
            {
                "qa-eval": {
                    "is_answered": 0.2171952736318408,
                    "em": 0.03078358208955224,
                    "f1": 0.05688114487088367,
                    "lerc": 0.5280342313984585,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.2706778606965174,
                    "em": 0.08286691542288557,
                    "f1": 0.11367400349443259,
                    "lerc": 0.8588525844061404,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.4552238805970149,
                    "em": 0.05223880597014925,
                    "f1": 0.10360696517412935,
                    "lerc": 1.2307390170310861,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.2671408582089552,
                    "em": 0.04582555970149253,
                    "f1": 0.05402803689883914,
                    "lerc": 0.6782244059549116,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.17126063232225966,
                    "em": 0.025276841598459315,
                    "f1": 0.04173576561636263,
                    "lerc": 0.40871678001285994,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.3291829383548209,
                    "em": 0.029159756771697066,
                    "f1": 0.0543755246092705,
                    "lerc": 0.6477515654560587,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.34836235489220563,
                    "em": 0.05223880597014925,
                    "f1": 0.09381412591922542,
                    "lerc": 0.947292007320556,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.4337987481945113,
                    "em": 0.04537794896485315,
                    "f1": 0.12145356515842792,
                    "lerc": 1.2629075305115793,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.44427039821776665,
                    "em": 0.06434837092731831,
                    "f1": 0.10272833079850623,
                    "lerc": 1.1977039740821571,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.40391255917571706,
                    "em": 0.09642160957950431,
                    "f1": 0.13482779720666102,
                    "lerc": 1.2360802221434326,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.5345864661654135,
                    "em": 0.12349624060150374,
                    "f1": 0.16393273976257167,
                    "lerc": 1.5575424717221045,
                }
            },
            {
                "qa-eval": {
                    "is_answered": 0.5204365079365079,
                    "em": 0.12678571428571428,
                    "f1": 0.16151234567901235,
                    "lerc": 1.4713040575976408,
                }
            },
        ]
        self._check_output(metric, expected_output)

    @pytest.mark.skipif(
        "LERC_MODEL" not in os.environ,
        reason="`LERC_MODEL` environment variable not set",
    )
    @pytest.mark.skipif(
        "LERC_PRETRAINED_MODEL" not in os.environ,
        reason="`LERC_PRETRAINED_MODEL` environment variable not set",
    )
    def test_return_qa_pairs(self):
        metric = QAEval(
            generation_model_path=os.environ["GENERATION_MODEL"],
            answering_model_dir=os.environ["ANSWERING_MODEL"],
            use_lerc=True,
            lerc_model_path=os.environ["LERC_MODEL"],
            lerc_pretrained_model_path=os.environ["LERC_PRETRAINED_MODEL"],
        )

        summaries = [
            "Dan walked to the bakery this morning.",
            "He bought some scones today",
        ]
        references_list = [
            ["Dan went to buy scones earlier this morning."],
            ["Dan went to buy scones earlier this morning."],
        ]

        results_list = metric.score_batch(summaries, references_list, return_qa_pairs=True)
        assert len(results_list) == 2
        metrics, qa_pairs_list = results_list[0]
        assert metrics["qa-eval"]["is_answered"] == 1.0
        assert metrics["qa-eval"]["em"] == 0.5
        assert metrics["qa-eval"]["f1"] == 0.5
        self.assertAlmostEqual(metrics["qa-eval"]["lerc"], 3.171376943588257, places=4)
        assert len(qa_pairs_list) == 1
        qa_pairs = qa_pairs_list[0]
        assert len(qa_pairs) == 2
        assert (
            qa_pairs[0]["question"]["question"]
            == "Who went to buy scones earlier this morning?"
        )
        assert qa_pairs[0]["prediction"]["prediction"] == "Dan"
        assert qa_pairs[0]["prediction"]["start"] == 0
        assert qa_pairs[0]["prediction"]["end"] == 3
        assert qa_pairs[0]["prediction"]["is_answered"] == 1.0
        assert qa_pairs[0]["prediction"]["em"] == 1.0
        assert qa_pairs[0]["prediction"]["f1"] == 1.0
        self.assertAlmostEqual(
            qa_pairs[0]["prediction"]["lerc"], 5.035197734832764, places=4
        )
        assert (
            qa_pairs[1]["question"]["question"]
            == "What did Dan go to buy earlier this morning?"
        )
        assert qa_pairs[1]["prediction"]["prediction"] == "bakery"
        assert qa_pairs[1]["prediction"]["start"] == 18
        assert qa_pairs[1]["prediction"]["end"] == 24
        assert qa_pairs[1]["prediction"]["is_answered"] == 1.0
        assert qa_pairs[1]["prediction"]["em"] == 0.0
        assert qa_pairs[1]["prediction"]["f1"] == 0.0
        self.assertAlmostEqual(
            qa_pairs[1]["prediction"]["lerc"], 1.30755615234375, places=4
        )

        metrics, qa_pairs_list = results_list[1]
        assert metrics["qa-eval"]["is_answered"] == 0.5
        assert metrics["qa-eval"]["em"] == 0.5
        assert metrics["qa-eval"]["f1"] == 0.5
        self.assertAlmostEqual(metrics["qa-eval"]["lerc"], 2.492440700531006, places=4)
        assert len(qa_pairs_list) == 1
        qa_pairs = qa_pairs_list[0]
        assert len(qa_pairs) == 2
        assert (
            qa_pairs[0]["question"]["question"]
            == "Who went to buy scones earlier this morning?"
        )
        assert qa_pairs[0]["prediction"]["prediction"] == "He"
        assert qa_pairs[0]["prediction"]["start"] == 0
        assert qa_pairs[0]["prediction"]["end"] == 2
        assert qa_pairs[0]["prediction"]["is_answered"] == 0.0
        assert qa_pairs[0]["prediction"]["em"] == 0.0
        assert qa_pairs[0]["prediction"]["f1"] == 0.0
        assert qa_pairs[0]["prediction"]["lerc"] == 0.0
        assert (
            qa_pairs[1]["question"]["question"]
            == "What did Dan go to buy earlier this morning?"
        )
        assert qa_pairs[1]["prediction"]["prediction"] == "scones"
        assert qa_pairs[1]["prediction"]["start"] == 15
        assert qa_pairs[1]["prediction"]["end"] == 21
        assert qa_pairs[1]["prediction"]["is_answered"] == 1.0
        assert qa_pairs[1]["prediction"]["em"] == 1.0
        assert qa_pairs[1]["prediction"]["f1"] == 1.0
        self.assertAlmostEqual(
            qa_pairs[1]["prediction"]["lerc"], 4.984881401062012, places=4
        )
