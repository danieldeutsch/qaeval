import os
import pytest
import unittest

from qaeval.answering import QuestionAnsweringModel


@pytest.mark.skipif('ANSWERING_MODEL_DIR' not in os.environ, reason='Answering model environment variable not set')
class TestQuestionAnsweringModel(unittest.TestCase):
    def test_answering(self):
        model = QuestionAnsweringModel(os.environ['ANSWERING_MODEL_DIR'], cuda_device=0)
        question = 'Who does the A380 super - jumbo passenger jet surpass and break their monopoly?'
        context = "The superjumbo Airbus A380 , the world 's largest commercial airliner , took off Wednesday into cloudy skies over southwestern France for its second test flight . The European aircraft maker , based in the French city of Toulouse , said the second flight -- which came exactly a week after the A380 's highly anticipated maiden voyage -- would last about four hours . As opposed to the international media hype that surrounded last week 's flight , with hundreds of journalists on site to capture the historic moment , Airbus chose to conduct Wednesday 's test more discreetly ."
        answer, probability, null_probability = model.answer(question, context)

        assert answer == 'the world \'s largest'
        assert probability == pytest.approx(0.00428164186632745, abs=1e-5)
        assert null_probability == pytest.approx(0.9895479613676263, abs=1e-5)
