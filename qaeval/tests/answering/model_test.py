import os
import pytest
import unittest

from qaeval.answering import QuestionAnsweringModel


@pytest.mark.skipif('ANSWERING_MODEL_DIR' not in os.environ, reason='Answering model environment variable not set')
class TestQuestionAnsweringModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = QuestionAnsweringModel(os.environ['ANSWERING_MODEL_DIR'], cuda_device=0)

    def test_answering(self):
        question = 'Who does the A380 super - jumbo passenger jet surpass and break their monopoly?'
        context = "The superjumbo Airbus A380 , the world 's largest commercial airliner , took off Wednesday into cloudy skies over southwestern France for its second test flight . The European aircraft maker , based in the French city of Toulouse , said the second flight -- which came exactly a week after the A380 's highly anticipated maiden voyage -- would last about four hours . As opposed to the international media hype that surrounded last week 's flight , with hundreds of journalists on site to capture the historic moment , Airbus chose to conduct Wednesday 's test more discreetly ."
        answer, probability, null_probability = self.model.answer(question, context)

        assert answer == 'the world \'s largest'
        assert probability == pytest.approx(0.00428164186632745, abs=1e-5)
        assert null_probability == pytest.approx(0.9895479613676263, abs=1e-5)

    def test_answering_with_offsets(self):
        question = 'Who does the A380 super - jumbo passenger jet surpass and break their monopoly?'
        context = "The superjumbo Airbus A380 , the world 's largest commercial airliner , took off Wednesday into cloudy skies over southwestern France for its second test flight . The European aircraft maker , based in the French city of Toulouse , said the second flight -- which came exactly a week after the A380 's highly anticipated maiden voyage -- would last about four hours . As opposed to the international media hype that surrounded last week 's flight , with hundreds of journalists on site to capture the historic moment , Airbus chose to conduct Wednesday 's test more discreetly ."
        answer, probability, null_probability, offsets = self.model.answer(question, context, return_offsets=True)

        assert answer == 'the world \'s largest'
        assert probability == pytest.approx(0.00428164186632745, abs=1e-5)
        assert null_probability == pytest.approx(0.9895479613676263, abs=1e-5)
        assert offsets == (29, 49)

    def test_answering_with_fixing_offsets(self):
        question = 'What is my name?'
        context = 'My name is Dan!'

        # Verify the original, unfixed offsets are not correct
        answer, probability, null_probability, offsets = self.model.answer(
            question, context, return_offsets=True, try_fixing_offsets=False
        )
        start, end = offsets
        assert answer == 'Dan'
        assert context[start:end] == 'Dan!'

        # `try_fixing_offsets=True` by default
        answer, probability, null_probability, offsets = self.model.answer(
            question, context, return_offsets=True
        )
        start, end = offsets
        assert answer == 'Dan'
        assert context[start:end] == 'Dan'

    def test_return_dict(self):
        question = 'Who does the A380 super - jumbo passenger jet surpass and break their monopoly?'
        context = "The superjumbo Airbus A380 , the world 's largest commercial airliner , took off Wednesday into cloudy skies over southwestern France for its second test flight . The European aircraft maker , based in the French city of Toulouse , said the second flight -- which came exactly a week after the A380 's highly anticipated maiden voyage -- would last about four hours . As opposed to the international media hype that surrounded last week 's flight , with hundreds of journalists on site to capture the historic moment , Airbus chose to conduct Wednesday 's test more discreetly ."
        result = self.model.answer(
            question, context, return_offsets=True, return_dict=True
        )

        assert result['prediction'] == 'the world \'s largest'
        assert result['probability'] == pytest.approx(0.00428164186632745, abs=1e-5)
        assert result['null_probability'] == pytest.approx(0.9895479613676263, abs=1e-5)
        assert result['start'] == 29
        assert result['end'] == 49
