import os
import pytest
import unittest

from qaeval.generation.model import QuestionGenerationModel


@pytest.mark.skipif('GENERATION_MODEL' not in os.environ, reason='Generation model environment variable not set')
class TestGenerationModel(unittest.TestCase):
    def test_generation(self):
        model = QuestionGenerationModel(os.environ['GENERATION_MODEL'])

        # "The superjumbo Airbus A380"
        question = model.generate('The superjumbo Airbus A380 , the world \'s largest commercial airliner , took off Wednesday into cloudy skies over southwestern France for its second test flight .',
                                  0, 26)
        assert question == 'What world\'s largest commercial airliner took off Wednesday into cloudy skies over southwestern France for its second test flight?'

        # "the world 's largest commercial airliner"
        question = model.generate(
            'The superjumbo Airbus A380 , the world \'s largest commercial airliner , took off Wednesday into cloudy skies over southwestern France for its second test flight .',
            29, 69)
        assert question == 'What superjumbo Airbus A380 took off Wednesday into cloudy skies over southwestern France for its second test flight?'
