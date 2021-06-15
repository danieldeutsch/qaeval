import unittest

from qaeval.answering.utils import fix_answer_span, SpanFixError


class TestUtils(unittest.TestCase):
    def test_fix_answer_span(self):
        assert fix_answer_span('Dan', 'Dan!', 0, 4) == (0, 3)
        assert fix_answer_span('Dan', 'Dan!', 10, 14) == (10, 13)
        assert fix_answer_span('Dan', ' Dan!', 0, 5) == (1, 4)
        assert fix_answer_span('Dan', ' Dan! ', 0, 6) == (1, 4)
        assert fix_answer_span('Dan', '  Dan!  ', 0, 8) == (2, 5)

        assert fix_answer_span('is Dan', 'is Dan!', 0, 7) == (0, 6)
        assert fix_answer_span('is Dan', ' is Dan!', 0, 8) == (1, 7)
        assert fix_answer_span('is Dan', ' is Dan! ', 0, 9) == (1, 7)
        assert fix_answer_span('is Dan', 'is  Dan!', 0, 8) == (0, 7)
        assert fix_answer_span('is Dan', 'is   Dan!', 0, 9) == (0, 8)
        assert fix_answer_span('is Dan', ' is   Dan! ', 0, 11) == (1, 9)

        # Length is too long
        with self.assertRaises(SpanFixError):
            fix_answer_span('Dan!', 'Dan', 0, 3)
        with self.assertRaises(SpanFixError):
            fix_answer_span('is  Dan', 'is Dan', 0, 6)

        # Not a substring
        with self.assertRaises(SpanFixError):
            fix_answer_span('Dan', 'Not a substring', 0, 15)

    def test_fix_answer_span_unicode(self):
        prediction = 'track and   field, swimming, diving'
        document_span = 'that…track and field, swimming,  diving,'
        assert fix_answer_span(prediction, document_span, 0, 40) == (5, 39)
