# coding: utf-8

import evaluate
import unittest

class EvaluateTest(unittest.TestCase):
    # def test_get_ngram(self):
    #     sentences = [
    #         [1, 2, 3, 4, 5],
    #         [6, 7, 8, 9]
    #     ]
    #     print(evaluate._get_ngrams(sentences, 4))

    def test_BELU(self):
        refer = [['the', 'cat', 'is', 'on', 'the', 'mat', '.']]
        trans = [['the', 'the', 'the', 'the', 'the', 'the']]
        print(evaluate.BELU(refer, trans, 4))

if __name__ == "__main__":
    unittest.main()
