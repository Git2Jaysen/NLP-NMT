# coding: utf-8

import json
import unittest
from gensim import corpora

class CommonTest(unittest.TestCase):
    """Common test.
    """
    # def test_config(self):
    #     params = json.load(open("data/config.json"))
    #     print(params)

    def test_dict(self):
        src_dictionary = corpora.Dictionary.load("data/source.dict")
        print(len(src_dictionary))
        tgt_dictionary = corpora.Dictionary.load("data/target.dict")
        print(len(tgt_dictionary))

if __name__ == "__main__":
    unittest.main()
