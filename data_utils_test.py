# coding: utf-8

import unittest
import data_utils

class DataTest(unittest.TestCase):
    '''Unit Test for data_utils module.
    '''
    def test_split_sentences(self):
        '''Check and print the splited sentences of data/en-tiny.txt.
        '''
        sentences = data_utils.split_sentences('data/en-tiny.txt')
        self.assertTrue(isinstance(sentences, list))
        self.assertTrue(len(sentences) > 0)
        self.assertTrue(isinstance(sentences[0], list))
        # print(sentences)

    def test_data_generator(self):
        '''Check the token_ids generated from data/en-tiny.txt.
        '''
        for token_ids in data_utils.data_generator('data/en-tiny.txt'):
            print(token_ids)

if __name__ == '__main__':
    unittest.main()
