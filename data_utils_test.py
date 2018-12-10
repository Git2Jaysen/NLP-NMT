# coding: utf-8

import unittest
import data_utils
import tensorflow as tf

class DataTest(unittest.TestCase):
    '''Unit Test for data_utils module.
    '''
    # def test_split_sentences(self):
    #     sentences = data_utils.split_sentences('data/en-tiny.txt', True)
    #     self.assertTrue(isinstance(sentences, list))
    #     self.assertTrue(len(sentences) > 0)
    #     self.assertTrue(isinstance(sentences[0], list))
    #     print(sentences)

    # def test_data_generator(self):
    #     for token_ids in data_utils.data_generator('data/en-tiny.txt'):
    #         print(token_ids)

    # def test_generate_database(self):
    #     dataset = data_utils.generate_dataset('data/en-tiny.txt')
    #     iterator = dataset.make_initializable_iterator()
    #     with tf.Session() as sess:
    #         sess.run(iterator.initializer)
    #         for _ in range(5):
    #             print(sess.run(iterator.get_next()))

    # def test_load_train_test(self):
    #     train_dataset, test_dataset = (
    #         data_utils.generate_train_test('data/en-tiny.txt',
    #                                    'data/cn-tiny.txt',
    #                                    2,
    #                                    5)
    #     )
    #     train_iterator = train_dataset.make_initializable_iterator()
    #     test_iterator = test_dataset.make_initializable_iterator()
    #     with tf.Session() as sess:
    #         sess.run([train_iterator.initializer, test_iterator.initializer])
    #         for _ in range(3):
    #             print(sess.run(train_iterator.get_next()))
    #         print('======================================')
    #         for _ in range(2):
    #             print(sess.run(test_iterator.get_next()))

    def test_batch_dataset(self):
        train_iterator, test_iterator = (
            data_utils.generate_train_test('data/en-tiny.txt',
                                           'data/cn-tiny.txt',
                                           2,
                                           1,
                                           5)
        )
        with tf.Session() as sess:
            sess.run(train_iterator.initializer)
            for _ in range(2):
                (source, src_lengths), (target, target_lengths) = sess.run(
                    train_iterator.get_next()
                )
                print(source)

if __name__ == '__main__':
    unittest.main()
