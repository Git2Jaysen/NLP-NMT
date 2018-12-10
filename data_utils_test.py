# coding: utf-8

import unittest
import data_utils
import tensorflow as tf

class DataTest(unittest.TestCase):
    '''Unit Test for data_utils module.
    '''
    def test_generate_train_test(self):
        train_dataset, test_dataset = data_utils.generate_train_test(
            "data/en.txt",
            "data/cn.txt",
            0.2,
            32
        )
        train_iterator = train_dataset.make_initializable_iterator()
        test_iterator = test_dataset.make_initializable_iterator()
        with tf.Session() as sess:
            sess.run([train_iterator.initializer, test_iterator.initializer])
            for _ in range(1):
                print(sess.run(train_iterator.get_next()))
            print("\n=================================\n")
            for _ in range(1):
                print(sess.run(test_iterator.get_next()))

if __name__ == '__main__':
    unittest.main()
