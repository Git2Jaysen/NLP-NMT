# coding: utf-8

import json
import model
import unittest
import tensorflow as tf

class ModelTest(unittest.TestCase):
    def test_input_fn(self):
        params = json.load(open("data/config.json"))
        dataset = model.input_fn(False, params)
        iterator = dataset.make_initializable_iterator()
        with tf.Session() as sess:
            sess.run(iterator.initializer)
            print(sess.run(iterator.get_next()))

if __name__ == "__main__":
    unittest.main()
