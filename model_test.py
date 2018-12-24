# coding: utf-8

import json
import models
import unittest
import tensorflow as tf

class ModelTest(unittest.TestCase):
    # def test_input_fn(self):
    #     params = json.load(open("data/config.json"))
    #     train_dataset = models.input_fn(True, params)
    #     test_dataset = models.input_fn(False, params)
    #     train_iterator = train_dataset.make_initializable_iterator()
    #     test_iterator = test_dataset.make_initializable_iterator()
    #     with tf.Session() as sess:
    #         sess.run(train_iterator.initializer)
    #         (features, labels) = sess.run(train_iterator.get_next())
    #         src_sequences, src_lengths = features
    #         tgt_sequences, tgt_lengths = labels
    #         print(src_sequences)
    #         print(src_lengths)
    #         print(tgt_sequences)
    #         print(tgt_lengths)
    #     with tf.Session() as sess:
    #         sess.run(test_iterator.initializer)
    #         (features, labels) = sess.run(test_iterator.get_next())
    #         src_sequences, src_lengths = features
    #         tgt_sequences, tgt_lengths = labels
    #         print(src_sequences)
    #         print(src_lengths)
    #         print(tgt_sequences)
    #         print(tgt_lengths)

    def test_model_fn(self):
        params = json.load(open("data/config.json"))
        estimator = tf.estimator.Estimator(
            model_fn = models.RNN_model_fn,
            model_dir = "model",
            params = params)
        estimator.train(
            lambda: models.input_fn(True, params),
            steps=1
        )
        predictions = estimator.predict(
            lambda: models.input_fn(False, params)
        )
        for pred in predictions:
            print(pred)

if __name__ == "__main__":
    unittest.main()
