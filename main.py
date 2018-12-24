# coding: utf-8

import json
import hooks
import models
import data_utils

def main():
    # read config
    with open("data/config.json") as f:
        params = data_utils.preprocessing(json.load(f))
    # build RNN-Search model
    estimator = tf.estimator.Estimator(
        model_fn = models.RNN_model_fn,
        model_dir = "model",
        params = params
    )
    # train RNN-Search model
    estimator.train(
        lambda: models.input_fn(True, params),
        steps=1000
    )
    # use RNN-Search model to predict
    predictions = estimator.predict(
        lambda: models.input_fn(False, params)
    )
    # evaluate the predictions

if __name__ == "__main__":
    main()
