# coding: utf-8

import json
import hooks
import models
import evaluate
import data_utils

def refresh_params(params):
    """Refresh params needed by model_fn.

    Args:
        params: a Dict, config params.
    """
    params["batch_size"] = params["n_test_samples"]

def main():
    # preprocessing
    data_utils.preprocessing()
    # read config
    with open("data/config.json") as f:
        params = data_utils.preprocessing(json.load(f))
    # build RNN-Search model
    train_estimator = tf.estimator.Estimator(
        model_fn = models.RNN_model_fn,
        model_dir = "model",
        params = params
    )
    # define TrainSpec
    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: models.input_fn(True, params),
        max_steps = 1000
    )
    # define EarlyStoppingHook
    early_stopping_hook = EarlyStoppingHook()
    # define EvalSpec
    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: models.input_fn(False, params),
        hooks = early_stopping_hook
    )
    # train and evaluate RNN-Search model
    tf.estimator.train_and_evaluate(
        train_estimator, train_spec, eval_spec
    )
    # refresh params and rebuild model
    refresh_params(params)
    test_estimator = tf.estimator.Estimator(
        model_fn = models.RNN_model_fn,
        model_dir = "model",
        params = params,
        warm_start_from = "model")
    # use RNN-Search model to predict
    predictions = test_estimator.predict(
        lambda: models.input_fn(False, params)
    )
    # evaluate BELU score of the predictions
    with open(params["tgt_test_sentences"]) as f:
        references = json.load(f)
    print(evaluate.BELU(references, predictions))

if __name__ == "__main__":
    main()
