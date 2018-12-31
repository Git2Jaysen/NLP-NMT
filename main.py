# coding: utf-8

import json
import hooks
import evals
import models
import logging
import data_utils
import tensorflow as tf

# config logging prints
logging.basicConfig(format='%(asctime)s - <%(levelname)s> - %(message)s',
                    level=logging.INFO)

def refresh_params(params):
    """Refresh params needed by model_fn.

    Args:
        params: a Dict, config params.
    """
    params["batch_size"] = params["n_test_samples"]

def main():
    # preprocessing
    logging.info("preprocessing data.")
    data_utils.preprocessing()
    # read config
    logging.info("loading config.")
    with open("data/config.json") as f:
        params = json.load(f)
    # define run config
    run_config = tf.estimator.RunConfig(
        model_dir = "model",
        save_summary_steps = 1,
        save_checkpoints_steps = 1
    )
    # build RNN-Search model
    logging.info("building train estimator.")
    train_estimator = tf.estimator.Estimator(
        model_fn = models.MultiRNN_model_fn,
        model_dir = "model",
        params = params,
        config = run_config
    )
    # define TrainSpec
    logging.info("defining train spec.")
    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: models.input_fn(True, params),
        max_steps = 1000
    )
    # define EarlyStoppingHook
    logging.info("defining early stopping hook")
    early_stopping_hook = hooks.EarlyStoppingHook()
    # define EvalSpec
    logging.info("defining eval spec.")
    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: models.input_fn(False, params),
        hooks = [early_stopping_hook],
        throttle_secs = 1
    )
    # train and evaluate RNN-Search model
    logging.info("training and evaluating.")
    try:
        tf.estimator.train_and_evaluate(
            train_estimator, train_spec, eval_spec
        )
    except ValueError as e:
        logging.info("training stopped.")
    # refresh params and rebuild model
    logging.info("refreshing params.")
    refresh_params(params)
    logging.info("rebuilding test estimator.")
    test_estimator = tf.estimator.Estimator(
        model_fn = models.MultiRNN_model_fn,
        model_dir = "model",
        params = params,
        warm_start_from = "model")
    # use RNN-Search model to predict
    logging.info("predicting.")
    predictions = test_estimator.predict(
        lambda: models.input_fn(False, params)
    )
    # evaluate BELU score of the predictions
    logging.info("evaluating BELU score of predictions.")
    with open(params["tgt_test_path"]) as f:
        references = json.load(f)
    with open(params["eval_belu_path"], "w") as f:
        f.write("BELU Score: {}".format(evals.BELU(references, predictions)))

if __name__ == "__main__":
    main()
