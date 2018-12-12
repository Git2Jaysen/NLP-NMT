# coding: utf-8

import os
import json
import data_utils
import tensorflow as tf
from gensim import corpora

def input_fn(is_training, params):
    """Build input_fn, it is required by tf.estimator.Estimator when calling
    Estimator.train(input_fn) or test(input_fn).

    Args:
        params: a Dict, indicating the parameters when building input_fn.

    Returns:
        A tf.data.Dataset instance containing (features, labels), where features
        consists of (src_sequences, src_lengths) and labels consists of
        (tgt_sequences, tgt_lengths).
    """
    # load train data
    if is_training:
        assert os.path.exists(params["src_train_path"])
        assert os.path.exists(params["tgt_train_path"])
        src_sentences = json.load(open(params["src_train_path"]))
        tgt_sentences = json.load(open(params["tgt_train_path"]))
    else:
        assert os.path.exists(params["src_test_path"])
        assert os.path.exists(params["tgt_test_path"])
        src_sentences = json.load(open(params["src_test_path"]))
        tgt_sentences = json.load(open(params["tgt_test_path"]))
    assert len(src_sentences) == len(tgt_sentences)
    # load source and target dictionary
    assert os.path.exists(params["src_dict_path"])
    assert os.path.exists(params["tgt_dict_path"])
    src_dictionary = corpora.Dictionary.load(params["src_dict_path"])
    tgt_dictionary = corpora.Dictionary.load(params["tgt_dict_path"])
    # generate dataset
    src_dataset = data_utils.transform_sentences2dataset(src_sentences,
                                                         src_dictionary)
    tgt_dataset = data_utils.transform_sentences2dataset(tgt_sentences,
                                                         tgt_dictionary)
    # target eos id
    tgt_eos_id = tgt_dictionary.token2id[data_utils.tgt_eos]
    # shuffle, batch, pad zero and prefetch
    dataset = data_utils.process_dataset(
        tf.data.Dataset.zip((src_dataset, tgt_dataset)),
        params["batch_size"] if is_training else len(src_sentences),
        tgt_eos_id,
        is_training
    )
    # return dataset
    return dataset

def BiRNN_model_fn(features, labels, mode, params):
    # unzip source and target data
    src_sequences, src_lengths = features
    tgt_sequences, tgt_lengths = labels
    # define model graph

    # define model behavior
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        loss = ...
    else:
        loss = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = ...
    else:
        train_op = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = ...
    else:
        predictions = None
    # return EstimatorSpec instance
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)

# def Transformer_model_fn(features, labels, mode, params):
#     # unzip source and target data
#     src_sequences, src_lengths = features
#     tgt_sequences, tgt_lengths = labels
#     # define model graph
#
#     # define model behavior
#     if (mode == tf.estimator.ModeKeys.TRAIN or
#         mode == tf.estimator.ModeKeys.EVAL):
#         loss = ...
#     else:
#         loss = None
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         train_op = ...
#     else:
#         train_op = None
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         predictions = ...
#     else:
#         predictions = None
#     # return EstimatorSpec instance
#     return tf.estimator.EstimatorSpec(
#         mode=mode,
#         predictions=predictions,
#         loss=loss,
#         train_op=train_op)
