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
    # shuffle, batch, pad zero and prefetch,
    # and test dataset will NOT be bachted
    dataset = data_utils.process_dataset(
        tf.data.Dataset.zip((src_dataset, tgt_dataset)),
        params["batch_size"] if is_training else len(src_sentences),
        tgt_eos_id,
        is_training)
    # return dataset
    return dataset

def RNN_model_fn(features, labels, mode, params):
    """The model is defined in (Bahdanau et al., 2015)， which was named as
    RNN-Search. This implementation is referring to "Neural Machine Translation
    (seq2seq) Tutorial" on Github, with a Bi-LSTM encoder and a LSTM decoder,
    using attention mechanism.

    Args:
        features: a tuple, (source sequences, source sequence lengths)
        labels: a tuple, (target sequences, target sequence lengths)
        mode: a tf.estimator.ModeKeys instance, denoting TRAIN, EVAL
               and PREDICT.
        params: a Dict, using for building computational graph.

    Returns:
        A tf.estimator.EstimatorSpec instance.
    """
    # unzip source and target data
    # src_sequences shape: [batch_size, max_sequence_length]
    # src_lengths shape: [batch_size]
    src_sequences, src_lengths = features
    # tgt_sequences shape: [batch_size, max_sequence_length]
    # tgt_lengths shape: [batch_size]
    tgt_sequences, tgt_lengths = labels
    # define model graph
    # encoder part
    embedding_encoder = tf.Variable(
        tf.truncated_normal([params["src_word_size"],
                             params["embedding_size"]]),
        name = "embedding_encoder")
    # shape: [batch_size, max_sequence_length, embedding_size]
    encoder_emb_inp = tf.nn.embedding_lookup(
        embedding_encoder,
        src_sequences)
    encoder_cell = tf.nn.rnn_cell.LSTMCell(params["rnn_units"])
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        cell = encoder_cell,
        inputs = encoder_emb_inp,
        sequence_length = src_lengths,
        dtype = tf.float32)
    # encoder_cell_fw = tf.nn.rnn_cell.LSTMCell(params["rnn_units"])
    # encoder_cell_bw = tf.nn.rnn_cell.LSTMCell(params["rnn_units"])
    # (output_fw, output_bw), (state_fw, state_bw) = (
    #     tf.nn.bidirectional_dynamic_rnn(
    #         encoder_cell_fw,
    #         encoder_cell_bw,
    #         encoder_emb_inp,
    #         sequence_length=src_lengths,
    #         dtype=tf.float32))
    # concat forward and backward outputs, using for attention
    # shape: [batch_size, max_sequence_length, 2 * rnn_units]
    # encoder_outputs = tf.concat([output_fw, output_bw], axis=-1)
    # concat the final state of forward and backward, using for decoder
    # shape: [batch_size, 2 * rnn_units]
    # encoder_state = tf.reshape(
    #     tf.concat([state_fw, state_bw], axis=-1),
    #     [params["batch_size"], 2 * params["rnn_units"]])
    # decoder part
    embedding_decoder = tf.Variable(
        tf.truncated_normal([params["tgt_word_size"],
                             params["embedding_size"]]),
        name = "embedding_decoder")
    # shape: [batch_size, max_sequence_length, embedding_size]
    decoder_emb_inp = tf.nn.embedding_lookup(
        embedding_decoder,
        tgt_sequences)
    # attention
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        params["rnn_units"],
        encoder_outputs,
        memory_sequence_length=src_lengths)
    # using for initializing decoder
    decoder_cell = tf.nn.rnn_cell.LSTMCell(params["rnn_units"])
    # wrap decoder cell with attention
    attended_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell,
        attention_mechanism,
        attention_layer_size = params["rnn_units"])
    # wrapper encoder_state to an AttentionWrapperState instance
    decoder_initial_state = attended_decoder_cell.zero_state(
        params["batch_size"], tf.float32)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=encoder_state)
    # map the output dim to tgt_word_size(using for softmax)
    projection_layer = tf.layers.Dense(
        params["tgt_word_size"],
        use_bias = False)
    # different helper when training and testing
    if mode == tf.estimator.ModeKeys.TRAIN:
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp,
            tgt_lengths)
    else:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            decoder_emb_inp,
            tgt_lengths)
    # build basic decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        attended_decoder_cell,
        helper,
        decoder_initial_state,
        output_layer=projection_layer)
    # different dynamic_decode paramters when training and testing
    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    else:
        # define maximum decoding length
        maximum_iterations = tf.round(tf.reduce_max(src_lengths) * 2)
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=maximum_iterations)
    # define loss
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        logits = decoder_outputs.rnn_output
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_outputs, logits=logits)
        loss = tf.reduce_sum(
            crossent * target_weights / params["batch_size"],
            name = "loss")
    else:
        loss = None
    # define training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        variables = tf.trainable_variables()
        gradients = tf.gradients(loss, variables)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients,
            params["max_global_norm"])
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.apply_gradients(zip(clipped_gradients,
                                                 variables))
    else:
        train_op = None
    # define predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = decoder_outputs.sample_ids
    else:
        predictions = None
    # return EstimatorSpec instance
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = predictions,
        loss = loss,
        train_op = train_op)
