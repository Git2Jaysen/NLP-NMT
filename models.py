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
        with open(params["src_train_path"]) as f:
            src_sentences = json.load(f)
        with open(params["tgt_train_path"]) as f:
            tgt_sentences = json.load(f)
    else:
        assert os.path.exists(params["src_test_path"])
        assert os.path.exists(params["tgt_test_path"])
        with open(params["src_test_path"]) as f:
            src_sentences = json.load(f)
        with open(params["tgt_test_path"]) as f:
            tgt_sentences = json.load(f)
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
    # shuffle, batch and pad zero
    # test dataset will NOT be bachted(batch_size = n_test_samples)
    dataset = data_utils.process_dataset(
        tf.data.Dataset.zip((src_dataset, tgt_dataset)),
        params["batch_size"],
        params["tgt_eos_id"],
        is_training
    )
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
    if mode == tf.estimator.ModeKeys.TRAIN:
        tgt_sequences, tgt_lengths = labels
    # define model graph
    # encoder part
    embedding_encoder = tf.Variable(
        tf.truncated_normal([params["src_word_size"],
                             params["embedding_size"]]),
        name = "embedding_encoder"
    )
    # shape: [batch_size, max_sequence_length, embedding_size]
    encoder_emb_inp = tf.nn.embedding_lookup(
        embedding_encoder,
        src_sequences
    )
    encoder_cell_fw = tf.nn.rnn_cell.LSTMCell(params["rnn_units"])
    encoder_cell_bw = tf.nn.rnn_cell.LSTMCell(params["rnn_units"])
    encoder_outputs, encoder_state = (
        tf.nn.bidirectional_dynamic_rnn(
            encoder_cell_fw,
            encoder_cell_bw,
            encoder_emb_inp,
            sequence_length = src_lengths,
            dtype = tf.float32
        )
    )
    # concat forward and backward outputs, using for attention
    # shape: [batch_size, max_sequence_length, 2 * rnn_units]
    encoder_outputs = tf.concat(encoder_outputs, axis=-1)
    # decoder part
    embedding_decoder = tf.Variable(
        tf.truncated_normal([params["tgt_word_size"],
                             params["embedding_size"]]),
        name = "embedding_decoder"
    )
    # shape: [batch_size, max_sequence_length, embedding_size]
    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_emb_inp = tf.nn.embedding_lookup(
            embedding_decoder,
            tgt_sequences
        )
    # attention
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        params["rnn_units"],
        encoder_outputs,
        memory_sequence_length = src_lengths
    )
    # using for initializing decoder
    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(params["rnn_units"]) for _ in range(2)]
    )
    # wrap decoder cell with attention
    attended_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell,
        attention_mechanism,
        attention_layer_size = params["rnn_units"]
    )
    # wrapper encoder_state to an AttentionWrapperState instance
    decoder_initial_state = attended_decoder_cell.zero_state(
        params["batch_size"], tf.float32
    )
    decoder_initial_state = decoder_initial_state.clone(
        cell_state = encoder_state
    )
    # map the output dim to tgt_word_size(using for softmax)
    projection_layer = tf.layers.Dense(
        params["tgt_word_size"],
        use_bias = False
    )
    # different helper when training and testing
    if mode == tf.estimator.ModeKeys.TRAIN:
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp,
            tgt_lengths
        )
    else:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_decoder,
            tf.fill([params["batch_size"]], params["tgt_sos_id"]),
            params["tgt_eos_id"]
        )
    # build basic decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        attended_decoder_cell,
        helper,
        decoder_initial_state,
        output_layer = projection_layer
    )
    # different dynamic_decode paramters when training and testing
    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    else:
        # define maximum decoding length
        maximum_iterations = tf.round(tf.reduce_max(src_lengths) * 2)
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations = maximum_iterations
        )
    # define loss
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        logits = decoder_outputs.rnn_output
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tgt_sequences, logits=logits
        )
        target_weights = tf.sequence_mask(
            tgt_lengths,
            tf.reduce_max(tgt_lengths),
            dtype = tf.float32,
            name = "mask"
        )
        loss = tf.reduce_sum(
            crossent * target_weights / params["batch_size"],
            name = "loss"
        )
    else:
        loss = None
    # define training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        variables = tf.trainable_variables()
        gradients = tf.gradients(loss, variables)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients,
            params["max_global_norm"]
        )
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.apply_gradients(
            zip(clipped_gradients, variables),
            global_step = tf.train.get_global_step()
        )
    else:
        train_op = None
    # define predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = decoder_outputs.sample_id
    else:
        predictions = None
    # return EstimatorSpec instance
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = predictions,
        loss = loss,
        train_op = train_op
    )
