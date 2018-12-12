# coding: utf-8

import tensorflow as tf

class RNNModel:
    '''Neural Machine Translation Model.

    Supported Models:
        Bi-LSTM + attention
    '''
    def __init__(self,
                 src_dictionary,
                 tgt_dictionary,
                 max_training_step,
                 batch_size,
                 word_dim,
                 num_units
                 ):
        '''Create the model.

        Args:
            src_dictionary: gensim.corpora.Dictionay, source dictionary.
            tgt_dictionary: gensim.corpora.Dictionay, target dictionary.
            max_training_step: int, max training step.
            batch_size: int, batch size using for train dataset.
            word_dim: int, word dim in embedding lookup table.
        '''
        self._set_session_graph()
        self.src_dictionary = src_dictionary
        self.tgt_dictionary = tgt_dictionary
        self.max_training_step = max_training_step
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.num_units = num_units

    def _set_session_graph(self):
        '''Set 3 separate graphs and sessions for train, eval and test.
        By doing this, there are many benefits. Because train, eval and test are
        separated, weights sharing are implemented by tf.train.Saver.
        '''
        self.train_graph = tf.Graph()
        self.eval_graph = tf.Graph()
        self.test_graph = tf.Graph()
        self.train_sess = tf.Session(graph=self.train_graph)
        self.eval_sess = tf.Session(graph=self.eval_graph)
        self.test_sess = tf.Session(graph=self.test_graph)

    def _RNN_model_fn(features, labels, mode, param):
        # unzip source and target data
        src_sequences, src_lengths = features
        tgt_sequences, tgt_lengths = labels
        # define graph
        with tf.variable_scope("encoder", reuse=reuse):
            embedding_encoder = tf.get_variable(
                "embedding_encoder",
                [len(self.src_dictionary), self.word_dim]
            )
            encoder_emb_inp = tf.nn.embedding_lookup(
                embedding_encoder,
                encoder_inputs
            )
            encoder_cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_units)
            encoder_cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_units)
            (output_fw, output_bw), (state_fw, state_bw) = (
                tf.nn.bidirectional_dynamic_rnn(
                    encoder_cell_fw,
                    encoder_cell_bw,
                    encoder_emb_inp,
                    sequence_length=src_sequence_length,
                    dtype=tf.int32
                )
            )
            encoder_outputs = tf.concat([output_fw, output_bw], axis=-1)
            encoder_state = tf.concat([state_fw, state_bw], axis=-1)

        with tf.variable_scope("decoder", reuse=reuse):
            embedding_decoder = tf.get_variable(
                "embedding_decoder",
                [len(self.tgt_dictionary), self.word_dim]
            )
            decoder_emb_inp = tf.nn.embedding_lookup(
                embedding_decoder,
                decoder_inputs
            )
            decoder_cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
            train_helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp,
                tgt_sequence_length
            )
            projection_layer = tf.layers.Dense(
                len(self.tgt_dictionary),
                use_bias=False
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell,
                train_helper,
                encoder_state,
                output_layer=projection_layer
            )
            decoder_outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = decoder_outputs.rnn_output
        # define behavior
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

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

    def _build_encoder(self,
                       encoder_inputs,
                       src_sequence_length,
                       decoder_inputs,
                       tgt_sequence_length,
                       reuse=False):
        '''Build BiRNN model.
        '''
        with tf.variable_scope("encoder", reuse=reuse):
            embedding_encoder = tf.get_variable(
                "embedding_encoder",
                [len(self.src_dictionary), self.word_dim]
            )
            encoder_emb_inp = tf.nn.embedding_lookup(
                embedding_encoder,
                encoder_inputs
            )
            encoder_cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_units)
            encoder_cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_units)
            (output_fw, output_bw), (state_fw, state_bw) = (
                tf.nn.bidirectional_dynamic_rnn(
                    encoder_cell_fw,
                    encoder_cell_bw,
                    encoder_emb_inp,
                    sequence_length=src_sequence_length,
                    dtype=tf.int32
                )
            )
            encoder_outputs = tf.concat([output_fw, output_bw], axis=-1)
            encoder_state = tf.concat([state_fw, state_bw], axis=-1)

        with tf.variable_scope("decoder", reuse=reuse):
            embedding_decoder = tf.get_variable(
                "embedding_decoder",
                [len(self.tgt_dictionary), self.word_dim]
            )
            decoder_emb_inp = tf.nn.embedding_lookup(
                embedding_decoder,
                decoder_inputs
            )
            decoder_cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
            train_helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp,
                tgt_sequence_length
            )
            projection_layer = tf.layers.Dense(
                len(self.tgt_dictionary),
                use_bias=False
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell,
                train_helper,
                encoder_state,
                output_layer=projection_layer
            )
            decoder_outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = decoder_outputs.rnn_output
