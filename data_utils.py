# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from gensim import corpora

# start of sentence
sos = "<SOS>"
# end of sentence
eos = "<EOS>"
# dictionary id of sos, need to be modified when generateing dictionary
sos_id = -1
# dictionary id of eos, need to be modified when generateing dictionary
eos_id = -1
# source dictionary size, need to be modified when generateing dictionary
src_dict_size = -1
# target dictionary size, need to be modified when generateing dictionary
tgt_dict_size = -1

def split_sentences(file_path, is_target=False):
    '''Split sentences in file_path.

    Args:
        file_path: string, the file's path to be splited.
        is_target: bool, whether to add "<GO>" and "<EOS>" to each sentence.

    Returns:
        A 2-D list denotes the splited sentences, each sentence is a 1-D list
        contains many tokens.

    Raises:
        AssertError: if the file_path do not exist.
    '''
    assert os.path.exists(file_path), "file_path does not exists."
    if is_target:
        sentences = [[sos] + sentence.strip().split(" ") + [eos]
                     for sentence in open(file_path, encoding="utf-8")]
    else:
        sentences = [sentence.strip().split(" ")
                     for sentence in open(file_path, encoding="utf-8")]

def generate_dictionary(sentences, is_target=False):
    '''Generate dictionary for the giving sentences.

    Args:
        sentences: a 2-D list, denoting a series of splited sentences.
        is_target: bool, denoting the dictionary generated for source or target.

    Returns:
        A gensim.corpora.Dictionay instance.

    Raises:
        AssertError: if sentences is None.
    '''
    assert sentences is not None, "sentences to build ditcionary is None."
    dictionary = corpora.Dictionary(sentences)
    # set global config
    if is_target:
        global sos_id, eos_id, tgt_dict_size
        sos_id = dictionary.token2id[sos]
        eos_id = dictionary.token2id[eos]
        tgt_dict_size = len(dictionary.items())
    else:
        global src_dict_size
        src_dict_size = len(dictionary.items())
    # return dictionary
    return dictionary

def map_tokens2ids(sentences, dictionary):
    '''Map tokens in sentences to ids according to the dictionary.

    Args:
        sentences: a 2-D list, denoting a series of splited sentences.
        dictionary: a gensim.corpora.Dictionay instance.

    Returns:
        A 2-D list denotes the token ids corresponding to sentences.
    '''
    return [dictionary.doc2idx(sentence, len(dictionary)) for sentence in sentences]

def data_generator(file_path):
    '''Build data generator from file_path.

    Args:
        file_path: string, the file's path.

    Returns:
        Yield a list of token ids of a sentence one by one.
    '''
    sentences = split_sentences(file_path)
    dictionary = generate_dictionary(sentences)
    for _, token_ids in enumerate(map_tokens2ids(sentences, dictionary)):
        yield token_ids

def generate_dataset(file_path, batch_size, for_training=True):
    '''Generate a tf.data.Dataset instance from file_path.

    Args:
        file_path: string, the file's path.
        batch_size: int, the batch size using for "batching" data.
        for_training: bool, whether the dataset to generate is using for training.
                      if True, then the dataset should be repeated.

    Returns:
        A tf.data.Dataset instance shuffled, batched and zero-padded.
    '''
    # build a tf.data.Dataset instance from a data generator.
    dataset = tf.data.Dataset.from_generator(lambda: data_generator(file_path))
    # add sentence length for each sentence
    dataset = dataset.map(lambda s: (s, tf.size(s)))
    # shuffle, batch, zero-pad and prefetch
    dataset = (
        dataset.shuffle(50)
               .padded_batch(
                    batch_size,
                    padded_shapes=([None], []),
                    padding_values=(0, 0) # the second value is unused
               )
               .prefetch(batch_size)
    )
    # if using for training, then repeat the dataset forever
    if for_training:
        dataset = dataset.repeat()
    # return the generated dataset
    return dataset
