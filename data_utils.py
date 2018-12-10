# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from gensim import corpora

# start of sentence
tgt_sos = "<SOS>"
# end of sentence
tgt_eos = "<EOS>"

def split_sentences(file_path, is_target=False):
    '''Split sentences in file_path.

    Args:
        file_path: string, the file's path to be splited.
        is_target: bool, whether to add "<GO>" and "<tgt_eos>" to each sentence.

    Returns:
        A 2-D list denotes the splited sentences, each sentence is a 1-D list
        contains many tokens.

    Raises:
        AssertError: if the file_path do not exist.
    '''
    assert os.path.exists(file_path), "file_path does not exists."
    if is_target:
        with open(file_path, encoding="utf-8") as file:
            sentences = [[tgt_sos] + sentence.strip().split(" ") + [tgt_eos]
                         for sentence in file]
    else:
        with open(file_path, encoding="utf-8") as file:
            sentences = [sentence.strip().split(" ") for sentence in file]
    return sentences

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
        dictionary.save("data/target.dict")
    else:
        dictionary.save("data/source.dict")
    # return dictionary
    return dictionary

def find_token_id(token, in_target=True):
    '''Find the corresponding id in dictionary according to token.

    Args:
        token: string.
        in_target: bool, whether the token in target sentences or not.

    Returns:
        the corresponding token id.
    '''
    if in_target:
        dictionary = corpora.Dictionary.load("data/target.dict")
    else:
        dictionary = corpora.Dictionary.load("data/source.dict")
    return dictionary.token2id[token]

def map_tokens2ids(sentences, dictionary):
    '''Map tokens in sentences to ids according to the dictionary.

    Args:
        sentences: a 2-D list, denoting a series of splited sentences.
        dictionary: a gensim.corpora.Dictionay instance.

    Returns:
        A 2-D list denotes the token ids corresponding to sentences.
    '''
    return [dictionary.doc2idx(sentence, len(dictionary))
            for sentence in sentences]

def data_generator(file_path, is_target=False):
    '''Build data generator from file_path.

    Args:
        file_path: string, the file's path.
        is_target: bool, denoting the dictionary generated for source or target.

    Returns:
        Yield a list of token ids of a sentence one by one.
    '''
    sentences = split_sentences(file_path, is_target)
    dictionary = generate_dictionary(sentences, is_target)
    for _, token_ids in enumerate(map_tokens2ids(sentences, dictionary)):
        yield token_ids

def generate_dataset(file_path, is_target=False):
    '''Build a tf.data.Dataset instance from a data generator,
       and add sentence length for each sentences.

       Args:
           file_path: string, the path of the file to be loaded.
           is_target: bool, denoting the dictionary generated for source or target.

       Returns:
           A tf.data.Dataset instance denoting the loaded data.
    '''
    dataset = (
        tf.data.Dataset.from_generator(lambda: data_generator(file_path,
                                                              is_target),
                                       tf.int32)
    )
    dataset = dataset.map(lambda s: (s, tf.size(s)))
    return dataset

def split_dataset(dataset, n_parts, data_range):
    '''Split dataset.

    Args:
        dataset: tf.data.Dataset, the dataset to be splited.
        n_parts: int, the number of parts to be splited.
        data_range: list or range, indicating which should be selected.

    Returns:
        A tf.data.Dataset instance.

    Raises:
        AssertError: if data_range's length is 0.
    '''
    assert len(data_range) > 0, "data_range is None or empty."
    new_dataset = dataset.shard(n_parts, data_range[0])
    for i in data_range[1:]:
        new_dataset = new_dataset.concatenate(dataset.shard(n_parts, i))
    return new_dataset

def batch_dataset(dataset, batch_size):
    '''Process dataset, note that it's for train dataset only.

    # pipeline
    1. shuffle the dataset.
    2. batch dataset with batch size and pad zero.
    3. repeat the dataset forever(useful for training).
    4. prefetch data(promote perfermance).

    Args:
        dataset: tf.data.Dataset, the dataset to be batched.
        batch_size: int, the batch size using for batching data.

    Returns:
        the processed dataset.
    '''
    tgt_eos_id = find_token_id(tgt_eos, True)
    dataset = (
        dataset.shuffle(10)
               .padded_batch(
                    batch_size,
                    padded_shapes=(([None], []),
                                   ([None], [])),
                    padding_values=((0, 0),          # unused second
                                    (tgt_eos_id, 0)) # unused second
               )
               .repeat()
               .prefetch(batch_size)
    )
    return dataset

def generate_train_test(src_file_path,
                        tgt_file_path,
                        batch_size,
                        n_test=2,
                        n_parts=10):
    '''Generate datasets from src_file_path and tgt_file_path, and split them
       into train and test.

       # pipeline
       1. build two tf.data.Dataset instances.
       2. zip the two datasets into one.
       3. split the dataset into two parts: train and test.
       4. return their iterators.

    Args:
        src_file_path: string, source file's path.
        tgt_file_path: string, target file's path.
        batch_size: int, denoting the batch size using for train dataset.
        n_test: int, the number of parts use for test dataset.
        n_parts: int, total parts of the splited datset.

    Returns:
        Iterators of train and test datasets.
    '''
    src_dataset = generate_dataset(src_file_path)
    tgt_dataset = generate_dataset(tgt_file_path, True)
    all_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    # construct train dataset
    train_dataset = batch_dataset(
        split_dataset(all_dataset,
                      n_parts,
                      range(0, n_parts - n_test)),
        batch_size
    )
    # construct test dataset
    test_dataset = split_dataset(all_dataset,
                                 n_parts,
                                 range(n_parts - n_test, n_parts))
    return (
        train_dataset.make_initializable_iterator(),
        test_dataset.make_initializable_iterator()
    )
