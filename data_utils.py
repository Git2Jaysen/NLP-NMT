# coding: utf-8

import os
import json
import numpy as np
import tensorflow as tf
from gensim import corpora
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# start of sentence
tgt_sos = "<SOS>"
# end of sentence
tgt_eos = "<EOS>"

def split_sentences(file_path, is_target=False):
    """Split sentences in file_path.

    Args:
        file_path: string, the file's path to be splited.
        is_target: bool, whether to add "<GO>" and "<tgt_eos>" to each sentence.

    Returns:
        A 2-D list denotes the splited sentences, each sentence is a 1-D list
        contains many tokens.

    Raises:
        AssertError: if the file_path do not exist.
    """
    assert os.path.exists(file_path), "file_path does not exists."
    if is_target:
        with open(file_path, encoding="utf-8") as file:
            sentences = [[tgt_sos] + sentence.strip().split(" ") + [tgt_eos]
                         for sentence in file]
    else:
        with open(file_path, encoding="utf-8") as file:
            sentences = [sentence.strip().split(" ") for sentence in file]
    return sentences

def split_train_test(src_file_path, tgt_file_path, test_size):
    """Split train and test data according to test_rate.

    Args:
        src_file_path: string, source file's path.
        tgt_file_path: string, target file's path.
        test_size: float, the test size.

    Returns:
        source sentences, target sentences and their train and test sentences.
    """
    src_sentences, tgt_sentences = (
        split_sentences(src_file_path),
        split_sentences(tgt_file_path, True)
    )
    return (
        src_sentences, tgt_sentences,
        train_test_split(src_sentences, tgt_sentences, test_size=test_size)
    )

def map_tokens2ids(sentences, dictionary):
    """Map tokens in sentences to ids according to the dictionary.

    Args:
        sentences: a 2-D list, denoting a series of splited sentences.
        dictionary: a gensim.corpora.Dictionay instance.

    Returns:
        A 2-D list denotes the token ids corresponding to sentences.
    """
    return [dictionary.doc2idx(sentence, len(dictionary))
            for sentence in sentences]

def data_generator(sequences):
    """Build data generator from sequences. Using for tf.data.Dataset.
    from_generator. Can not use tf.data.Dataset.from_tensors or
    tf.data.Dataset.from_tensor_slices, because the setence length is vaiable.
    If want to use the two above, try 0-padding first.

    Args:
        sentences: 2-D list.

    Returns:
        yield a sentence one by one.
    """
    for sequence in sequences:
        yield sequence

def generate_dataset(sequences):
    """Build a tf.data.Dataset instance from a data generator,
       and add sentence length for each sentences.

       Args:
           sequences: 2-D list.

       Returns:
           A tf.data.Dataset instance denoting the loaded data.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(sequences),
        tf.int32
    )
    return dataset.map(lambda s: (s, tf.size(s)))

def transform_sentences2dataset(sentences, dictionary):
    """Transform sentences to dataset according to dictionary.

    Args:
        sentences: 2-D list.
        dictionary: corpora.Dictionay, the corresponding dictionary.

    Returns:
        A tf.data.Dataset instance.
    """
    sequences = map_tokens2ids(sentences, dictionary)
    dataset = generate_dataset(sequences)
    return dataset

def process_dataset(dataset, batch_size, tgt_eos_id, is_training):
    """Process dataset with shuffling, zero-padding, prefetching and repeating.

    # pipeline
    1. shuffle the dataset.
    2. batch dataset with batch size and pad zero.
    3. repeat the dataset forever(useful for training).
    4. prefetch data(promote perfermance).

    Args:
        dataset: tf.data.Dataset, the dataset to be batched.
        batch_size: int, the batch size using for batching data.
        tgt_eos_id: int, index of target eos in dictionary
        is_training: bool, denoting training or not. Using for batch dataset,
                     for testing, dataset consits of one batch, whose size is
                     the number of test samples.

    Returns:
        the processed dataset.
    """
    dataset = (
        dataset.shuffle(10)
               .padded_batch(
                    batch_size,
                    padded_shapes=(([None], []),
                                   ([None], [])),
                    padding_values=((0, 0),          # unused second
                                    (tgt_eos_id, 0)) # unused second
               )
    )
    return dataset.repeat() if is_training else dataset

def preprocessing(config_path = "data/config.json"):
    """Preprocess the given data in params.

    Pipeline:
        1. split sentencs from file and generate train and test sentences.
        2. generate source and target dictionaries.
        3. save results to data dir.
        4. set config info and refresh config.

    Args:
        config_path: a string, config file path.
    """
    assert os.path.exists(config_path), "config file not found."
    with open(config_path) as f:
        params = json.load(f)
    assert os.path.exists(params["src_path"]), "source file not found."
    assert os.path.exists(params["tgt_path"]), "target file not found."
    # pipeline 1
    (
        src_sentences, tgt_sentences,
        (
            src_train_sentences, src_test_sentences,
            tgt_train_sentences, tgt_test_sentences
        )
    ) = split_train_test(
        params["src_path"], params["tgt_path"], params["test_size"]
    )
    # pipeline 2
    src_dictionary = corpora.Dictionary(src_sentences)
    tgt_dictionary = corpora.Dictionary(tgt_sentences)
    # pipeline 3
    json.dump(src_train_sentences,
              open(params["src_train_path"], 'w'),
              ensure_ascii=False)
    json.dump(src_test_sentences,
              open(params["src_test_path"], 'w'),
              ensure_ascii=False)
    json.dump(tgt_train_sentences,
              open(params["tgt_train_path"], 'w'),
              ensure_ascii=False)
    json.dump(tgt_test_sentences,
              open(params["tgt_test_path"], 'w'),
              ensure_ascii=False)
    src_dictionary.save(params["src_dict_path"])
    tgt_dictionary.save(params["tgt_dict_path"])
    # pipeline 4
    params["src_word_size"] = len(src_dictionary)
    params["tgt_word_size"] = len(tgt_dictionary)
    params["tgt_sos_id"] = tgt_dictionary.token2id[tgt_sos]
    params["tgt_eos_id"] = tgt_dictionary.token2id[tgt_eos]
    with open(config_path, "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=True)

# ============================== unused below ==============================

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

def generate_train_test(src_file_path,
                        tgt_file_path,
                        test_size,
                        batch_size):
    '''Generate datasets from src_file_path and tgt_file_path, and split them
       into train and test.

       # pipeline
       1. generate train and test sentences.
       2. generate source and target dictionaries.
       3. transform sentences to datasets.

    Args:
        src_file_path: string, source file's path.
        tgt_file_path: string, target file's path.
        batch_size: int, denoting the batch size using for train dataset.
        test_size: the test size when splitting data.

    Returns:
        Iterators of train and test datasets.
    '''
    # pipeline 1
    (
        src_sentences, tgt_sentences,
        (
            src_train_sentences, src_test_sentences,
            tgt_train_sentences, tgt_test_sentences
        )
    ) = split_train_test(src_file_path, tgt_file_path, test_size)
    # pipeline 2
    src_dictionary = generate_dictionary(src_sentences, False)
    tgt_dictionary = generate_dictionary(tgt_sentences, True)
    # pipeline 3
    tgt_eos_id = tgt_dictionary.token2id[tgt_eos]
    train_dataset = process_dataset(
        tf.data.Dataset.zip((
            transform_sentences2dataset(src_train_sentences, src_dictionary),
            transform_sentences2dataset(tgt_train_sentences, tgt_dictionary),
        )),
        batch_size,
        tgt_eos_id
    )
    test_dataset = tf.data.Dataset.zip((
        transform_sentences2dataset(src_test_sentences, src_dictionary),
        transform_sentences2dataset(tgt_test_sentences, tgt_dictionary),
    ))
    # return train and test datasets
    return train_dataset, test_dataset
