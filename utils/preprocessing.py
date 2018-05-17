import re
import numpy as np
import logging
import gzip
import deepdish as dd
import os.path
import pickle


def wordNormalize(word):
    word = word.lower()
    word = word.replace("--", "-")
    word = re.sub("\"+", '"', word)
    word = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}", 'DATE_TOKEN', word)
    word = re.sub("[0-9]{2}:[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
    word = re.sub("[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
    word = re.sub("[0-9.,]+", 'NUMBER_TOKEN', word)
    return word


def readEmbeddings(embeddingsPath, dataset):
    """
    Reads the embeddingsPath.
    :param embeddingsPath: File path to pretrained embeddings
    :return:
    """

    neededVocab = {}
    # :: Read in word embeddings ::
    logging.info("Read file: %s" % embeddingsPath)
    word2Idx = {}
    embeddings = []
    logging.info("Generate new embeddings files for a dataset")

    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath,
                                                                                               encoding="utf8")

    embeddingsDimension = None

    for line in embeddingsIn:
        split = line.rstrip().split(" ")
        word = split[0]

        if embeddingsDimension == None:
            embeddingsDimension = len(split) - 1

        if (len(
                split) - 1) != embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
            print(
                "ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
            continue

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension)
            embeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)
            embeddings.append(vector)

        vector = np.array([float(num) for num in split[1:]])

        if len(neededVocab) == 0 or word in neededVocab:
            if word not in word2Idx:
                embeddings.append(vector)
                word2Idx[word] = len(word2Idx)

    # Extend embeddings file with new tokens
    embeddings = np.array(embeddings)
    logging.info('Embedding Read')

    logging.info('Embedding Saved')
    return embeddings, word2Idx


def save_obj(obj, name):
    with open('pkl/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('pkl/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def find_ngrams(input_list):
    l = []
    for ind in range(1, len(input_list) - 1):
        x1 = input_list[ind - 1]
        x2 = input_list[ind]
        x3 = input_list[ind + 1]
        l.append((x1, x2, x3))
    return l
