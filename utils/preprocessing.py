import re
import numpy as np
import logging
import gzip
import deepdish as dd


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
    logging.info("Generate new embeddings files for a dataset")

    neededVocab = {}
    # :: Read in word embeddings ::
    logging.info("Read file: %s" % embeddingsPath)
    word2Idx = {}
    embeddings = []

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
    dd.io.save('./pkl/' + dataset + '.h5',
               {'embeddings': embeddings, 'word2Idx': word2Idx})

    return embeddings, word2Idx
