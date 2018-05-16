"""Defines our Mora SSL model
"""
from graph import Graph
import logging
import numpy as np
from utils.preprocessing import readEmbeddings, save_obj, load_obj, wordNormalize


class Mora(Graph):

    def __init__(self, dataset, embedding_file):
        super().__init__(dataset)
        self.embeddings, self.word2Idx = readEmbeddings(
            embedding_file, dataset)

    def build_graph(self, window=3):
        logging.info('Initiating build vector graph..')
        final = self.train + self.test + self.un_labeled
        concat_list = self.find_ngrams(final, window)
        tag_dict = dict()
        tag_inv_dict = dict()
        tag_count = 0

        x_train = []
        y_train = []
        embedding_list = []

        ngram_dict = dict()

        for i in range(0, len(concat_list)):
            ngram = concat_list[i]
            pointer = ngram[1]
            embedding = None
            for elm in ngram:
                token = elm['token']
                # Mean or concatenation
                if embedding is None:
                    embedding = self._get_embeddings(token)
                else:
                    embedding = np.add(embedding, self._get_embeddings(token))
            embedding = np.divide(embedding, 3)
            tag = pointer['tag']

            if(tag != None):
                if(tag not in tag_dict):
                    tag_dict[tag] = tag_count
                    tag_inv_dict[tag_count] = tag
                    tag_count += 1
            test = pointer['test']
            train = pointer['train']
            if train:
                x_train.append(embedding)
                y_train.append(tag_dict[tag])
            ngram_dict[i] = ngram
            embedding_list.append(embedding)

        matrix_len = embedding_list[0].shape[0]
        print(matrix_len)

    def _get_embeddings(self, token):
        if token not in self.word2Idx:
            token = 'UNKNOWN_TOKEN'
        return self.embeddings[self.word2Idx[token]]
