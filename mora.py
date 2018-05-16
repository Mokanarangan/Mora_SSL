"""Defines our Mora SSL model
"""
from graph import Graph
import logging
from utils.preprocessing import readEmbeddings


class Mora(Graph):

    def __init__(self, dataset, embedding_file):
        super().__init__(dataset)
        self.embeddings, self.word2Idx = readEmbeddings(
            embedding_file, dataset)

    def build_graph(self, window=3):
        logging.info('Initiating build vector graph..')
        final = self.train + self.test + self.un_labeled
        concat_list = self.find_ngrams(final, window)
        for ngram in concat_list:
            for elm in ngram:
                token = elm['token']
                tag = elm['tag']

    def _get_embeddings(self, token):
        if token not in self.word2Idx:
            print(token)
            token = 'UNKNOWN_TOKEN'
        return self.embeddings[self.word2Idx[token]]
