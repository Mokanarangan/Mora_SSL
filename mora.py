"""Defines our Mora SSL model
"""
from graph import Graph
import logging
from utils.preprocessing import readEmbeddings


class Mora(Graph):

    def __init__(self, dataset, embedding_file):
        super().__init__(dataset)
        self.embeddings, self.word2Idx = readEmbeddings(embedding_file)

    def build_graph(self, window=3):
        logging.info('Initiating build vector graph..')
        final = self.train + self.test + self.un_labeled
        concat_list = self.find_ngrams(final, window)
        print(self.embeddings[self.word2Idx[concat_list[0][0]['token']]])
        # for i in range(0, len(concat_list)):
        # n_gram = concat_list[i]
