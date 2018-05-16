"""Defines our Mora SSL model
"""
from graph import Graph
import logging
from utils.preprocessing import readEmbeddings


class Mora(Graph):

    def __init__(self, dataset, embedding_file):
        super().__init__(dataset)
        self.embeddings, self.word2Idx = readEmbeddings(embedding_file)

    def build_graph(self):
        logging.info('Initiating build vector graph..')
        print(self.train[20032])
