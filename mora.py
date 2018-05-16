"""Defines our Mora SSL model
"""
from graph import Graph
import logging
from utils.preprocessing import readEmbeddings


class Mora(Graph):

    def __init__(self, dataset, embedding_file):
        super().__init__(dataset)
        embeddings, word2Idx = readEmbeddings(embedding_file)
        print(embeddings)

    def build_graph(self):
        logging.info('Initiating build vector graph..')
