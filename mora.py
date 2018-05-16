"""Defines our Mora SSL model
"""
from graph import Graph


class Mora(Graph):

    def __init__(self, dataset, embedding_file):
        super().__init__(dataset)
        print('Loading Embeddings')

    def build_graph(self):
        print('Initiating build vector graph..')
