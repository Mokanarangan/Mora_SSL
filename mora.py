"""Defines our Mora SSL model
"""
from graph import Graph
from utils.preprocessing import readEmbeddings


class Mora(Graph):

    def __init__(self, dataset, embedding_file):
        super().__init__(dataset)
        readEmbeddings(embedding_file)

    def build_graph(self):
        print('Initiating build vector graph..')
