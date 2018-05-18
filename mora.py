"""Defines our Mora SSL model
"""
from graph import Graph
import logging
import numpy as np
from utils.preprocessing import readEmbeddings, save_obj, load_obj, wordNormalize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist
from annoy import AnnoyIndex


class Mora(Graph):

    def __init__(self, dataset, embedding_file):
        super().__init__(dataset)
        self.embeddings, self.word2Idx = readEmbeddings(
            embedding_file, dataset)

    def build_graph(self, window=3):
        logging.info('Initiating build vector graph..')
        final = self.train + self.test + self.un_labeled
        # final = self.train + self.test
        concat_list = self.find_ngrams(final, window)
        tag_dict = dict()
        tag_inv_dict = dict()
        tag_count = 0

        x_train = []
        y_train = []
        embedding_list = []

        ngram_dict = dict()
        total_size = 0

        for i in range(0, len(concat_list)):

            ngram = concat_list[i]
            word_comb = ' '.join([ngram[0]['token'],
                                  ngram[1]['token'], ngram[2]['token']])
            pointer = ngram[1]
            if(pointer['token'] == '</s>'):
                continue
            embedding = None
            for elm in ngram:
                token = elm['token']
                # Mean or concatenation
                if embedding is None:
                    embedding = self._get_embeddings(token)
                else:
                    embedding = embedding + self._get_embeddings(token)
            # embedding = np.divide(embedding, 3)
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
            ngram_dict[total_size] = {'ngram': word_comb}
            total_size += 1
            embedding_list.append(embedding)
        logging.info('Transforming vector..')
        clf = LinearDiscriminantAnalysis()
        clf.fit(np.matrix(x_train), np.array(y_train))
        embedding_list = clf.transform(np.array(embedding_list))
        logging.info('Building Graph')
        ann = AnnoyIndex(embedding_list[0].shape[0], metric='euclidean')

        connected_vertices = dict()

        for i in range(0, total_size):
            ann.add_item(i, embedding_list[i])
        ann.build(10)
        for i in range(0, total_size):
            near_arr = ann.get_nns_by_item(i, 50)
            temp = []
            for near in near_arr:
                temp.append(ngram_dict[near]['ngram'])
            connected_vertices[ngram_dict[i]['ngram']] = temp

        logging.info('Drawing graph')
        f = open('./data/' + self.dataset + '/graph_mora.txt', 'w')
        for key in connected_vertices:
            print(key + '<|>' + '<|>'.join(connected_vertices[key]), file=f)
        logging.info('Graph drawn')

    def _get_embeddings(self, token):
        if token not in self.word2Idx:
            token = 'UNKNOWN_TOKEN'
        return self.embeddings[self.word2Idx[token]]
