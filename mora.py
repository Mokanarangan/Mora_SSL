"""Defines our Mora SSL model
"""
from graph import Graph
import logging
import numpy as np
from utils.preprocessing import readEmbeddings, save_obj, load_obj, wordNormalize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist


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
                    embedding = np.concatenate(
                        embedding, self._get_embeddings(token))
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
            ngram_dict[i] = {'ngram': word_comb}
            embedding_list.append(embedding)
        clf = LinearDiscriminantAnalysis()
        clf.fit(np.matrix(x_train), np.array(y_train))
        embedding_list = clf.transform(np.array(embedding_list))

        matrix_len = len(embedding_list)
        chunk_size = 100

        def similarity_by_chunk(start, end):
            if end > matrix_len:
                end = matrix_len
            return euclidean_distances(X=embedding_list[start:end], Y=embedding_list)
            # return cdist(embedding_list[start:end], embedding_list, 'euclidean')

        connected_vertices = dict()
        for chunk_start in range(0, matrix_len, chunk_size):
            logging.info('Analyzing: %d' % chunk_start)
            similarity_chunk = similarity_by_chunk(
                chunk_start, chunk_start + chunk_size)
            logging.info('Sorting: %d' % chunk_start)
            for i in range(0, len(similarity_chunk)):
                arr = np.argsort(similarity_chunk[i])[:50]
                temp = []
                for j in arr:
                    if j in ngram_dict:
                        temp.append(ngram_dict[j]['ngram'])
                if (i + chunk_start) in ngram_dict:
                    connected_vertices[ngram_dict[i +
                                                  chunk_start]['ngram']] = temp
        logging.info('Drawing graph')
        f = open('./data/' + self.dataset + '/graph_mora.txt', 'w')
        for key in connected_vertices:
            print(key + '<|>' + '<|>'.join(connected_vertices[key]), file=f)
        logging.info('Graph drawn')

    def _get_embeddings(self, token):
        if token not in self.word2Idx:
            token = 'UNKNOWN_TOKEN'
        return self.embeddings[self.word2Idx[token]]
