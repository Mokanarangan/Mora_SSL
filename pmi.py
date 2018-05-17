import math
from collections import defaultdict
import numpy as np
import os.path
from scipy.sparse import lil_matrix
import pickle
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from graph import Graph


class PMI(Graph):
    """Subramany et al's PMI model
    """

    def build_graph(self, window=3):
        """build the PMI graph
        """
        logging.info('Extracting n-grams ...')
        n_gram_total = dict()
        total_count = defaultdict(lambda: 0)

        final = self.train + self.test + self.un_labeled
        concat_list = self.find_ngrams(final, window)
        count = 0
        logging.info('Extracting features')

        for i in range(0, len(concat_list)):
            n_gram = concat_list[i]
            if(n_gram[1]['token'] == '</s>'):
                continue
            word_comb = ' '.join([n_gram[0]['token'],
                                  n_gram[1]['token'], n_gram[2]['token']])
            if(i == 0):
                x1 = '</s>'
            else:
                x1 = concat_list[i - 1][0]['token']

            if(i >= len(concat_list) - 1):
                x5 = '</s>'
            else:
                x5 = concat_list[i + 1][2]['token']
            x2 = n_gram[0]['token']
            x3 = n_gram[1]['token']
            x4 = n_gram[2]['token']
            if word_comb not in n_gram_total:
                n_gram_total[word_comb] = defaultdict(lambda: 0)
            # Features
            trigram_context = ' '.join([x1, x2, x3, x4, x5])
            trigram = ' '.join([x2, x3, x4])
            left = ' '.join([x1, x2])
            right = ' '.join([x4, x5])
            center = x2
            trigram_center = ' '.join([x2, x4])
            left_word_right = ' '.join([x2, x4, x5])
            left_context_right = ' '.join([x1, x2, x4])

            n_gram_total[word_comb][trigram_context] += 1
            total_count[trigram_context] += 1
            n_gram_total[word_comb][trigram] += 1
            total_count[trigram] += 1
            n_gram_total[word_comb][left] += 1
            total_count[left] += 1
            n_gram_total[word_comb][right] += 1
            total_count[right] += 1
            n_gram_total[word_comb][center] += 1
            total_count[center] += 1
            n_gram_total[word_comb][trigram_center] += 1
            total_count[trigram_center] += 1
            n_gram_total[word_comb][left_word_right] += 1
            total_count[left_word_right] += 1
            n_gram_total[word_comb][left_context_right] += 1
            total_count[left_context_right] += 1

        logging.info('Features extracted')
        logging.info('Calculating PMI values..')

        unique_graph = dict()
        final_list = []
        feat_count = dict()

        for n_gram in concat_list:
            word_comb = ' '.join([n_gram[0]['token'],
                                  n_gram[1]['token'], n_gram[2]['token']])
            if word_comb not in unique_graph and word_comb in n_gram_total:
                unique_graph[word_comb] = n_gram_total[word_comb]
                final_list.append(
                    {'ngram': word_comb, 'feat': n_gram_total[word_comb]})
                for key in n_gram_total[word_comb].keys():
                    if key not in feat_count:
                        feat_count[key] = len(feat_count.keys())
            count += 1

        spr_matrix = lil_matrix(
            (len(final_list), len(feat_count.keys()) + 1), dtype=np.float)

        total = len(concat_list) * 8
        for i in range(0, len(final_list)):
            key = final_list[i]['ngram']
            for key2 in unique_graph[key].keys():
                pmi_val = math.log((unique_graph[key][key2] / total) /
                                   ((total_count[key] / total) * (total_count[key2] / total)), 2)
                spr_matrix[i, feat_count[key2]] = pmi_val + 1
        logging.info('PMI values calculated')
        chunk_size = 500

        matrix_len = spr_matrix.shape[0]  # Not sparse numpy.ndarray

        def similarity_cosine_by_chunk(start, end):
            if end > matrix_len:
                end = matrix_len
            return cosine_distances(X=spr_matrix[start:end], Y=spr_matrix)

        connected_vertices = dict()

        for chunk_start in range(0, matrix_len, chunk_size):
            logging.info('Analyzing: %d' % chunk_start)
            cosine_similarity_chunk = similarity_cosine_by_chunk(
                chunk_start, chunk_start + chunk_size)
            for i in range(0, len(cosine_similarity_chunk)):
                arr = np.argsort(cosine_similarity_chunk[i])[:50]
                temp = []
                for j in arr:
                    temp.append(final_list[j]['ngram'])
                connected_vertices[final_list[i + chunk_start]['ngram']] = temp
        logging.info('Drawing graph')
        f = open('./data/' + self.dataset + '/graph.txt', 'w')
        for key in connected_vertices:
            print(key + '<|>' + '<|>'.join(connected_vertices[key]), file=f)
        logging.info('Graph drawn')
        logging.info('Total ngram count: %d' % count)
        logging.info('Total uique ngram count: %d' % len(unique_graph.keys()))
        logging.info('Total feat count: %d' % len(feat_count.keys()))
        return connected_vertices
