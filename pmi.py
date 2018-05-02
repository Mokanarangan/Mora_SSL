import math
from collections import defaultdict
import numpy as np
import os.path
from scipy.sparse import lil_matrix
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances


class PMI():
    """Subramany et al's PMI model
    """

    def __init__(self, dataset, BIO):
        """Initiate
        Arguments:
            train {string} -- train data path
            un_labeled {string} -- un_labeled data path
            test {string} -- test data path
        """
        self.dataset = dataset
        self.train = self._process_info('./data/' + dataset + '/train.txt')
        self.un_labeled = self._process_info(
            './data/' + dataset + '/un_labeled.txt')
        self.test = self._process_info('./data/' + dataset + '/test.txt')
        print('Number of Train lines: %d' % len(self.train))
        print('Number of Test lines: %d' % len(self.test))
        print('Number of Unlabeled lines: %d' % len(self.un_labeled))

    def _process_info(self, file_name):
        """Process data and stores in variable.
        File should be in 'X X' format
        Arguments:
            file_name {string} -- path of the file
        """

        file = open(file_name)
        data = []
        for line in file:
            split = line.split()
            tag = None
            if line in ['\n', '\r\n']:
                data.append({'token': '<new>', 'tag': tag})
                continue
            if len(split) > 1:
                tag = split[1]
            data.append({'token': split[0], 'tag': tag})
        return data

    def build_graph(self, window=3):
        """build the PMI graph
        """
        print('Extracting n-grams ...')
        n_gram_total = dict()
        total_count = defaultdict(lambda: 0)

        final = self.train + self.test + self.un_labeled
        concat_list = self.find_ngrams(final, window)
        concat_graph_list = self.find_ngrams(self.test + self.train, window)
        count = 0
        traverse_list = defaultdict(list)
        print('Extracting features')

        for i in range(0, len(concat_list)):
            n_gram = concat_list[i]
            word_comb = n_gram[0]['token'] + \
                n_gram[1]['token'] + n_gram[2]['token']
            if(i == 0):
                x1 = '<new>'
            else:
                x1 = concat_list[i - 1][0]['token']

            if(i >= len(concat_list) - 1):
                x5 = '<new>'
            else:
                x5 = concat_list[i + 1][2]['token']
            x2 = n_gram[0]['token']
            x3 = n_gram[1]['token']
            x4 = n_gram[2]['token']
            if word_comb not in n_gram_total:
                n_gram_total[word_comb] = defaultdict(lambda: 0)
            # Features
            trigram_context = x1 + x2 + x3 + x4 + x5
            trigram = x2 + x3 + x4
            left = x1 + x2
            right = x4 + x5
            center = x2
            trigram_center = x2 + x4
            left_word_right = x2 + x4 + x5
            left_context_right = x1 + x2 + x4

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

            traverse_list[left].append(word_comb)
            traverse_list[right].append(word_comb)
            traverse_list[center].append(word_comb)
            traverse_list[trigram_center].append(
                word_comb)
            traverse_list[left_word_right].append(
                word_comb)
            traverse_list[left_context_right].append(
                word_comb)

        print('Features extracted')
        print('Calculating PMI values..')

        unique_graph = dict()
        final_list = []
        feat_count = dict()

        for n_gram in concat_graph_list:
            word_comb = n_gram[0]['token'] + \
                n_gram[1]['token'] + n_gram[2]['token']
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
        print('PMI values calculated')
        chunk_size = 500

        matrix_len = spr_matrix.shape[0]  # Not sparse numpy.ndarray

        def similarity_cosine_by_chunk(start, end):
            if end > matrix_len:
                end = matrix_len
            return cosine_distances(X=spr_matrix[start:end], Y=spr_matrix)

        connected_vertices = dict()

        for chunk_start in range(0, matrix_len, chunk_size):
            print('Analyzing: %d' % chunk_start)
            cosine_similarity_chunk = similarity_cosine_by_chunk(
                chunk_start, chunk_start + chunk_size)
            for i in range(0, len(cosine_similarity_chunk)):
                arr = np.argsort(cosine_similarity_chunk[i])[:6]
                temp = []
                for j in arr:
                    temp.append(final_list[j]['ngram'])
                connected_vertices[final_list[i + chunk_start]['ngram']] = temp

        print('Total ngram count: %d' % count)
        print('Total uique ngram count: %d' % len(unique_graph.keys()))
        print('Total feat count: %d' % len(feat_count.keys()))
        return connected_vertices

    def propagate(self, connected):
        print('Setup propagate')
        total_size = len(self.train + self.test)
        Graph = lil_matrix((total_size, total_size))
        count = 0
        train_dict = defaultdict(lambda: [])
        for ind in range(0, len(self.train)):
            if(ind > 0):
                x1 = self.train[ind - 1]['token']
            else:
                x1 = '<new>'
            x2 = self.train[ind]['token']
            if(ind < len(self.train) - 1):
                x3 = self.train[ind + 1]['token']
            else:
                x3 = '<new>'
            ngram = x1 + x2 + x3
            train_dict[ngram].append(ind)
            if ngram not in connected:
                count += 1
        print(count)

    def find_ngrams(self, input_list, n):
        return list(zip(*[input_list[i:] for i in range(n)]))
