import math
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


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
        self.n_gram = dict()
        self.graph_n_gram = dict()
        total_count = defaultdict(lambda: 0)

        final = self.train + self.test + self.un_labeled
        concat_list = self.find_ngrams(final, window)
        concat_graph_list = self.find_ngrams(self.test + self.train, window)
        count = 0
        print('Extracting features')

        for i in range(0, len(concat_list)):
            n_gram = concat_list[i]
            word_comb = n_gram[0]['token'] + "|" + \
                n_gram[1]['token'] + "|" + n_gram[2]['token']
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
            if word_comb not in self.n_gram:
                self.n_gram[word_comb] = defaultdict(lambda: 0)
            # Features
            trigram_context = x1 + x2 + x3 + x4 + x5
            trigram = x2 + x3 + x4
            left = x1 + x2
            right = x4 + x5
            center = x2
            trigram_center = x2 + x4
            left_word_right = x2 + x4 + x5
            left_context_right = x1 + x2 + x4

            self.n_gram[word_comb][trigram_context] += 1
            total_count[trigram_context] += 1
            self.n_gram[word_comb][trigram] += 1
            total_count[trigram] += 1
            self.n_gram[word_comb][left] += 1
            total_count[left] += 1
            self.n_gram[word_comb][right] += 1
            total_count[right] += 1
            self.n_gram[word_comb][center] += 1
            total_count[center] += 1
            self.n_gram[word_comb][trigram_center] += 1
            total_count[trigram_center] += 1
            self.n_gram[word_comb][left_word_right] += 1
            total_count[left_word_right] += 1
            self.n_gram[word_comb][left_context_right] += 1
            total_count[left_context_right] += 1

        print('Features extracted')
        print('Calculating PMI values')

        for key in self.n_gram.keys():
            for feat in self.n_gram[key].keys():
                print(key, feat)

        print('PMI values calculated')

        for n_gram in concat_graph_list:
            word_comb = n_gram[0]['token'] + "|" + \
                n_gram[1]['token'] + "|" + n_gram[2]['token']
            self.graph_n_gram[word_comb] = True
            count += 1

        print('Total ngram count: %d' % count)
        print('Total unique ngram count: %d' % len(self.graph_n_gram.keys()))
        print('Calculating nearest neighbors..')

        def distance_fun(x, y):
            return 0
        nbrs = NearestNeighbors(
            n_neighbors=4, algorithm='ball_tree', metric=distance_fun)

    def find_ngrams(self, input_list, n):
        return list(zip(*[input_list[i:] for i in range(n)]))
