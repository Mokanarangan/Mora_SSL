"""Classifier for the built graph
"""
from scipy.sparse import lil_matrix
from collections import defaultdict
from label_propagation import LGC
import numpy as np


class Classifier():

    def __init__(self, dataset, BIO):
        """Initiate
        Arguments:
            train {string} -- train data path
            un_labeled {string} -- un_labeled data path
            test {string} -- test data path
        """
        print('Processing the classifier')
        self.dataset = dataset
        self.BIO = BIO
        self.graph = self._process_graph('./data/' + dataset + '/graph.txt')
        self.train = self._process_info(
            './data/' + dataset + '/train.txt', False, True)
        self.test = self._process_info(
            './data/' + dataset + '/test.txt', True, False)
        self.un_labeled = self._process_info(
            './data/' + dataset + '/un_labeled.txt')

    def classify(self):
        """Classify the data
        """
        total = self.train + self.test + self.un_labeled
        total_size = len(total)
        Graph = lil_matrix((total_size, total_size))
        ngram_dict = dict()
        ngram_index_dict = dict()
        tag_dict = dict()
        tag_inv_dict = dict()
        tag_count = 0
        for ind in range(0, len(total)):
            if(ind > 0):
                x1 = total[ind - 1]['token']
            else:
                x1 = '<new>'
            x2 = total[ind]['token']
            tag = total[ind]['tag']
            test = total[ind]['test']
            train = total[ind]['train']
            if(ind < len(total) - 1):
                x3 = total[ind + 1]['token']
            else:
                x3 = '<new>'
            ngram = ' '.join([x1, x2, x3])
            if(tag not in tag_dict):
                tag_dict[tag] = tag_count
                tag_inv_dict[tag_count] = tag
                tag_count += 1
            setup = {'ngram': ngram,
                     'token': x2, 'tag': tag_dict[tag], 'test': test, 'train': train}
            if(ngram in ngram_dict):
                arr = ngram_dict[ngram]['index']
                arr.append(ind)
                setup['index'] = arr

            else:
                setup['index'] = [ind]
            ngram_dict[ngram] = setup
            ngram_index_dict[ind] = {'tag': tag, 'token': x2}
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        print('Fitting graph')
        for node in self.graph:
            if(node in ngram_dict):
                index_arr = ngram_dict[node]['index']
                for index in index_arr:
                    if(ngram_dict[node]['train']):
                        if(ngram_dict[node]['token'] == 'Reuters'):
                            print(ngram_dict[node])
                        x_train.append(index)
                        y_train.append(ngram_dict[node]['tag'])
                    elif(ngram_dict[node]['test']):
                        x_test.append(index)
                        y_test.append(ngram_dict[node]['tag'])
                    for connected in self.graph[node]:
                        Graph[index, ngram_dict[connected]['index']] = 1
        print(len(x_train), len(x_test))
        print('Classifying')
        clf = LGC(graph=Graph, max_iter=1000)
        clf.fit(np.array(x_train), np.array(y_train))
        y_predict = clf.predict(np.array(x_test))
        for ind in x_train:
            print(ngram_index_dict[ind])
        # for ind in range(0, len(x_test)):
            # elm = ngram_index_dict[x_test[ind]]
            # print(elm['token'], elm['tag'], tag_inv_dict[y_predict[ind]])

    def _process_graph(self, file_name):
        """Process the created in the graph file
        """
        f = open(file_name)
        graph_dict = defaultdict(lambda: [])
        for line in f:
            split_list = line.split("<|>")
            node = split_list[0]
            for ind in range(1, len(split_list)):
                ngram = split_list[ind]
                if(node != ngram):
                    graph_dict[node].append(ngram.replace('\n', ''))
        return graph_dict

    def _process_info(self, file_name, test=False, train=False):
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
                data.append({'token': '<new>', 'tag': tag,
                             'test': test, 'train': train})
                continue
            if len(split) > 1:
                if(self.BIO and split[1] != 'O'):
                    tag = split[1].split('-')[1]
                else:
                    tag = split[1]
            data.append(
                {'token': split[0], 'tag': tag, 'test': test, 'train': train})
        return data
