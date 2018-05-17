"""Classifier for the built graph
"""
from scipy.sparse import lil_matrix
from collections import defaultdict
from label_propagation import LGC, HMN
import numpy as np
import string
from utils.preprocessing import find_ngrams


class Classifier():

    def __init__(self, dataset, BIO, graph_name='graph.txt'):
        """Initiate
        Arguments:
            train {string} -- train data path
            un_labeled {string} -- un_labeled data path
            test {string} -- test data path
        """
        print('Processing the classifier')
        self.dataset = dataset
        self.BIO = BIO
        self.graph_name = graph_name
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
        ngrams = find_ngrams(total)
        fl = open('tt.txt', 'w')
        tag_dict = dict()
        tag_count = 0
        X_train = []
        X_test = []
        for index in range(0, len(ngrams)):
            ngram = ngrams[index]
            word_comb = ''
            for elm in ngram:
                word_comb = word_comb + ' ' + elm['token']
            tag = ngram[1]['tag']
            token = ngram[1]['token']
            train = ngram[1]['train']
            test = ngram[1]['test']
            if(token == '</s>'):
                continue
            if(tag != None and tag in string.punctuation):
                tag = 'O'
            if(train or test):
                if(tag not in tag_dict):
                    tag_dict[tag] = tag_count
                    tag_count += 1
                if(train):
                    X_train.append(tag_dict[tag])
                elif(test):
                    X_test.append(tag_dict[tag])

    def _process_graph(self, file_name):
        """Process the created in the graph file
        """
        f = open(file_name)
        graph_dict = defaultdict(lambda: [])
        for line in f:
            split_list = line.split("<|>")
            node = split_list[0]
            for ind in range(2, len(split_list)):
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
                data.append({'token': '</s>', 'tag': tag,
                             'test': test, 'train': train})
                continue
            if len(split) > 1:
                if(self.BIO and split[1] != 'O'):
                    tag = split[1].split('-')[1]
                else:
                    tag = split[1]
            data.append(
                {'token': split[0].replace('\n', ''), 'tag': tag, 'test': test, 'train': train})
        return data
