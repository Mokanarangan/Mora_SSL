"""Classifier for the built graph
"""
from scipy.sparse import lil_matrix
from collections import defaultdict
from label_propagation import LGC, HMN
import numpy as np
import string


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
        self.graph = self._process_graph(
            './data/' + dataset + '/' + graph_name)
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
        for ind in range(1, len(total) - 1):
            x1 = total[ind - 1]['token']
            x2 = total[ind]['token']
            tag = total[ind]['tag']
            if(tag != None and tag in string.punctuation):
                tag = 'O'
            if(x2 == '</s>'):
                continue
            test = total[ind]['test']
            train = total[ind]['train']
            x3 = total[ind + 1]['token']
            ngram = ' '.join([x1, x2, x3])
            if(tag not in tag_dict and tag != None):
                tag_dict[tag] = tag_count
                tag_inv_dict[tag_count] = tag
                tag_count += 1
            tag_val = None
            if(tag != None):
                tag_val = tag_dict[tag]

            setup = {'ngram': ngram,
                     'token': x2}
            if(ngram in ngram_dict):
                arr = ngram_dict[ngram]['index']
                test_arr = ngram_dict[ngram]['test']
                train_arr = ngram_dict[ngram]['train']
                tag_arr = ngram_dict[ngram]['tag']
                arr.append(ind)
                test_arr.append(test)
                train_arr.append(train)
                tag_arr.append(tag_val)

                setup['index'] = arr
                setup['test'] = test_arr
                setup['train'] = train_arr
                setup['tag'] = tag_arr
            else:
                setup['index'] = [ind]
                setup['test'] = [test]
                setup['train'] = [train]
                setup['tag'] = [tag_val]
            ngram_dict[ngram] = setup
            ngram_index_dict[ind] = {'tag': tag, 'token': x2}
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        print('Fitting graph')
        count = 0
        fl = open('tt.txt')
        for node in self.graph:
            if(node in ngram_dict):
                index_arr = ngram_dict[node]['index']
                train_arr = ngram_dict[node]['train']
                test_arr = ngram_dict[node]['test']
                tag_arr = ngram_dict[node]['tag']
                token = ngram_dict[node]['token']
                if(token == '</s>'):
                    continue
                for ind in range(0, len(index_arr)):
                    index = index_arr[ind]
                    if(train_arr[ind]):
                        print(node, token, tag_arr[ind], file=fl)
                        x_train.append(index)
                        y_train.append(tag_arr[ind])
                    elif(test_arr[ind]):
                        x_test.append(index)
                        y_test.append(tag_arr[ind])
                    for connected in self.graph[node]:
                        Graph[index, ngram_dict[connected]['index']] = 1
        print(count)
        print(len(x_train), len(x_test))
        print('Classifying')
        clf = HMN(graph=Graph, max_iter=1000)
        x_train = np.array(x_train)
        print(y_train)
        y_train = np.array(y_train)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(np.array(x_test))
        print('Predicting')
        x_test.sort()
        total = 0
        correct = 0
        for ind in range(0, len(x_test)):
            elm = ngram_index_dict[x_test[ind]]
            if(elm['tag'] != None):
                total += 1
            print(elm['token'], elm['tag'], tag_inv_dict[y_predict[ind]])
            if(elm['tag'] == tag_inv_dict[y_predict[ind]]):
                correct += 1

        print(correct / total)

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
