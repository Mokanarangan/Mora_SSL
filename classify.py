"""Classifier for the built graph
"""
from scipy.sparse import lil_matrix
from collections import defaultdict


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
        self.graph = self._process_graph('./data/' + dataset + '/graph.txt')
        self.train = self._process_info('./data/' + dataset + '/train.txt')
        self.test = self._process_info('./data/' + dataset + '/test.txt')
        self.un_labeled = self._process_info(
            './data/' + dataset + '/un_labeled.txt')

    def classify(self):
        """Classify the data
        """
        total = self.train + self.test + self.un_labeled
        total_size = len(total)
        Graph = lil_matrix((total_size, total_size))
        for ind in range(0, len(total)):
            if(ind > 0):
                x1 = total[ind - 1]['token']
            else:
                x1 = '<new>'
            x2 = total[ind]['token']
            tag = total[ind]['tag']
            if(ind < len(total) - 1):
                x3 = total[ind + 1]['token']
            else:
                x3 = '<new>'
            ngram = ' '.join([x1, x2, x3])
            if(tag != None and tag != 'O'):
                print(ngram, tag)

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
                    graph_dict[node].append(ngram)
        return graph_dict

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
