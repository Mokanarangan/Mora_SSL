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
        self.dataset = dataset
        self.graph = self._process_graph('./data/' + dataset + '/graph.txt')
        self.train = self._process_info('./data/' + dataset + '/train.txt')
        self.test = self._process_info('./data/' + dataset + '/test.txt')

    def classify(self):
        """Classify the data
        """
        total_size = len(self.train + self.test)
        Graph = lil_matrix((total_size, total_size))
        train_dict = defaultdict(lambda: [])

    def _process_graph(self, file_name):
        """Process the created in the graph file
        """
        f = open(file_name)
        for line in f:
            split_list = line.split("|")
            if(len(split_list) != 7):
                print('here')

    def _ngram(self, data):
        """Setup the ngram
        """
        ngram_dict = defaultdict(lambda: [])
        for ind in range(0, len(data)):
            if(ind > 0):
                x1 = data[ind - 1]['token']
            else:
                x1 = '<new>'
            x2 = data[ind]['token']
            if(ind < len(data) - 1):
                x3 = data[ind + 1]['token']
            else:
                x3 = '<new>'
            ngram = ' '.join([x1, x2, x3])
            ngram_dict[ngram].append(ind)

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
