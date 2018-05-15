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
        self.graph = self._process_info('./data/' + dataset + '/graph.txt')
        self.train = self._process_info('./data/' + dataset + '/train.txt')
        self.test = self._process_info('./data/' + dataset + '/test.txt')

    def classify(self):
        """Classify the data
        """
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
            ngram = ' '.join([x1, x2, x3])
            train_dict[ngram].append(ind)
            if ngram not in connected:
                count += 1
        print(count)

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
