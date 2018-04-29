import math


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
        final = self.train + self.test + self.un_labeled
        concat_list = self.find_ngrams(final, window)
        count = 0
        for n_gram in concat_list:
            word_comb = n_gram[0]['token'] + "|" + \
                n_gram[1]['token'] + "|" + n_gram[2]['token']
            count += 1
            self.n_gram[word_comb] = True
        print('Total ngram count: %d' % count)
        print('Total unique ngram count: %d' % len(self.n_gram.keys()))

    def find_ngrams(self, input_list, n):
        return zip(*[input_list[i:] for i in range(n)])
