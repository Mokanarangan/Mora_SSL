"""Abstract class graph
"""


class Graph():
    def __init__(self, dataset):
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

    def find_ngrams(self, input_list, n):
        l = []
        for ind in range(0, len(input_list)):
            if(ind > 0):
                x1 = input_list[ind - 1]
            else:
                x1 = {'token': '<new>', 'tag': None}
            x2 = input_list[ind]
            if(ind < len(input_list) - 1):
                x3 = input_list[ind + 1]
            else:
                x3 = {'token': '<new>', 'tag': None}
            l.append((x1, x2, x3))
        return l

    def build_graph(self):
        print("Build Graph Method")