class PMI():
    """Subramany et al's PMI model
    """

    def __init__(self, dataset):
        """Initiate 
        Arguments:
            train {string} -- train data path
            un_labeled {string} -- un_labeled data path
            test {string} -- test data path
        """

        self.tag_info = dict()
        self.train = self._process_info('./data/' + dataset + '/train.txt')
        self.un_labeled = self._process_info(
            './data/' + dataset + '/un_labeled.txt')
        self.test = self._process_info('./data/' + dataset + '/test.txt')

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
            tag = -1
            if line in ['\n', '\r\n']:
                data.append({'token': None})
                continue
            if len(split) > 1:
                tag = split[1]
            if tag not in self.tag_info and tag != -1:
                self.tag_info[tag] = len(self.tag_info.keys())
            data.append({'token': split[0], 'tag': tag})

    def build_graph(self):
        """build the PMI graph
        """
        print('buidling print graph')
