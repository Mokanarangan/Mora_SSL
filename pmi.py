class PMI():
    """Subramany et al's PMI model
    """

    def __init__(self, train, dev, test):
        """Initiate 
        Arguments:
            train {string} -- train data path
            dev {string} -- dev data path
            test {string} -- test data path
        """

        self.train = train
        self.dev = dev
        self.test = test

    def build_graph(self):
        """build the PMI graph
        """
        f = open(self.train)
