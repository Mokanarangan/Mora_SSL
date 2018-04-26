from annoy import AnnoyIndex


class Graph_Model():
    """Graph model used for semi supervised learning
    """

    def __init__(self, vertices, metric='euclidean', load_graph=False):
        self.metric = metric
        self.load_graph = load_graph
        self.vertices = vertices

    def build_graph(self):
        print('Bulding graph ....')
        if(self.load_graph):
            with open('dimension.txt', 'r') as f:
                self.t = AnnoyIndex(int(f.readline()), metric='euclidean')
                self.t.load('graph.ann')
                print('Graph file loaded from file')
        else:
            print('building graph')
            self.t = AnnoyIndex(self.vertices.shape[1], metric='euclidean')
            for i in range(0, self.vertices.shape[0]):
                self.t.add_item(i, self.vertices[i])
            self.t.build(10)
            print('graph built')
            print('graph saving')
            self.t.save('graph.ann')
            with open('dimension.txt', 'w') as f:
                f.write('%d' % self.vertices.shape[1])
