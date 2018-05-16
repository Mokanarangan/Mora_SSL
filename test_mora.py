import argparse
from mora import Mora
from classify import Classifier

parser = argparse.ArgumentParser()


parser.add_argument(
    '--dataset', help='Location of word dataset files', type=str, required='true'
)
parser.add_argument(
    '--embeddings', help='Location of embedding file', type=str, required='true'
)
parser.add_argument(
    '--BIO', help='State if its bio_tag', type=bool, default=False
)

args = parser.parse_args()
# Build the graph
pmi = Mora(args.dataset, args.embeddings)
connected = pmi.build_graph()
# Performs classification
classifier = Classifier(args.dataset, args.BIO)
classifier.classify()
