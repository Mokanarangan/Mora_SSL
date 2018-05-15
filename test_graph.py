import argparse
from pmi import PMI
from classify import Classifier

parser = argparse.ArgumentParser()


parser.add_argument(
    '--dataset', help='Location of word dataset files', type=str, required='true'
)
parser.add_argument(
    '--BIO', help='State if its bio_tag', type=bool, default=False
)

args = parser.parse_args()
# Build the graph
pmi = PMI(args.dataset)
# connected = pmi.build_graph()

# Performs classification
classifier = Classifier(args.dataset, args.BIO)
classifier.classify()
