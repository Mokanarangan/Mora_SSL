import argparse
from pmi import PMI

parser = argparse.ArgumentParser()


parser.add_argument(
    '--dataset', help='Location of word dataset files', type=str, required='true'
)
parser.add_argument(
    '--BIO', help='State if its bio_tag', type=bool, default=False
)

args = parser.parse_args()
pmi = PMI(args.dataset, args.BIO)
connected = pmi.build_graph()
pmi.build_graph()
print(connected)
