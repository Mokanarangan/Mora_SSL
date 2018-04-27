import argparse
from pmi import PMI

parser = argparse.ArgumentParser()


parser.add_argument(
    '--dataset', help='Location of word dataset files', type=str, required='true'
)

args = parser.parse_args()
pmi = PMI(args.dataset)
