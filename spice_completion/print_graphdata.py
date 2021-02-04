import datasets
import sys
import argparse

parser = argparse.ArgumentParser('Print netlists in textual graph data format')
parser.add_argument('files', nargs='+')
parser.add_argument('--names', action='store_true', default=False)
args = parser.parse_args()

dataset = datasets.graphdata(args.files, args.names)
print(dataset)
