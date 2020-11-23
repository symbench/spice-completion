import datasets
import sys
import argparse

parser = argparse.ArgumentParser('Print netlists in textual graph data format')
parser.add_argument('files', nargs='+')
args = parser.parse_args()

dataset = datasets.graphdata(args.files)
print(dataset)
