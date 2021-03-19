import spice_completion.datasets as datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('--min-edge-count', default=0, type=int)
args = parser.parse_args()

dataset = datasets.omitted(args.files, min_edge_count=args.min_edge_count)
print(len(dataset))  # prints 1270 when running with kicad_github/*
