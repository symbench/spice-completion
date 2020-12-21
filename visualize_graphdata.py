"""
Generate PNGs of graphdata
"""
import sys
import argparse
from itertools import chain
from os import path
from graphviz import Graph
from datasets import helpers as h

parser = argparse.ArgumentParser('Write subgraphs to netlists')
parser.add_argument('files', nargs='*')
parser.add_argument('--prefix', default='./netlist_')
parser.add_argument('--names', action='store_true', default=False)
args = parser.parse_args()

def graphdata(filenames):
    for fname in filenames:
        for l in open(fname, 'r'):
            yield l

graph_idx = 0
graph = None
lines = graphdata(args.files) if len(args.files) > 0 else sys.stdin
for line in lines:
    if line[0] == 't':
        if graph is not None:
            graph.render(f'{args.prefix}{graph_idx}')
            graph_idx += 1
        graph = Graph(comment=line, format='png')
    elif line[0] == 'v':
        _, n_id, class_idx = line.split(' ')
        label = ' '.join(line.split(' ')[2:]) if args.names else h.component_index_name(int(class_idx))
        graph.node(n_id, label)
    elif line[0] == 'e':
        _, src, dst, _ = line.split(' ')
        graph.edge(src, dst)

graph.render(f'{args.prefix}{graph_idx}')
