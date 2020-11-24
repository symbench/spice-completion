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
parser.add_argument('files', nargs='+')
parser.add_argument('--prefix', default='./netlist_')
args = parser.parse_args()

def graphdata(filenames):
    for fname in filenames:
        for l in open(fname, 'r'):
            yield l

graph_idx = 0
graph = None
for line in graphdata(args.files):
    if line[0] == 't':
        if graph is not None:
            graph.render(f'{args.prefix}{graph_idx}')
            graph_idx += 1
        graph = Graph(comment=line)
    elif line[0] == 'v':
        _, n_id, class_idx = line.split(' ')
        graph.node(n_id, h.component_index_name(int(class_idx)))
    elif line[0] == 'e':
        _, src, dst, _ = line.split(' ')
        graph.edge(src, dst)

graph.render(f'{args.prefix}{graph_idx}')
