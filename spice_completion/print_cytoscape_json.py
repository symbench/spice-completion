import numpy as np
import tensorflow as tf

import datasets
import sys
import argparse
from model import model
import json

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--graph', default=-1, type=int)
args = parser.parse_args()

np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# Load data
dataset = datasets.omitted_with_actions(args.files, shuffle=False)
graphs = dataset
if args.graph > -1:
    graphs = [dataset[args.graph]]

def node_id(graph_idx, node_idx):
    return f'g_{graph_idx}/n_{node_idx}'

def create_node(graph_idx, node_idx, data):
    # TODO: Add embedding, label name, etc
    return {
        'data': {
            'id': node_id(graph_idx, node_idx)
        },
        'group': 'nodes',
        'position': {
            'x': 100,
            'y': 100,
        }
    }

def create_edge(graph_idx, source_idx, target_idx):
    return {
        'data': {
            'id': f'{graph_idx}/{source_idx}/{target_idx}',
            'source': node_id(graph_idx, source_idx),
            'target': node_id(graph_idx, target_idx),
        },
        'group': 'edges',
        'position': {
            'x': 100,
            'y': 100,
        }
    }

cytoscape_data = []
for (g_idx, graph) in enumerate(graphs):
    for (n_idx, row) in enumerate(graph.x):
        cytoscape_data.append(create_node(g_idx, n_idx, row))
    # For each entry in the adjacency matrix, add an edge
    row_idx, col_idx = graph.a.nonzero()
    for (source, target) in zip(row_idx, col_idx):  # TODO: confirm this isn't backwards
        cytoscape_data.append(create_edge(g_idx, source, target))

# TODO: add labels
print(json.dumps(cytoscape_data))
