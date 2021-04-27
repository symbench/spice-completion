from PySpice.Spice.Parser import SpiceParser
from . import helpers as h
from . import augmenters as a
import sys
import pickle
import networkx as nx
import torch

def fetch_graphs(files, min_edges=0):
    filenames = h.valid_netlist_sources(files)
    source = next(filenames)

    parser = SpiceParser(source=source)
    circuit = parser.build_circuit()
    (nodes, edges) = h.get_nodes_edges(circuit)

    # TODO: randomly remove one each time until it is empty
    graphs = []
    while len(edges) > min_edges:
        graphs.append((nodes, edges))
        (nodes, edges) = a.remove_random_node(nodes, edges)

    return graphs

def encode_component(data):
    # TODO
    component = data['component']
    tensor = torch.ones(len(h.component_types))
    tensor[h.get_component_type_index(component)] = 1.0
    return {'node_feature': tensor}

def mkgraph(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from([ (n_id, encode_component(component)) for (n_id, component) in nodes ])
    graph.add_edges_from([ (src, dst, {'edge_label': d['pin']}) for (src, dst, d) in edges ])  # TODO: encode the edges...
    return graph

# TODO: add parsing, transformations here

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('netlists', nargs='+')
    parser.add_argument('--min-edge-count', default=0, type=int)
    parser.add_argument('--outfile', default='graphs.pkl', type=str)
    args = parser.parse_args()

    graphs = fetch_graphs(args.netlists, args.min_edge_count)
    outfile = args.outfile.replace('.pkl', '') + '.pkl'
    with open(args.outfile, 'wb') as outfile:
        pickle.dump([ mkgraph(n, e) for (n, e) in graphs ], outfile)

    print(f'Saved {len(graphs)} graphs as {args.outfile}')

