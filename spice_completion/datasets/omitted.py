import numpy as np
import networkx as nx
from . import helpers as h
from spektral.data import Dataset, Graph
import scipy.sparse as sp
import random
component_types = h.component_types

class OmittedDataset(Dataset):
    def __init__(self, filenames, resample=True, shuffle=True, normalize=True, min_edge_count=0, **kwargs):
        self.filenames = h.valid_netlist_sources(filenames)
        self.resample = resample
        self.shuffle = shuffle
        self.normalize = normalize
        self.epsilon = 0.
        self.mean = 0
        self.std = 1
        self.min_edge_count = min_edge_count
        super().__init__(**kwargs)

    def read(self):
        graphs = []
        for filename in self.filenames:
            graphs.extend(self.load_graphs(filename))

        if self.resample:
            graphs_by_label = {}
            for (i, graph) in enumerate(graphs):
                label = self.graph_label_type(graph)
                if label not in graphs_by_label:
                    graphs_by_label[label] = []
                graphs_by_label[label].append(i)

            counts = [ len(vals) for vals in graphs_by_label.values() ]
            counts.sort()
            middle_idx = len(counts)//2
            median_count = counts[middle_idx]
            #median_count = min(counts)
            print(f'Resampling classes to median size ({median_count})')

            graph_idx = []
            for label_idx in graphs_by_label.values():
                if len(label_idx) > median_count:
                    idx = random.sample(label_idx, median_count)
                else:
                    idx = label_idx

                graph_idx.extend(idx)

            graphs = [ graphs[i] for i in graph_idx ]

        if self.normalize:
            graphs = self.normalize_graphs(graphs)

        return graphs

    def unnormalize(self, graph_nodes):
        return (graph_nodes * (self.std + self.epsilon)) + self.mean

    def normalize_graphs(self, graphs):
        node_count = sum(( graph.x.shape[0] for graph in graphs ))
        graph_nodes = np.concatenate([ graph.x for graph in graphs ], axis=0)
        mean = np.sum(graph_nodes, axis=0) / node_count
        residuals = graph_nodes - mean
        raw_std = np.sum(residuals, axis=0) / node_count
        nonzero_idx = raw_std.nonzero()[0]
        std = np.ones(raw_std.shape)
        std[nonzero_idx] = raw_std[nonzero_idx]
        for graph in graphs:
            graph.x = (graph.x - mean) / (std + self.epsilon)

        self.mean = mean
        self.std = std
        return graphs

    def graph_label_type(self, graph):
        omitted_type_idx = np.argmax(graph.y)
        return omitted_type_idx

    def get_node_types(self, nodes, normalized=True):
        if normalized:
            nodes = self.unnormalize(nodes)
        node_types = np.argmax(nodes > 0.99999, axis=1)
        return node_types

    def load_graph(self, components, adj, omitted_idx):
        embedding_size = len(h.component_types)
        all_component_types = np.array([ h.get_component_type_index(c) for c in components ])
        omitted_type = all_component_types[omitted_idx]
        included_idx = [ i for i in range(len(all_component_types)) if i != omitted_idx]
        component_types = all_component_types[included_idx]
        component_count = component_types.size

        # nodes...
        x = np.zeros((component_count, embedding_size))
        x[np.arange(component_types.size), component_types] = 1

        expanded_adj = np.zeros((x.shape[0], x.shape[0]))
        for (new_i, old_i) in enumerate(included_idx):
            expanded_adj[new_i] = adj[old_i,included_idx]

        # labels...
        y = np.zeros((len(h.component_types),))
        y[omitted_type] = 1

        if self.shuffle:
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x = np.take(x, indices, axis=0)
            expanded_adj = np.take(expanded_adj, indices, axis=0)

        a = sp.csr_matrix(expanded_adj)

        return Graph(x=x, a=a, y=y)

    def load_graphs(self, filename):
        (components, adj) = h.netlist_as_graph(filename)
        count = len(components)

        graphs = ( self.load_graph(components, adj, omitted_idx) for omitted_idx in range(count) )
        return [ graph for graph in graphs if edge_count(graph) >= self.min_edge_count ]

    def to_networkx(self, sgraph):
        graph = nx.Graph()
        node_count = sgraph.x.shape[0]
        nodes = ( (i, {'embedding': sgraph.x[i]}) for i in range(node_count) )
        graph.add_nodes_from(nodes)

        row_idx, col_idx = sgraph.a.nonzero()
        edges = zip(row_idx, col_idx)
        graph.add_edges_from(edges)

        return graph

def edge_count(graph):
    row_idx, col_idx = graph.a.nonzero()
    return len(row_idx)

def load(filenames, **kwargs):
    return OmittedDataset(filenames, **kwargs)
