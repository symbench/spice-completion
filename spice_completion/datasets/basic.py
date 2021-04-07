"""
    This is a standard dataset w/o any augmentation
"""
import numpy as np
import random
from . import helpers as h
from spektral.data import Dataset, Graph
import scipy.sparse as sp
import itertools

all_component_types = h.component_types
embedding_size = len(all_component_types)  # FIXME: Test the size
action_index = len(all_component_types)

class LinkDataset(Dataset):
    def __init__(self, filenames, resample=True, normalize=True, mean=None, std=None, **kwargs):
        self.filenames = h.valid_netlist_sources(filenames)
        self.resample = resample
        self.normalize = normalize
        if normalize:
            self.mean = mean
            self.std = std
        else:
            self.mean = 0
            self.std = 1
        self.epsilon = 0.
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
            median_count = min(counts)
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
        if self.mean is None or self.std is None:
            node_count = sum(( graph.x.shape[0] for graph in graphs ))
            graph_nodes = np.concatenate([ graph.x for graph in graphs ], axis=0)
            mean = np.sum(graph_nodes, axis=0) / node_count
            residuals = graph_nodes - mean
            raw_std = np.sum(residuals, axis=0) / node_count
            nonzero_idx = raw_std.nonzero()[0]
            std = np.ones(raw_std.shape)
            std[nonzero_idx] = raw_std[nonzero_idx]
            self.mean = mean
            self.std = std

        for graph in graphs:
            graph.x = (graph.x - self.mean) / (self.std + self.epsilon)

        return graphs


    def graph_label_type(self, graph):
        label_idx = np.argmax(graph.y)
        class_idx = np.argmax(graph.x[label_idx])
        return class_idx

    def get_node_types(self, nodes, normalized=True):
        if normalized:
            nodes = self.unnormalize(nodes)
        node_types = np.argmax(nodes > 0.99999, axis=1)
        return node_types

    @staticmethod
    def load_graph(components, adj):
        component_count = len(components)
        component_types = np.array([ h.get_component_type_index(c) for c in components ])

        # nodes...
        x = np.zeros((component_count, embedding_size))
        x[np.arange(component_types.size), component_types] = 1

        a = sp.csr_matrix(adj)
        return Graph(x=x, a=a)

    def load_graphs(self, source):
        (components, adj) = h.netlist_as_graph(source)
        graphs = [ self.load_graph(components, adj) ]
        return graphs


def load(filenames, **kwargs):
    return LinkDataset(filenames, **kwargs)
