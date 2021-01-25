"""
    This creates a dataset where X is the graph where one component has been removed.
    "Action nodes" have been added for the possible actions to take.
"""
import numpy as np
import random
from . import helpers as h
from spektral.data import Dataset, Graph
import scipy.sparse as sp
import itertools

all_component_types = h.component_types
embedding_size = len(all_component_types) + 1
action_index = len(all_component_types)
np.set_printoptions(threshold=100000)

class OmittedWithActionsDataset(Dataset):
    def __init__(self, filenames, resample=True, **kwargs):
        self.filenames = h.valid_netlist_sources(filenames)
        self.resample = resample
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

        graphs = self.normalize_graphs(graphs)
        return graphs

    def unnormalize(self, graph_nodes):
        return (graph_nodes * (self.std + self.epsilon)) + self.mean

    def normalize_graphs(self, graphs):
        node_count = sum(( graph.x.shape[0] for graph in graphs ))
        graph_nodes = np.concatenate([ graph.x for graph in graphs ], axis=0)
        mean = np.sum(graph_nodes, axis=0) / node_count
        residuals = graph_nodes - mean
        std = np.sum(residuals, axis=0) / node_count
        for graph in graphs:
            graph.x = (graph.x - mean) / (std + self.epsilon)

        self.mean = mean
        self.std = std
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

    def load_graph(self, components, adj, omitted_idx):
        component_count = len(components)
        action_component_count = len(all_component_types)
        total_components = component_count + action_component_count - 1

        component_types = np.array([ h.get_component_type_index(c) for c in components ])
        omitted_type = component_types[omitted_idx]

        # nodes...
        x = np.zeros((total_components, embedding_size))
        x[np.arange(component_types.size), component_types] = 1

        # prototype nodes...
        action_offset = component_types.size
        num_actions = len(all_component_types)
        action_indices = np.zeros(num_actions).astype(int)
        action_indices[0] = omitted_idx
        action_indices[1:] = np.arange(action_offset, action_offset + num_actions - 1)
        action_types = [idx for idx in range(len(all_component_types)) if idx != omitted_type]
        action_types.insert(0, omitted_type)
        action_types = np.array(action_types).astype(int)
        x[action_indices, action_index] = 1
        x[action_indices, action_types] = 1

        expanded_adj = np.zeros((x.shape[0], x.shape[0]))
        # add connectivity to the action nodes (unidirectional)
        expanded_adj[:component_count,action_indices] = 1
        expanded_adj[omitted_idx, :] = 0

        # labels... -1 for nodes to mask
        y = np.zeros((total_components,))
        y[np.arange(component_types.size)] = -1
        y[omitted_idx] = 1

        # shuffle
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = np.take(x, indices, axis=0)
        y = np.take(y, indices, axis=0)
        expanded_adj = np.take(expanded_adj, indices, axis=0)

        a = sp.csr_matrix(expanded_adj)
        return Graph(x=x, a=a, y=y)

    def load_graphs(self, filename):
        (components, adj) = h.netlist_as_graph(filename)
        count = len(components)
        return [ self.load_graph(components, adj, omitted_idx) for omitted_idx in range(count) ]

def load(filenames, **kwargs):
    return OmittedWithActionsDataset(filenames, **kwargs)
