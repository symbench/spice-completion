"""
    This creates a dataset where X is the graph where one component has been removed.
    "Action nodes" have been added for the possible actions to take.
"""
import numpy as np
from . import helpers as h
from spektral.data import Dataset, Graph
import scipy.sparse as sp

all_component_types = h.component_types
embedding_size = len(all_component_types) + 1
action_index = len(all_component_types)
np.set_printoptions(threshold=100000)

class OmittedWithActionsDataset(Dataset):
    def __init__(self, filenames, **kwargs):
        self.filenames = h.valid_netlist_sources(filenames)
        super().__init__(**kwargs)

    def read(self):
        graphs = []
        for filename in self.filenames:
            graphs.extend(self.load_graphs(filename))
        return graphs

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

def load(filenames):
    return OmittedWithActionsDataset(filenames)