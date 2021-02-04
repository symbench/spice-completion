import numpy as np
from . import helpers as h
component_types = h.component_types

def load(textfiles):
    sources = h.valid_netlist_sources(textfiles)
    graphs = [ h.netlist_as_graph(source) for source in sources ]
    counts = [ len(components) for (components, _) in graphs ]
    max_components = max(counts)
    data_count = sum(counts)
    A = np.zeros((data_count, max_components, max_components))
    X = np.zeros((data_count, max_components, len(component_types)))
    y = np.zeros((data_count, max_components, len(component_types)))

    start = 0
    for (i, (components, adj)) in enumerate(graphs):
        data_points = len(components)
        end = start + data_points
        encode_masked_netlist((components, adj), A[start:end], X[start:end], y[start:end])
        start = end

    return A, X, y

def encode_masked_netlist(graph, A, X, y):
    (component_list, adj) = graph
    element_types = np.array([ h.get_component_type_index(e) for e in component_list ])

    X[:,np.arange(element_types.size), element_types] = 1
    y[:,np.arange(element_types.size), element_types] = 1
    for idx in range(element_types.size):
        actual_type = element_types[idx]
        X[idx, idx, actual_type] = 0
        X[idx, idx, 0] = 1
        A[idx,:adj.shape[0],:adj.shape[1]] = adj

    return A, X
