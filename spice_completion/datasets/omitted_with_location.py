"""
    This creates a dataset where X is the graph where one component has been removed.
    The y values are the same graph where the component attached to the removed component's
    first pin is labeled with the removed component's type. The remaining components are labeled "None"
"""
import numpy as np
from . import helpers as h
component_types = h.component_types

def netlist_as_graph(textfile):
    parser = SpiceParser(source=textfile)
    circuit = parser.build_circuit()
    component_list = []
    adj = {}

    for element in circuit.elements:
        if element not in component_list:
            component_list.append(element)

        nodes = [ pin.node for pin in element.pins ]
        for node in nodes:
            if node not in component_list:
                component_list.append(node)

        element_id = component_list.index(element)
        if element_id not in adj:
            adj[element_id] = []

        node_ids = [component_list.index(node) for node in nodes]
        adj[element_id].extend(node_ids)

        for node_id in node_ids:
            if node_id not in adj:
                adj[node_id] = []
            adj[node_id].append(element_id)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj)).toarray()
    return component_list, adj

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
        encode_omitted_netlist((components, adj), A[start:end], X[start:end], y[start:end])
        start = end

    return A, X, y

def encode_omitted_netlist(graph, A, X, y):
    (component_list, adj) = graph
    element_types = np.array([ h.get_component_type_index(e) for e in component_list ])

    X[:,np.arange(element_types.size), element_types] = 1
    y[:, np.arange(element_types.size), 0] = 1
    for idx in range(element_types.size):
        actual_type = element_types[idx]
        # clear the node representation
        X[idx, idx, actual_type] = 0
        # TODO: get the first connection and set it to the node type
        # I don't have this info anymore...
        X[idx, idx, actual_type] = 0
        # disconnect the node
        A[idx,:adj.shape[0],:adj.shape[1]] = adj
        A[idx,:,idx] = 0
        A[idx,idx,:] = 0

    return A, X

