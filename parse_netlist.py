import numpy as np
import networkx as nx
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice import BasicElement
from PySpice.Spice.Netlist import Node

component_types = [
    'unknown',
    BasicElement.Resistor,
    BasicElement.BehavioralCapacitor,
    BasicElement.VoltageSource,
    BasicElement.Mosfet,
    BasicElement.SubCircuitElement,
    Node,
]

subcircuit_types = {}
with open('subcircuit-types.txt', 'r') as f:
    subcircuit_label_pairs = (line.split(' ') for line in f if len(line.split(' ')) == 2)
    for (subcircuit, label) in subcircuit_label_pairs:
        label = label.strip()
        subcircuit_types[subcircuit] = label
        if label not in component_types:
            component_types.append(label)

def get_component_type_index(element):
    element_type = type(element)
    if element_type is BasicElement.SubCircuitElement:
        element_type = subcircuit_types.get(element.subcircuit_name, element_type)

    return component_types.index(element_type)

np.random.seed(1234)
def load_netlist(textfile):
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

    element_types = np.array([ get_component_type_index(e) for e in component_list ])

    X = np.zeros((element_types.size, element_types.size, len(component_types)))
    X[:,np.arange(element_types.size), element_types] = 1
    A = np.zeros((X.size, adj.shape[0], adj.shape[1]))
    y = np.copy(X)
    for idx in range(element_types.size):
        actual_type = element_types[idx]
        X[idx, idx, actual_type] = 0
        X[idx, idx, 0] = 1
        A[idx,:,:] = adj

    return A, X, y

# TODO: Load an entire dataset

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'rb') as f:
        print(load_netlist(f.read().decode('utf-8', 'ignore')))
