import numpy as np
import networkx as nx
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice import BasicElement
from PySpice.Spice.Netlist import Pin

component_types = [
    BasicElement.Resistor,
    BasicElement.BehavioralCapacitor,
    BasicElement.VoltageSource,
    BasicElement.Mosfet,
    BasicElement.SubCircuitElement,
    Pin,
]

def load_netlist(textfile):
    parser = SpiceParser(source=textfile)
    circuit = parser.build_circuit()
    component_list = []

    adj = {}

    for element in circuit.elements:
        if element not in component_list:
            component_list.append(element)
        for pin in element.pins:
            if pin not in component_list:
                component_list.append(pin)

        element_id = component_list.index(element)
        if element_id not in adj:
            adj[element_id] = []

        pin_ids = [component_list.index(pin) for pin in element.pins]
        adj[element_id].extend(pin_ids)

        for pin_id in pin_ids:
            if pin_id not in adj:
                adj[pin_id] = []
            adj[pin_id].append(element_id)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))

    element_types = np.array([ component_types.index(type(e)) for e in component_list ])
    X = np.zeros((element_types.size, len(component_types)))
    X[np.arange(element_types.size), element_types] = 1

    labels = X  # TODO: Not sure if this is what I want...

    # For now, just mask out the first non-pin (second for validation)
    # TODO: Make the mask more intentional
    mask = np.ones(element_types.size)
    nonPinIndices = [i for (i, element) in enumerate(component_list) if type(element) is not Pin]
    mask_index = nonPinIndices[0]
    mask[mask_index] = 0
    train_mask = np.array(mask, dtype=np.bool)

    mask = np.ones(element_types.size)
    mask_index = nonPinIndices[1]
    mask[mask_index] = 0
    val_mask = np.array(mask, dtype=np.bool)

    mask = np.ones(element_types.size)
    mask_index = nonPinIndices[2]
    mask[mask_index] = 0
    test_mask = np.array(mask, dtype=np.bool)

    return adj, X, labels, train_mask, val_mask, test_mask

# TODO: Load an entire dataset

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'rb') as f:
        print(load_netlist(f.read().decode('utf-8', 'ignore')))
