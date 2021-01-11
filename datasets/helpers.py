import numpy as np
import networkx as nx
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice import BasicElement
from PySpice.Spice.Netlist import Node
import os
np.random.seed(1235)

component_types = [
    'unknown',
    BasicElement.Resistor,
    BasicElement.BehavioralCapacitor,
    BasicElement.VoltageSource,
    BasicElement.Mosfet,
    BasicElement.SubCircuitElement,
    Node,
    BasicElement.Diode,
    BasicElement.BehavioralInductor,
    BasicElement.CurrentSource,
    BasicElement.VoltageControlledCurrentSource,
    BasicElement.VoltageControlledVoltageSource,
    BasicElement.Capacitor,
    BasicElement.CoupledInductor,
    BasicElement.JunctionFieldEffectTransistor,
    BasicElement.BipolarJunctionTransistor,
    BasicElement.XSpiceElement,
    BasicElement.BehavioralSource,
    BasicElement.SemiconductorResistor,
    BasicElement.Mesfet,
    BasicElement.Inductor,
]

subcircuit_types = {}
script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_dir, 'subcircuit-types.txt'), 'r') as f:
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

def valid_netlist_sources(files):
    netlists = ( open(f, 'rb').read().decode('utf-8', 'ignore') for f in files )
    return ( text for text in netlists if is_valid_netlist(text) )

def is_valid_netlist(textfile, name=None):
    try:
        parser = SpiceParser(source=textfile)
        circuit = parser.build_circuit()
        return True
    except:
        if name:
            print(f'invalid spice file: {name}', file=sys.stderr)
        return False

def component_index_name(idx):
    component = component_types[idx]
    if type(component) is not str:
        return component.__name__
    return component
