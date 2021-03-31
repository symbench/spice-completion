import spice_completion.datasets as datasets
import spice_completion.datasets.helpers as h
from spice_completion.datasets import PrototypeLinkDataset
import numpy as np

expected_label_count = {1: 4, 2: 2, 3:3, 4: 1, 5: 1, 6: 9}
filename = 'LT1001_TA05.net'

def test_load_graphs():
    dataset = PrototypeLinkDataset(['LT1001_TA05.net'], resample=False)
    assert len(dataset) == 20, f'Expected to find 20 graphs. Found {len(dataset)}'

def test_flag_omitted_node():
    source = open(filename, 'rb').read().decode('utf-8', 'ignore')
    (components, adj) = h.netlist_as_graph(source)
    for omitted_idx in range(len(components)):
        graph = PrototypeLinkDataset.load_graph(components, adj, omitted_idx)
        assert graph.x[omitted_idx][-1] == 1

def test_omitted_node_connected():
    source = open(filename, 'rb').read().decode('utf-8', 'ignore')
    (components, adj) = h.netlist_as_graph(source)
    for omitted_idx in range(len(components)):
        graph = PrototypeLinkDataset.load_graph(components, adj, omitted_idx)
        assert len(graph.a[omitted_idx].nonzero()[0]) > 0

def test_action_nodes_disconnected():
    source = open(filename, 'rb').read().decode('utf-8', 'ignore')
    (components, adj) = h.netlist_as_graph(source)
    for omitted_idx in range(len(components)):
        graph = PrototypeLinkDataset.load_graph(components, adj, omitted_idx)
        assert len(graph.a[len(components)].nonzero()[0]) == 0

def test_to_networkx():
    dataset = PrototypeLinkDataset(['LT1001_TA05.net'], resample=False)
    nx_data = h.to_networkx(dataset)

    assert len(dataset) == len(nx_data), f'Expected to find 20 graphs. Found {len(nx_data)}'
    for (i, nx_graph) in enumerate(nx_data):
        # Check node counts
        expected_node_count = 19 + len(h.component_types)
        assert len(nx_graph.nodes) == expected_node_count, f'Expected {expected_node_counts} nodes. Found {len(nx_graph.nodes)}'

        # Check edge counts
        graph = dataset[i]
        expected_edge_count = graph.n_edges/2  # Converting to bidirectional graph
        edge_count = len(nx_graph.edges)
        assert edge_count == expected_edge_count, f'Expected {expected_edge_count} edges. Found {edge_count}'

def test_to_deepsnap():
    dataset = PrototypeLinkDataset(['LT1001_TA05.net'], resample=False)
    ds_data = h.to_deepsnap(dataset)

    assert len(dataset) == len(ds_data), f'Expected to find 20 graphs. Found {len(ds_data)}'
    for (i, ds_graph) in enumerate(ds_data):
        # Check node counts
        expected_node_count = 19 + len(h.component_types)
        node_count = ds_graph.node_feature.shape[0]
        assert node_count == expected_node_count

        # Check edge counts
        graph = dataset[i]
        expected_edge_count = graph.n_edges
        edge_count = ds_graph.edge_index.shape[1]
        assert edge_count == expected_edge_count, f'Expected {expected_edge_count} edges. Found {edge_count}'

def test_dont_omit_for_inference():
    source = open(filename, 'rb').read().decode('utf-8', 'ignore')
    (components, adj) = h.netlist_as_graph(source)
    dataset = PrototypeLinkDataset(['LT1001_TA05.net'], resample=False, train=False)

    assert len(dataset) == 1
    graph = dataset[0]
    expected_node_count = len(components) + len(h.component_types)

    assert graph.n_nodes == expected_node_count 

def test_inference_proto_count():
    source = open(filename, 'rb').read().decode('utf-8', 'ignore')
    (components, adj) = h.netlist_as_graph(source)
    dataset = PrototypeLinkDataset(['LT1001_TA05.net'], resample=False, train=False, normalize=False)

    graph = dataset[0]

    print(graph.x[:,-1])
    proto_count = len(graph.x[:, -1].nonzero()[0])
    print('proto_count', proto_count)
    assert proto_count == len(h.component_types)
