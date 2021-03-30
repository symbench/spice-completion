import spice_completion.datasets as datasets
from spice_completion.datasets import helpers as h
from spice_completion.datasets import OmittedDataset
import numpy as np
import tensorflow as tf

expected_label_count = {1: 4, 2: 2, 3:3, 4: 1, 5: 1, 6: 9}
filename = 'LT1001_TA05.net'

def test_load_graphs():
    dataset = datasets.omitted([filename], resample=False)
    assert len(dataset) == 20, f'Expected to find 20 graphs. Found {len(dataset)}'

def test_graph_label_size():
    dataset = datasets.omitted([filename], resample=False)
    for graph in dataset:
        assert graph.y.size == len(datasets.helpers.component_types)

def test_component_removed():
    dataset = datasets.omitted([filename], resample=False)
    contents = next(h.valid_netlist_sources([filename]))
    (components, adj) = h.netlist_as_graph(contents)
    for graph in dataset:
        assert graph.x.shape[0] == len(components) - 1

def test_no_nans():
    dataset = datasets.omitted([filename])
    for graph in dataset:
        print(tf.math.is_nan(graph.x))

def test_load_graph_adj():
    source = open(filename, 'rb').read().decode('utf-8', 'ignore')
    (components, adj) = h.netlist_as_graph(source)
    for omitted_idx in range(len(components)):
        graph = OmittedDataset.load_graph(components, adj, omitted_idx)
        omitted_degree = len(adj[omitted_idx].nonzero()[0]) + len(adj.transpose()[omitted_idx].nonzero()[0])
        assert len(adj[omitted_idx].nonzero()[0]) == len(adj.transpose()[omitted_idx].nonzero()[0]), f"Found edge without bidirectional edge: {omitted_idx}"
        assert len(adj.nonzero()[0]) - omitted_degree == len(graph.a.nonzero()[0])

        # Check that all edges existed in original
        all_edges = np.array(adj.nonzero()).transpose().tolist()
        edges = np.array(graph.a.nonzero()).transpose().tolist()
        for edge_pair in edges:
            original_ids = [ e + 1 if e >= omitted_idx else e for e in edge_pair ]
            assert original_ids in all_edges

def test_load_graph_adj_shuffle():
    source = open(filename, 'rb').read().decode('utf-8', 'ignore')
    (components, adj) = h.netlist_as_graph(source)
    for omitted_idx in range(len(components)):
        graph = OmittedDataset.load_graph(components, adj, omitted_idx, True)
        omitted_degree = len(adj[omitted_idx].nonzero()[0]) + len(adj.transpose()[omitted_idx].nonzero()[0])
        assert len(adj[omitted_idx].nonzero()[0]) == len(adj.transpose()[omitted_idx].nonzero()[0]), f"Found edge without bidirectional edge: {omitted_idx}"
        assert len(adj.nonzero()[0]) - omitted_degree == len(graph.a.nonzero()[0])

def test_to_networkx():
    dataset = datasets.omitted([filename], shuffle=False)
    nx_data = dataset.to_networkx()

    assert len(dataset) == len(nx_data), f'Expected to find 20 graphs. Found {len(nx_data)}'
    for (i, nx_graph) in enumerate(nx_data):
        # Check node counts
        assert len(nx_graph.nodes) == 19, f'Expected 19 nodes. Found {nx_graph.node_feature.shape[0]}'

        # Check edge counts
        graph = dataset[i]
        expected_edge_count = graph.n_edges/2  # Converting to bidirectional graph
        edge_count = len(nx_graph.edges)
        assert edge_count == expected_edge_count, f'Expected {expected_edge_count} edges. Found {edge_count}'

    # TODO: should I make sure they all differ by a single node?
    # TODO: check the graph labels?

def test_to_networkx_shuffle():
    dataset = datasets.omitted([filename])
    nx_data = dataset.to_networkx()

    assert len(dataset) == len(nx_data), f'Expected to find 20 graphs. Found {len(nx_data)}'
    for (i, nx_graph) in enumerate(nx_data):
        # Check node counts
        assert len(nx_graph.nodes) == 19, f'Expected 19 nodes. Found {nx_graph.node_feature.shape[0]}'

        # Check edge counts
        graph = dataset[i]
        expected_edge_count = graph.n_edges/2  # Converting to bidirectional graph
        edge_count = len(nx_graph.edges)
        assert edge_count == expected_edge_count, f'Expected {expected_edge_count} edges. Found {edge_count}'

    # TODO: should I make sure they all differ by a single node?
    # TODO: check the graph labels?

def test_to_deepsnap():
    dataset = datasets.omitted([filename])
    ds_data = dataset.to_deepsnap()
    print('ds_data', len(ds_data))

    assert len(dataset) == len(ds_data), f'Expected to find 20 graphs. Found {len(ds_data)}'
    for (i, ds_graph) in enumerate(ds_data):
        # Check node counts
        assert ds_graph.node_feature.shape[0] == 19, f'Expected 19 nodes. Found {ds_graph.node_feature.shape[0]}'

        # Check edge counts
        graph = dataset[i]
        expected_edge_count = graph.n_edges
        edge_count = ds_graph.edge_index.shape[1]
        assert edge_count == expected_edge_count, f'Expected {expected_edge_count} edges. Found {edge_count}'

    # TODO: should I make sure they all differ by a single node?
    # TODO: check the graph labels?
