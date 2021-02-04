import spice_completion.datasets as datasets
import numpy as np

expected_label_count = {1: 4, 2: 2, 3:3, 4: 1, 5: 1, 6: 9}

def test_load_graphs():
    dataset = datasets.omitted_with_actions(['LT1001_TA05.net'], resample=False)
    assert len(dataset) == 20, f'Expected to find 20 graphs. Found {len(dataset)}'

def get_label_type(dataset, graph):
    node_index = np.argmax(graph.y)
    node_embedding = dataset.unnormalize(graph.x)[node_index]
    node_type_index = np.argmax(node_embedding > 0.999999)
    return node_type_index

def test_correct_labels():
    dataset = datasets.omitted_with_actions(['LT1001_TA05.net'], resample=False)
    labels = [ get_label_type(dataset, graph) for graph in dataset ]
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    for (label, count) in label_dist.items():
        expected = expected_label_count[label]
        assert count == expected, f'Expected {expected} #{label} labels but found {count}'

def test_get_node_types():
    dataset = datasets.omitted_with_actions(['LT1001_TA05.net'], resample=False)
    graph = dataset[0]
    types = dataset.get_node_types(graph.x)
    unique, counts = np.unique(types, return_counts=True)
    single_graph_count = dict(zip(unique, counts))

    found_omitted_type = False
    for (label, component_count) in expected_label_count.items():
        count = single_graph_count[label]
        if count != (component_count + 1):
            if count == component_count and not found_omitted_type:
                found_omitted_type = True
                continue
            raise Exception(f'Expected {expected+1} #{label} labels but found {count}')

def test_correct_labels_resampled():
    dataset = datasets.omitted_with_actions(['LT1001_TA05.net'], resample=True)
    labels = [ get_label_type(dataset, graph) for graph in dataset ]
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    for (label, count) in label_dist.items():
        max_count = expected_label_count[label]
        assert count <= max_count, f'Expected <= {max_count} #{label} labels but found {count}'

def test_only_one_prototype_per_type():
    dataset = datasets.omitted_with_actions(['LT1001_TA05.net'], resample=False, normalize=False)
    for graph in dataset:
        prototype_idx = (graph.x[:,-1] == 1).nonzero()[0]
        prototypes = graph.x[prototype_idx]
        type_counts = np.sum(prototypes, axis=0)[0:-1]
        assert (type_counts > 1).nonzero()[0].size == 0

def test_targets_should_be_different_types():
    dataset = datasets.omitted_with_actions(['LT1001_TA05.net'], resample=False, normalize=False)
    for graph in dataset:
        prototype_idx = (graph.x[:,-1] == 1).nonzero()[0]
        valid_target_idx = (graph.y > -1).nonzero()[0]
        for (prototype, target) in zip(prototype_idx, valid_target_idx):
            assert prototype == target

def test_components_should_be_connected():
    dataset = datasets.omitted_with_actions(['LT1001_TA05.net'], resample=False, normalize=False, shuffle=False)
    for (i, graph) in enumerate(dataset):
        adj = graph.a.toarray()
        edge_counts = np.sum(adj, axis=0)
        for (edge_index, edge_count) in enumerate(edge_counts):
            assert edge_count > 0, f'Found disconnected node: {edge_index}'

def test_no_dupe_types_in_proto():
    dataset = datasets.omitted_with_actions(['LT1001_TA05.net'], resample=False, normalize=False, shuffle=False)
    for (i, graph) in enumerate(dataset):
        targets = graph.y
        prototypes = graph.x[(targets > -1).nonzero()[0]]
        node_types = dataset.get_node_types(prototypes)
        types, counts = np.unique(node_types, return_counts=True)
        type_dict = dict(zip(types, counts))
        for (node_type, count) in type_dict.items():
            assert count == 1, f'Found {count} nodes of type {node_type}'

def test_no_dupe_types_shuffled():
    dataset = datasets.omitted_with_actions(['LT1001_TA05.net'], resample=False, normalize=False)
    for (i, graph) in enumerate(dataset):
        targets = graph.y
        prototypes = graph.x[(targets > -1).nonzero()[0]]
        node_types = dataset.get_node_types(prototypes)
        print(node_types)
        types, counts = np.unique(node_types, return_counts=True)
        type_dict = dict(zip(types, counts))
        for (node_type, count) in type_dict.items():
            assert count == 1, f'Found {count} nodes of type {node_type}'
