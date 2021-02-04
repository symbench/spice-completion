import datasets
from datasets import helpers as h
import numpy as np

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
