from deepsnap.dataset import GraphDataset
from spice_completion.datasets.netlist_with_edge_labels import fetch_graphs, mkgraph

# TODO: test connectivity?
# TODO: test loading as DeepSnap
def test_edge_labels_deepsnap():
    filename = 'LT1001_TA05.net'
    graphs = [mkgraph(n, e) for (n, e) in fetch_graphs([filename])]
    dataset = GraphDataset(
        graphs,
        task='link_pred'
    )
    assert dataset[0].edge_label is not None

    # TODO: test graph labels
    # TODO: test node features
    # TODO: test edge labels
