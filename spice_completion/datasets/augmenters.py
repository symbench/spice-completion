import random
# TODO: what types of augmentation would we like?
# - omitting a node (adding graph label)
#   - How to add graph label in networkx?
# - remove a node or edge
#   - [ ] are they all completely connected in the github dataset?

def remove_nodes(nodes, edges, remove_ids):
    assert len(remove_ids) < len(nodes), f'Cannot remove {len(remove_ids)} node(s) when only {len(nodes)} exist'
    nodes = [ node for (i, node) in enumerate(nodes) if i not in remove_ids ]
    is_removed_edge = lambda edge: len(remove_ids.intersection(set([edge[0], edge[1]]))) > 0
    edges = [ edge for edge in edges if is_removed_edge(edge) ]
    return nodes, edges

def remove_random_node(nodes, edges, k=1):
    assert k < len(nodes), f'Cannot remove {k} nodes when only {len(nodes)} exist'
    remove_ids = set(random.sample(range(len(nodes)), k))
    return remove_nodes(nodes, edges, remove_ids)

def remove_random_edge(nodes, edges, k=1):
    assert k < len(edges), f'Cannot remove {k} edges when only {len(edges)} exist'
    remove_ids = set(random.sample(range(len(edges)), k))
    edges = [ edge for (i, edge) in enumerate(edges) if i not in remove_ids ]
    return nodes, edges
