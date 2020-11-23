from . import helpers as h
"""
This encodes a set of netlists for use with gSpan. The format is described in more
detail (with examples) at https://github.com/betterenvi/gSpan/tree/master/graphdata
"""

def load(files):
    sources = h.valid_netlist_sources(files)
    graphs = ( h.netlist_as_graph(source) for source in sources )
    lines = []
    for (i, (nodes, adj)) in enumerate(graphs):
        lines.append(f't # {i}')
        vertices = []
        for (i, n) in enumerate(nodes):
            label = h.get_component_type_index(n)
            lines.append(f'v {i} {label}')
            neighbor_idx = ( j for (j, c) in list(enumerate(adj[i]))[i:] if c == 1 )
            for n_idx in neighbor_idx:
                vertices.append((i, n_idx))
        for (n1, n2) in vertices:
            lines.append(f'e {n1} {n2} 0')

    return '\n'.join(lines)
