import networkx as nx
from networkx.algorithms import *
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":

    G = nx.Graph()
    with open(sys.argv[1]) as fp:
        for node, line in enumerate(fp):
            edge = line.strip().split()
            G.add_edge(int(edge[0]), int(edge[1]))

    k = int(sys.argv[2])

    while number_connected_components(G) < k:
        edges = edge_betweenness_centrality(G)

        cut_edge = None
        max_v = -1
        for key, v in edges.items():
            if v > max_v:
                max_v = v
                cut_edge = key

        G.remove_edge(cut_edge[0], cut_edge[1])

    subGs = connected_component_subgraphs(G)
    fw = open('output_partition.txt', 'w')
    for subG in subGs:
        nodes = [str(i) for i in subG.nodes()]
        print(' '.join(nodes))
        fw.write(' '.join(nodes) + '\n')
    fw.close()
