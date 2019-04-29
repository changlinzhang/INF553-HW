import networkx as nx
import numpy as np
import numpy.linalg
from networkx.algorithms import *
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":

    G = nx.Graph()
    with open(sys.argv[1]) as fp:
        for node, line in enumerate(fp):
            edge = line.strip().split()
            G.add_edge(int(edge[0]), int(edge[1]))

    G_list = []
    k = int(sys.argv[2])

    num_nodes = len(G)
    D = np.zeros((num_nodes, num_nodes))
    for key, value in G.degree():
        D[key-1][key-1] = value
    # print(D)
    A = np.zeros((num_nodes, num_nodes))
    for edge in G.edges():
        A[edge[0] - 1][edge[1] - 1] = 1
        A[edge[1] - 1][edge[0] - 1] = 1
    # print(A)
    L = D - A
    # print(L)
    
    lambs, vectors = np.linalg.eig(L)

    min = sys.float_info.max
    min_i = -1
    for i in range(len(lambs)):
        lamb = lambs[i]
        if lamb > 0 and lamb < min:
            min = lamb
            min_i = i
    v2 = vectors[:,min_i]
    print(v2)
    list_pos = []
    list_neg = []
    for i in range(1, len(v2)+1):
        if v2[i-1] > 0:
            list_pos.append(i)
        else:
            list_neg.append(i)
    G1 = G.subgraph(list_pos).copy()
    G2 = G.subgraph(list_neg).copy()

    # nx.draw(G1)
    # plt.savefig("b1.png")
    # plt.show()
    # nx.draw(G2)
    # plt.savefig("b2.png")
    # plt.show()

    # subGs = connected_component_subgraphs(G)
    # fw = open('output_partition.txt', 'w')
    # for subG in subGs:
    #     nodes = [str(i) for i in subG.nodes()]
    #     print(' '.join(nodes))
    #     fw.write(' '.join(nodes) + '\n')
    # fw.close()
