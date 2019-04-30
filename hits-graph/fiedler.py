import networkx as nx
import numpy as np
import numpy.linalg
from networkx.algorithms import *
import sys
import matplotlib.pyplot as plt


def divide(G, num_nodes):
    # D = np.zeros((num_nodes, num_nodes))
    # for key, value in G.degree():
    #     D[key-1][key-1] = value
    # # print(D)
    # A = np.zeros((num_nodes, num_nodes))
    # for edge in G.edges():
    #     A[edge[0] - 1][edge[1] - 1] = 1
    #     A[edge[1] - 1][edge[0] - 1] = 1
    # # print(A)
    # L = D - A
    # L = L[~np.all(L == 0, axis=0)]
    # idx = np.argwhere(np.all(L[..., :] == 0, axis=0))
    # L = np.delete(L, idx, axis=1)
    # print(L)
    #
    # L = [[int(element) for element in row] for row in L]
    # print(L)
    #
    # lambs, vectors = np.linalg.eig(L)
    #
    # print(lambs)
    # print(vectors)
    #
    # min = sys.float_info.max
    # min_i = -1
    # for i in range(len(lambs)):
    #     lamb = lambs[i]
    #     if lamb < min:
    #         min = lamb
    #         min_i = i
    # second_min = sys.float_info.max
    # second_i = -1
    # for i in range(len(lambs)):
    #     lamb = lambs[i]
    #     if lamb > min and lamb < second_min:
    #         second_min = lamb
    #         second_i = i
    # v2 = vectors[:, second_i]
    # print(v2)

    v2 = nx.fiedler_vector(G, normalized=True, seed=0)
    # print(v2)
    list_pos = []
    list_neg = []
    node_list = sorted(G.nodes())
    for i in range(len(v2)):
        if v2[i] > 0:
            list_pos.append(node_list[i])
        else:
            list_neg.append(node_list[i])
    G1 = G.subgraph(list_pos).copy()
    G2 = G.subgraph(list_neg).copy()
    # G1 = G.subgraph(list_pos)
    # G2 = G.subgraph(list_neg)
    return G1, G2


if __name__ == "__main__":

    ori_G = nx.Graph()
    with open(sys.argv[1]) as fp:
        for node, line in enumerate(fp):
            edge = line.strip().split()
            ori_G.add_edge(int(edge[0]), int(edge[1]))

    G_set = set()
    G_set.add(ori_G)
    k = int(sys.argv[2])

    num_nodes = len(ori_G)

    while len(G_set) < k:
        max = -1
        max_G = None
        for G in G_set:
            if len(G) > max:
                max = len(G)
                max_G = G
        # print(len(max_G))
        G_set.remove(max_G)
        G1, G2 = divide(max_G, num_nodes)
        G_set.add(G1)
        G_set.add(G2)

    fw = open('output_fiedler.txt', 'w')
    G_list = list(G_set)
    # G_list = sorted(list(G_set), key=lambda x: x.nodes())
    for G in G_list:
        nodes = [str(i) for i in G.nodes()]
        print(' '.join(nodes))
        fw.write(' '.join(nodes) + '\n')
    fw.close()
