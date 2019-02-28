# usage: python bipartite.py <graph.txt> <s> <t>
# <s> and <t> represent the number of nodes in the left and right

import sys
from itertools import combinations


def find_freq_set(thre, baskets, candi_sets):
    count = {}

    for basket in baskets:
        items = baskets[basket]
        for candi in candi_sets:
            if set(candi) <= set(items): # items include candi
                candi = tuple(candi)
                if count.has_key(candi):
                    num = count[candi]
                else:
                    num = 0
                count[candi] = num + 1

    freq_set = []
    for key in count:
        if count[key] >= thre:
            freq_set.append(key)

    return freq_set


def filt_permutation(permutations, freq_set, size):
    candi_sets = []
    for p in permutations:
        subsets = list(combinations(p, size - 1))
        flag = True
        for subset in subsets:
            if subset not in freq_set:
                flag = False
                break
        if flag:
            candi_sets.append(p)
    return candi_sets


def permute_candi_sets(freq_set, size):
    ones_list = []
    for i in range(0, len(freq_set)):
        ones_list += freq_set[i]
    ones_set = set(ones_list)

    permutations = list(combinations(ones_set, size))
    candi_sets = filt_permutation(permutations, freq_set, size)

    return candi_sets


def find_baskets(left_set, baskets, size):
    cor_baskets = []
    for basket in baskets:
        items = baskets[basket]
        if set(left_set) <= set(items):
            cor_baskets.append(basket)
    return list(combinations(cor_baskets, size))


if __name__ == "__main__":
    fp = open(sys.argv[1], 'r')
    s = int(sys.argv[2])
    t = int(sys.argv[3])

    thre = t
    baskets = {}
    candi_sets = set()
    for i, line in enumerate(fp):
        item, basket = line.strip().split(',')
        if baskets.has_key(basket):
            items = baskets[basket]
        else:
            items = []
        items.append(item)
        baskets[basket] = items
        candi_sets.add(item)
    # print(baskets)
    # print(candi_sets)

    fp.close()

    size = 1
    # TODO: s = 1
    while size <= s and candi_sets:
        if size > 1:
            candi_sets = permute_candi_sets(freq_set, size)
        freq_set = find_freq_set(thre, baskets, candi_sets)
        size += 1
        # print(freq_set)
        # print(candi_sets)

    # use candi_sets of size s to find corresponding baskets
    graphs = set()
    for left_set in candi_sets:
        right_sets = find_baskets(left_set, baskets, t)
        # print(right_sets)
        for right_set in right_sets:
            graph = [tuple(left_set), tuple(right_set)]
            # print(graph)
            graphs.add(tuple(graph))

    if not graphs:
        print("no such graph")

    for graph in graphs:
        # left_set = left_set.sort()
        left_string = '{' + ','.join(graph[0]) + '}'
        # right_set = right_set.sort()
        right_string = '{' + ','.join(graph[1]) + '}'
        print(left_string + right_string)
