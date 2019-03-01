from __future__ import print_function

import sys
from operator import add
from bipartite import *

from pyspark.sql import SparkSession

if __name__ == "__main__":
    def build():
        fp = open(sys.argv[1], 'r')
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
        fp.close()
        return baskets, candi_sets

    def apriori(lines):
        lines = list(lines)
        baskets = {}
        candi_sets = set()
        for line in lines:
            basket = line[0]
            for i in range(0, len(line[1])):
                candi_sets.add(line[1][i])
                if baskets.has_key(basket):
                    items = baskets[basket]
                else:
                    items = []
                items.append(line[1][i])
                baskets[basket] = items

        size = 1
        while size <= s and candi_sets:
            if size > 1:
                candi_sets = permute_candi_sets(freq_set, size)
            freq_set = find_freq_set(thre, baskets, candi_sets)
            size += 1

        return candi_sets

    def find_baskets(left_set):
        cor_baskets = []
        for basket in baskets:
            items = baskets[basket]
            if set(left_set) <= set(items):
                cor_baskets.append(basket)
        return left_set, list(combinations(cor_baskets, t))


    def pass2(left_set):
        return find_baskets(left_set)

    spark = SparkSession \
        .builder \
        .appName("BipartiteSon") \
        .getOrCreate()
    sc = spark.sparkContext

    num_chunk = 2

    lines = sc.textFile(sys.argv[1]).map(lambda l: l.split(',')).map(lambda x: [x[1], x[0]]).groupByKey().map(lambda x: (x[0], list(x[1])))
    lines = lines.coalesce(num_chunk)

    s = int(sys.argv[2])
    t = int(sys.argv[3])
    thre = t / num_chunk

    baskets, candi_sets = build()

    pass1_res = lines.mapPartitions(apriori)


    local_candi_sets = pass1_res
    candis = local_candi_sets.map(pass2).filter(lambda x: len(x) > 0).collect()

    graphs = set()
    for candi in candis:
        if candi[1]:
            for right_set in candi[1]:
                graph = [tuple(candi[0]), tuple(right_set)]
                graphs.add(tuple(graph))

    if not graphs:
        print("no such graph")

    for graph in graphs:
        left_string = '{' + ','.join(graph[0]) + '}'
        right_string = '{' + ','.join(graph[1]) + '}'
        print(left_string + right_string)


    sc.stop()
