import sys
import numpy as np

from pyspark.sql import SparkSession


def multi(m, v):
    join_tmp = m.join(v)
    join_tmp = join_tmp.map(lambda x: [x[1][0][0], x[1][0][1]*x[1][1]])
    res = join_tmp.reduceByKey(lambda U, x: U+x)
    # print(res.collect())
    return res


def normalize(v):
    max_i = v.map(lambda x: x[1]).reduce(lambda U, x: max(U, x))
    # print(max_i)
    v = v.map(lambda x: [x[0], 1.0*x[1]/max_i])
    # print(v.collect())
    return v


def print_v(v, num_node, f):
    vector = [0 for i in range(num_node)]
    for element in v:
        vector[element[0]-1] = element[1]
    for i in range(1, num_node+1):
        print("\t\t%d %.2f" % (i, vector[i-1]))
        f.write("\t\t%d %.2f\n" % (i, vector[i - 1]))


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("hits") \
        .getOrCreate()
    sc = spark.sparkContext

    num_chunk = 2  # 4
    lines = sc.textFile(sys.argv[1], num_chunk)
    num_node = int(sys.argv[2])
    num_iter = int(sys.argv[3])

    # Adj matrix, L[i,j]=1 if there is a link from node i to node j
    lines = lines.map(lambda line: line.split())\
        .map(lambda x: [int(x[0]), int(x[1])])
    m = lines.map(lambda x: [x[1], [x[0], 1]])
    # print(m.collect())
    m_t = lines.map(lambda x: [x[0], [x[1], 1]])
    # print(m_t.collect())

    f = open('output.txt', 'w')
    h = np.array([i for i in range(1, num_node+1)]).transpose()
    h = sc.parallelize(h).map(lambda x: [x, 1])
    print(h.collect())
    for i in range(num_iter):
        # compute a and normalize, use m_t
        a = multi(m_t, h)
        a = normalize(a)
        # print(a.collect())
        # compute h and normalize, use m
        h = multi(m, a)
        h = normalize(h)
        print("Iteration: %d" % i)
        f.write("Iteration: %d\n" % i)
        print("\tAuthorities:")
        f.write("\tAuthorities:\n")
        print_v(a.collect(), num_node, f)
        print("\tHubs:")
        f.write("\tHubs:\n")
        print_v(h.collect(), num_node, f)

    spark.stop()
