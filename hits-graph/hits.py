import sys

from pyspark.sql import SparkSession


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
    # m = [[0 for i in range(num_node)] for j in range(num_node)]
    num_node_bc = sc.broadcast(num_node)
    lines = lines.map(lambda line: line.split()) \
        .map(lambda x: [int(x[0]), int(x[1])])
    m = lines.groupByKey()\
        .map(lambda x: (x[0], set(x[1])))\
        .map(lambda x: [x[0], [1 if i in x[1] else 0 for i in range(1, num_node+1)]])
    print(m.collect())
    m_t = lines.map(lambda x: [x[1], x[0]]).groupByKey()\
        .map(lambda x: (x[0], set(x[1])))\
        .map(lambda x: [x[0], [1 if i in x[1] else 0 for i in range(1, num_node+1)]])
    print(m_t.collect())

    h_0 = [1 for i in range(num_node)]
    # for i in range(num_iter):
    #     # compute a and normalize, use m_t
    #
    #     # compute h and normalize, use m


    spark.stop()
