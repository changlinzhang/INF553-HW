import sys
from operator import add

from pyspark.sql import SparkSession


def hashi(movie_user, hi):
    sign = (5*movie_user[0]+13*hi)%100
    return [(user, sign) for user in movie_user[1]]


def minHash(x):
    x = list(x)
    y = zip(x[0], x[1], x[2], x[3])
    # candi_pairs =


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("LSH") \
        .getOrCreate()
    sc = spark.sparkContext

    num_chunk = 2  # 4
    lines = sc.textFile(sys.argv[1], num_chunk)

    user_movie = lines.mapPartitions(lambda x: x.split(',')) \
        .map(lambda x: (int(x[0][1:]), x[1:])) \

    movie_user = user_movie.flatMap(lambda x: [(int(movie), x[0]) for movie in x[1]]).groupByKey()
    # .mapValues(lambda values: [v for v in values ]) ???

    signs = movie_user.flatMap(lambda x: hashi(x, i) for i in range(0, 20)).reduceByKey(lambda x1, x2: min(x1, x2))

    bands = sc.parallelize(signs, 5)

    # candi_pairs = bands.mapPartitions(lambda x: [minHash(x)])






    # output_file = open(sys.argv[2], 'w')
    # for record in output:
    #     output_file.write("%s %s\n" % (record[1][1], record[1][0]))
    # output_file.close()

    spark.stop()
