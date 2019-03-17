import sys
from Queue import PriorityQueue
from itertools import combinations

from pyspark.sql import SparkSession


def hash(x, i): # i, 0-19
    h_value = (5*x + 13*i) % 100
    return h_value


# return candi pairs
def compLSH(input):
    result = []
    permutations = list(combinations(input, 2))
    for pair in permutations:
        u0 = pair[0]
        u1 = pair[1]
        s0 = set(u0[1])
        s1 = set(u1[1])
        if s0.issubset(s1) and s1.issubset(s0):
            candi = tuple([u0[0], u1[0]]) # user1 < user2 because of combinations operation
            candi_reverse = tuple([u1[0], u0[0]])
            result.append(candi)
            result.append(candi_reverse)
    return result


def top5users((k, v), ori_users):
    # top5 users
    ori_users = dict(ori_users.value)
    k_set = set(ori_users[k])
    pq = PriorityQueue()
    for candi in v:
        candi_set = set(ori_users[candi])
        i = k_set & candi_set
        u = k_set | candi_set
        j = len(i)/len(u)
        pq.put((-j, candi))

    num = 5
    top5_users = []
    while num > 0 and not pq.empty():
        top5_users.append(pq.get()[1])
        num -= 1

    film_count = {}
    for user in top5_users:
        film_list = ori_users[user]
        for film in film_list:
            if film_count.has_key(film):
                count = film_count[film]
            else:
                count = 0
            film_count[film] = count + 1

    pq = PriorityQueue()
    for k, v in film_count.items():
        pq.put((-v, k))

    num = 3
    top3_films = []
    while num > 0 and not pq.empty():
        top3_films.append(pq.get()[1])
        num -= 1

    top3_films.sort()
    return top3_films


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("LSH") \
        .getOrCreate()
    sc = spark.sparkContext

    num_chunk = 2  # 4
    lines = sc.textFile(sys.argv[1], num_chunk)
    # lines = sc.textFile("input2.txt", num_chunk)

    # Hint: While computing signatures, you can divide users(represented as a set of movies)
    # into partitions and compute signatures for the partitions in parallel.
    # movie_user = lines.map(lambda line: line.split(","))\
    #             .flatMap(lambda x: [(x[i], x[0]) for i in range(1, len(x))])\
    #             .groupByKey().mapValues(list)
    # print movie_user.collect()

    # preprocess
    lines = lines.map(lambda line: line.split(","))\
                .map(lambda x: (x[0], [] if len(x) == 1 else x[1:]))\
                .mapValues(lambda values: [int(v) for v in values])
    #print lines.collect()

    # compute signatures
    signs = lines.mapValues(lambda values: [min([hash(v, i) for v in values]) for i in range(0, 20)])
    # print signs.collect()

    # bands
    band_size = 5
    band_num = 4
    bands = signs.mapValues(lambda values: [values[i:i+band_size] for i in range(0, 20, 5)])
    # print bands.collect()


    # Hint: While computing LSH, you could take the band as key( or part of key), the user ID
    # as value, and then find the candidate pairs / users in parallel.

    # should here be repartitoned to 1?
    # pair: join or permutations
    bands = bands.flatMap(lambda (k, v): [(band_i, (k, v[band_i])) for band_i in range(0, band_num)])\
            .groupByKey().mapValues(list)
    # inner key is users
    # print bands.collect()

    candi_pairs = bands.mapValues(lambda values: compLSH(values))\
                    .flatMap(lambda (k, v): v).distinct()
    # print candi_pairs.collect()

    # Hint: You need to compute the Jaccard similarities of similar pairs identified by LSH
    # based on which you find the top - 5 users and top - 3 movies.
    top_users = candi_pairs.groupByKey().mapValues(list)
    # print top_users.collect()

    ori_users = sc.broadcast(lines.collect())
    recoms = top_users.map(lambda (k, v): (k, top5users((k, v), ori_users)))
    recoms = recoms.collect()
    recoms.sort(key=lambda k: int(k[0][1:]))
    # print recoms

    f = open(sys.argv[2], 'w')
    # f = open("output2.txt", 'w')
    for k,v in recoms:
        v = [str(_) for _ in v]
        out_line = k + ',' + ','.join(v)
        print(out_line)
        f.write("%s\n" % out_line)
    f.close()

    spark.stop()
