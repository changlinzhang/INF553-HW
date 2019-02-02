from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession


if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("PythonWordCount")\
        .getOrCreate()

    city_lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    country_lines = spark.read.text(sys.argv[2]).rdd.map(lambda r: r[0])

    cities = city_lines.map(lambda x: x.split('\t')) \
                  .filter(lambda x: int(x[4]) >= 1000000) \
                  .map(lambda x: (x[2], x[0])) \
                  .groupByKey() \
                  .filter(lambda x: len(x[1]) >= 3) \
                  .map(lambda x: (x[0], len(x[1])))
                  
    countries = country_lines.map(lambda x: x.split('\t')) \
        .map(lambda x: (x[0], x[1]))

    table = cities.join(countries)
    output = table.collect()

    output_file = open(sys.argv[3], 'w')
    for record in output:
        output_file.write("%s %s\n" % (record[1][1], record[1][0]))
    output_file.close()

    spark.stop()
