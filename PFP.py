import pyspark
from pyspark.sql.functions import split
from pyspark.ml.fpm import FPGrowth
import os

if __name__ == '__main__':
    spark = pyspark.sql.SparkSession.builder\
            .master("local[*]")\
            .appName("FPGrowth")\
            .getOrCreate()
    inFile = "transData"

    # data: DataFrame
    # \s: matches unicode white spaces
    data = spark.read.text(inFile)\
            .select(split("value", "\s+")\
            .alias("items"))

    data.show(truncate=False)
    
    fp = FPGrowth(minSupport=0.2, minConfidence=0.7)
    fpm = fp.fit(data)
    fpm.freqItemsets.show(5)
    fpm.associationRules.show(5)

    #transData = data.map(lambda s : s.trim.split(' '))

    spark.stop()