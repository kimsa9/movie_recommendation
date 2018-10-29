import pyspark

from pyspark.sql.types import StructField, StructType, StringType, ArrayType, DoubleType, LongType, FloatType, BooleanType
from pyspark.sql.functions import expr, to_timestamp, col, input_file_name, when, trim, struct, max, last, desc, from_json, udf, first
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg import Vectors

def compute_similarity(df):
    """
    Compute cosine
    :param df:dataframe of rating by user for movies
    :return:
    """

    # df = df.filter(df.movieId.isin([91542.0, 1.0, 5.0, 90.0, 2541.0, 1246.0, 1552.0, 4084.0, 5679.0]))

    df = df.groupBy("userId").pivot("movieId").agg(first(col('rating')).cast("double"))

    mat = IndexedRowMatrix(df.rdd.map(lambda row: IndexedRow(row[0], Vectors.dense(row[1:]))))

    cs = mat.columnSimilarities()

    path = "test"

    cs.entries.toDF().write.parquet(path)

    cs.entries.toDF().coalesce(1)\
       .write.format("com.databricks.spark.csv")\
       .option("header", "true")\
       .save("testtest.csv")


if __name__ == "__main__":
    path_rating = "../data/ratings.csv"

    conf = (pyspark.SparkConf().setAppName('test').set("spark.executor.memory", "2g").setMaster("local[4]")).set(
        'spark.sql.pivotMaxValues', u'50000')
    sc = pyspark.SparkContext(conf=conf)

    features_rating = StructType([
        StructField("userId", FloatType(), True),
        StructField("movieId", FloatType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", StringType(), True)
    ])

    df = pyspark.SQLContext(sc).read.format("csv").schema(features_rating).option("header", True).load(path_rating)
    df = df.select("userId", "movieId", "rating")

    compute_similarity(df)