import time
import os

import pyspark
from pyspark.sql.functions import mean, col
from pyspark.sql.types import StructField, StructType, StringType, ArrayType, DoubleType, LongType, FloatType, BooleanType

from movie_recommendation.build_als_model import build_model
from movie_recommendation.apply_als import apply_model

if __name__ == "__main__":

    # Build and apply the model
    path_rating ="data/ratings.csv"
    path_eval = "data/evaluation_ratings.csv"
    path_movie = "data/"

    if not os.path.exists("../output/"):
        os.mkdir("../output/")
    output_path = "../output/evaluation_rating.csv"

    start = time.time()

    # Define spark context
    conf = (pyspark.SparkConf().setAppName('test').set("spark.executor.memory", "2g").setMaster("local[4]"))
    sc = pyspark.SparkContext(conf=conf)

    # structure of the rating dataframe
    features_rating = StructType([
            StructField("userId", FloatType(), True),
            StructField("movieId", FloatType(), True),
            StructField("rating", FloatType(), True),
            StructField("timestamp", StringType(), True)
    ])

    # load the rating detaframe
    df_rating = pyspark.SQLContext(sc).read.format("csv").schema(features_rating).option("header", True).load(path_rating)
    df_rating = df_rating.select("userId", "movieId", "rating")

    # build the collaborative filtering model
    build_model(df_rating, save = True, output_model_path = "output/model", output_param_path = "output/params")

    # apply the model to the evaluation data
    apply_model(sc, path_eval, output_path)

    print("time :", time.time() - start)



