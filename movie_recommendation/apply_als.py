from pyspark.ml.recommendation import ALSModel
import pyspark
from pyspark.sql.types import StructField, StructType, StringType, ArrayType, DoubleType, LongType, FloatType, BooleanType


def apply_model(sc, path_eval, output_path = "../output/evaluation_rating.csv"):
    """
    Apply the model previously built to the evaluation file
    :param path_eval: csv file path
    :return: None, save the file in an output folder
    """
    features_rating = StructType([
            StructField("userId", FloatType(), True),
            StructField("movieId", FloatType(), True),
    ])

    model_als = ALSModel.load("../als_model")

    df = pyspark.SQLContext(sc).read.format("csv").schema(features_rating).option("header", True).load(path_eval)

    predictions = model_als.transform(df)

    predictions.coalesce(1)\
       .write.format("com.databricks.spark.csv")\
       .option("header", "true")\
       .save(output_path)