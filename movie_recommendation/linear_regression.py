import pyspark
from pyspark.sql.functions import to_timestamp, col, input_file_name, when, trim, struct, max, last, desc, from_json, udf
from pyspark.sql.types import StructField, StructType, StringType, ArrayType, DoubleType, LongType, FloatType, BooleanType
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml import Pipeline

def test_linear_regression():
    """
    Test the linear regression model on the cleaned data
    :param sc: Spark context
    :return: None
    """
    conf = (pyspark.SparkConf().setAppName('test').set("spark.executor.memory", "2g").setMaster("local[2]"))
    sc = pyspark.SparkContext(conf=conf)
    path_rating = "../data/rating_with_movie_data.csv"

    df = pyspark.SQLContext(sc).read.format("csv").option("header", True).load(path_rating)

    columns_to_drop = ['timestamp', 'imdbId', 'tmdbId', 'imdb_id', 'release_date']
    df = df.drop(*columns_to_drop)

    for col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast(FloatType()))

    df = df.fillna(0)

    assembler = VectorAssembler(inputCols=([x for x in df.columns if x not in ['rating']]), outputCol="features")
    pipeline = Pipeline(stages = [assembler])
    pipelineModel = pipeline.fit(df)

    df = pipelineModel.transform(df)

    selected_cols = ['features', 'rating']
    df = df.select(selected_cols)

    (trainingData, testData) = df.randomSplit([0.8, 0.2])

    lr = LinearRegression(featuresCol = 'features', labelCol = 'rating', maxIter=10)

    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[lr])

    evaluator = RegressionEvaluator(
        labelCol="rating", predictionCol="prediction", metricName="rmse")

    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(),
                              numFolds=2)

    # Train model.  This also runs the indexer.
    model = pipeline.fit(df)

    # Make predictions.
    predictions = model.transform(trainingData)

    predictions.show()

    # Select example rows to display.
    predictions.select("prediction", "rating", "features").show(5)

    # Select (prediction, true label) and compute test error
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)





