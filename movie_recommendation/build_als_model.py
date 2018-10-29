import time

import pyspark
from pyspark.sql.functions import mean, col
from pyspark.sql.types import StructField, StructType, StringType, ArrayType, DoubleType, LongType, FloatType, BooleanType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.recommendation import ALS

import json

def set_als_params(df):
    """
    Search the best parameters to use the ALS model
    :param df: dataframe with userId, movieId, rating
    :return:
    """

    (df_train, df_test) = df.randomSplit([85.0, 15.0])

    # Cache these datasets for performance
    training_df = df_train.cache()
    testing_df = df_test.cache()

    print('Training: {0}, test: {1}\n'.format(
        training_df.count(), testing_df.count()))

    print('Training mean rating: {0}, test mean rating: {1}\n'.format(
        training_df.select(mean(col('rating'))), testing_df.select(mean(col('rating')))))

    training_df.show(3)
    testing_df.show(3)

    # initialize ALS learner
    als = ALS()

    # set the parameters for the method
    als.setMaxIter(5) \
        .setRegParam(0.1) \
        .setUserCol("userId").setItemCol("movieId").setRatingCol("rating")

    # Create an RMSE evaluator using the label and predicted columns
    reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")

    ranks = [8, 10, 12, 15]
    max_iters = [5, 10, 20]
    reg_params = [0.05, 0.1, 0.2, 0.5]
    err = 0
    min_error = float('inf')
    best_rank = -1
    best_maxiter = -1
    best_regparam = -1

    for rank in ranks:
        for max_iter in max_iters:
            for reg_param in reg_params:

                # Set the rank
                als.setRank(rank).setRegParam(reg_param).setMaxIter(max_iter)

                # Create the model with these parameters.
                model = als.fit(training_df)

                # Run the model to create a prediction.
                predict_df = model.transform(testing_df)

                # Remove NaN values from prediction (due to SPARK-14489)
                predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))

                # Run the previously created RMSE evaluator
                error = reg_eval.evaluate(predicted_ratings_df)

                print('For rank %s, for max_iter %s, for reg_param %s, the RMSE is %s' % (rank, max_iter, reg_param, error))
                if error < min_error:
                    min_error = error
                    best_rank = rank
                    best_maxiter = max_iter
                    best_regparam = reg_param
                    # best_model = model

                err += 1

    # als.setRank(best_rank).setMaxIter(best_maxiter).setRegParam(best_regparam)

    print('The best model was trained with rank %s with max_iter %s, with reg_param %s' % (best_rank, best_maxiter, best_regparam))

    return best_rank, best_regparam, best_maxiter, min_error

def check_results(df, best_rank, best_regparam, best_maxiter, k = 5):
    """
    This function is used to remplace the cross validation
    The dataset is splitted in different random ways and the model is trained then tested to be able
    to determine the error in a more accurate way
    This is to avoid overfitting
    :param df:
    :param model:
    :param k:
    :return:
    """

    # Repeat the operation to have a more precise estimation of the result
    # ~ cross validate results

    evaluations = []
    for i in range(0, k):
        (trainingSet, testingSet) = df.randomSplit([k - 1.0, 1.0])
        als = ALS()

        # Now we set the parameters for the method
        als.setMaxIter(best_maxiter) \
            .setRegParam(best_regparam) \
            .setRank(best_rank) \
            .setUserCol("userId").setItemCol("movieId").setRatingCol("rating")

        model = als.fit(trainingSet)
        predictions = model.transform(testingSet)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        evaluation = evaluator.evaluate(predictions.na.drop())
        print("evaluation ", k, ": ", evaluation)
        evaluations.append(evaluation)

    error = sum(evaluations) / float(len(evaluations))
    print("The root mean squared error for our model is: ", error)
    return error

def cross_validated_model(ratings):
    """
    Cross validation with param grid search
    Not used because of the userId/movie splitting is not well handled (SPARK-14489)
    :param ratings: dataframe with userId, movieId, rating columns
    :return: best model
    """

    ## because of the SPARK-14489
    ## the CrossValidator function can't compute the root mean squared error most of the time
    ## and provides incorrect results

    (trainingRatings, validationRatings) = ratings.randomSplit([90.0, 10.0])
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    # Let's initialize our ALS learner
    als = ALS()
    # Now we set the parameters for the method
    als.setMaxIter(5) \
        .setRegParam(0.1) \
        .setUserCol("userId").setItemCol("movieId").setRatingCol("rating")

    paramGrid = ParamGridBuilder().addGrid(als.rank, [1, 5, 10]).addGrid(als.maxIter, [5, 20]).addGrid(als.regParam, [0.05, 0.1, 0.5]).build()
    crossval = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = crossval.fit(trainingRatings)
    predictions = cvModel.transform(validationRatings)

    print("The root mean squared error for our model is: " + str(evaluator.evaluate(predictions.na.drop())))
    return cvModel

def build_model(df, save = False, output_model_path = None, output_param_path = None):
    """
    Build the model and save it
    :param df:
    :return:
    """

    best_rank, best_regparam, best_maxiter, min_error = set_als_params(df)

    error = check_results(df, best_rank, best_regparam, best_maxiter, k=5)

    als = ALS()
    # Now we set the parameters for the method
    als.setMaxIter(best_maxiter) \
        .setRegParam(best_regparam) \
        .setRank(best_rank) \
        .setUserCol("userId").setItemCol("movieId").setRatingCol("rating")

    best_model = als.fit(df)

    if save :
        best_model.save(output_model_path)

        params = {"best_rank":best_rank, "best_param" : best_regparam, "best_maxiter" : best_maxiter, "error_check": error, "min_error_model":min_error}

        # save the model parameters
        with open(output_param_path, 'w') as fp:
            json.dump(params, fp)

    return best_model


