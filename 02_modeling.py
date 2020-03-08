
import pandas as pd 
import json

#Graph network imports
from pyspark import *
from pyspark.sql import *
import numpy as np
from pyspark.ml.linalg import *
from pyspark.sql.functions import *
from pyspark.sql.types import * #Import types == IntegerType, StringType etc.
from pyspark.ml.regression import LinearRegression

import nltk
from datetime import datetime

from pyspark.sql import SparkSession

#create Spark session
spark = SparkSession.builder.enableHiveSupport().appName('Final_project_read_write').getOrCreate()

conf = spark.sparkContext._conf.setAll([('spark.executor.memory', '15g'), 
                                        ('spark.app.name', 'Final_project_read_write'), 
                                        ('spark.executor.cores', '10'), 
                                        ('spark.cores.max', '10'), 
                                        ('spark.driver.memory','20g')])

#print spark configuration settings

#modeling_data = spark.read.parquet('modeling_data')
modeling_data = spark.read.parquet('big_data_project/modeling_data2')

modeling_data.printSchema()

#split data into train and test
train_df, test_df = modeling_data.select('id','inCitations_count', 'features').randomSplit([0.8, 0.2], seed=42)
train_df.show(1)

print('Train Length = ', train_df.count())
print('Test Length = ', test_df.count())


start = datetime.now()
print('Training start time is {}'.format(start))
#Elastic Net
lr = LinearRegression(featuresCol = 'features', labelCol='inCitations_count', regParam=0.3, maxIter=10)
lrm = lr.fit(train_df)


def modelsummary(model):
    import numpy as np
    print("Note: the last rows are the information for Intercept")
    print("##","-------------------------------------------------")
    print("##","  Estimate   |   Std.Error | t Values  |  P-value")
    coef = np.append(list(model.coefficients),model.intercept)
    Summary=model.summary

    for i in range(len(Summary.pValues)):
        print ("##",'{:10.6f}'.format(coef[i]),\
        '{:10.6f}'.format(Summary.coefficientStandardErrors[i]),\
        '{:8.3f}'.format(Summary.tValues[i]),\
        '{:10.6f}'.format(Summary.pValues[i]))

    print ("##",'---')
    print ("##","Mean squared error: % .6f" \
           % Summary.meanSquaredError, ", RMSE: % .6f" \
           % Summary.rootMeanSquaredError )
    print ("##","Multiple R-squared: %f" % Summary.r2, ", \
            Total iterations: %i"% Summary.totalIterations)

modelsummary(lrm)

end = datetime.now()
print('Total train time is {}'.format(end-start))

#coefficients
print("Coefficients: " + str(lrm.coefficients))
print("Intercept: " + str(lrm.intercept))

#model summary
print("RMSE: %f" % lrm.summary.rootMeanSquaredError)
print("r2: %f" % lrm.summary.r2)

model_name = 'lrm_model'+datetime.now().strftime('%Y.%m.%d.%H%M%S')
# saving:
print('Saving model as {}'.format(model_name))
lrm.save(model_name)

# # loading:
# from pyspark.ml.regression import LinearRegressionModel
# model = LinearRegressionModel.load('lrm_model_ss')


print('Making Predictions')
#make predictions
predictions = lrm.transform(test_df)

# from itertools import chain
# attrs = sorted(
#     (attr["idx"], attr["name"]) for attr in (chain(*predictions
#         .schema[lrm.summary.featuresCol]
#         .metadata["ml_attr"]["attrs"].values())))

print('Evaluating the model')
from pyspark.ml.evaluation import RegressionEvaluator

eval = RegressionEvaluator(labelCol="inCitations_count", predictionCol="prediction", metricName="rmse")

# Root Mean Square Error
rmse = eval.evaluate(predictions)
print("RMSE: %.3f" % rmse)
# r2 - coefficient of determination
r2 = eval.evaluate(predictions, {eval.metricName: "r2"})
print("r2: %.3f" %r2)

print('Running RF regressor')
from pyspark.ml.regression import RandomForestRegressor

start_rf = datetime.now()
print('Start time of RF is {}'.format(start_rf))

MAX_D = 10
NUM_TREES = 25
# Set parameters for the Random Forest.
rfr = RandomForestRegressor(maxDepth=MAX_D, numTrees=NUM_TREES, labelCol="inCitations_count", predictionCol="prediction")
# Fit the model to the data.
rfrm = rfr.fit(train_df)

end_rf = datetime.now()
print('End time of RF is {}'.format(end_rf))
print('Total train time is {}'.format(end_rf-start_rf))


model_name_rf = 'rf_model_t{}_dep{}'.format(NUM_TREES, MAX_D) +datetime.now().strftime('%Y.%m.%d.%H%M%S')

# saving:
print('Saving model as {}'.format(model_name_rf))
rfrm.save(model_name_rf)

print('Making predictions')
# Given a dataset, predict each point's label, and show the results.
predictions_rf = rfrm.transform(test_df)

eval = RegressionEvaluator(labelCol="inCitations_count", predictionCol="prediction", metricName="rmse")

# Root Mean Square Error
rmse = eval.evaluate(predictions_rf)
print("RMSE: %.3f" % rmse)

# Mean Square Error
mse = eval.evaluate(predictions_rf, {eval.metricName: "mse"})
print("MSE: %.3f" % mse)

# Mean Absolute Error
mae = eval.evaluate(predictions_rf, {eval.metricName: "mae"})
print("MAE: %.3f" % mae)

# r2 - coefficient of determination
r2 = eval.evaluate(predictions, {eval.metricName: "r2"})
print("r2: %.3f" %r2)


print('END')
# Exercise: Build a feature importance selector
# Reference:
# https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/
