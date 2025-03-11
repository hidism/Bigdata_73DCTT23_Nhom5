from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName("ModelEvaluation").getOrCreate()

data = spark.createDataFrame([(0, 1, 4.0), (0, 2, 5.0), (1, 2, 3.0)], ["user", "item", "rating"])
als = ALS(maxIter=5, regParam=0.01, userCol="user", itemCol="item", ratingCol="rating")
model = als.fit(data)

predictions = model.transform(data)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")
