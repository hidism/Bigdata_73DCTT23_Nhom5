from pyspark.sql import SparkSession
from pyspark.ml.stat import Summarizer
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("Statistics").getOrCreate()

data = [(Vectors.dense([1.0, 2.0, 3.0]),), (Vectors.dense([4.0, 5.0, 6.0]),)]
df = spark.createDataFrame(data, ["features"])

# Tính toán mean, variance
summary = df.select(Summarizer.mean(df.features), Summarizer.variance(df.features))
summary.show()
