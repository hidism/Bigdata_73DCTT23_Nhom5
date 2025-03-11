from pyspark.ml.feature import PCA
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("DimensionalityReduction").getOrCreate()

data = [(Vectors.dense([1.0, 2.0, 3.0]),), (Vectors.dense([4.0, 5.0, 6.0]),)]
df = spark.createDataFrame(data, ["features"])

pca = PCA(k=2, inputCol="features", outputCol="pca_features")
model = pca.fit(df)
result = model.transform(df)

result.select("pca_features").show(truncate=False)
