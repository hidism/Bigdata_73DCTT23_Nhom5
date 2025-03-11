from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession

# Khởi tạo Spark Session
spark = SparkSession.builder.appName("FeatureExtraction").getOrCreate()

# Tạo DataFrame
data = spark.createDataFrame([(0, "Apache Spark is fast"), (1, "Spark is scalable")], ["id", "text"])

# Tokenizer (Tách từ)
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(data)

# TF (Term Frequency)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# IDF (Inverse Document Frequency)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("id", "features").show(truncate=False)
