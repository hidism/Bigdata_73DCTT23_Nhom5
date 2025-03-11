from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Recommendation").getOrCreate()

# Dữ liệu mẫu: (userId, itemId, rating)
data = spark.createDataFrame([(0, 1, 4.0), (0, 2, 5.0), (1, 2, 3.0), (1, 3, 4.0)], ["user", "item", "rating"])

# Khởi tạo mô hình ALS
als = ALS(maxIter=10, regParam=0.1, userCol="user", itemCol="item", ratingCol="rating")
model = als.fit(data)

# Dự đoán cho user 0, item 3
test_data = spark.createDataFrame([(0, 3)], ["user", "item"])
predictions = model.transform(test_data)

predictions.show()
