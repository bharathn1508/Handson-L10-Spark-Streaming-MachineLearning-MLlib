import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, avg, window, hour, minute, to_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# ------------------- CONFIG ------------------- #
MODEL_PATH = "models/fare_trend_model_v2"
TRAINING_DATA_PATH = "training-dataset.csv"

# Demo vs Assignment toggles:
# If you need instant console output during testing, keep DEMO_FAST=True.
# For the original assignment spec (5-min window, 1-min slide), set DEMO_FAST=False.
DEMO_FAST = True

if DEMO_FAST:
    WINDOW_DURATION = "1 minute"
    SLIDE_DURATION = "10 seconds"
    WATERMARK_DELAY = "10 seconds"
    TRIGGER_INTERVAL = "5 seconds"
else:
    WINDOW_DURATION = "5 minutes"
    SLIDE_DURATION = "1 minute"
    WATERMARK_DELAY = "1 minute"
    TRIGGER_INTERVAL = "10 seconds"

# ------------------- SPARK ------------------- #
spark = (
    SparkSession.builder
    .appName("Task7_FareTrendPrediction_Assignment")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ------------------- MODEL TRAINING (Offline) ------------------- #
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] Training model using {TRAINING_DATA_PATH}...")

    # Load and process historical data
    hist_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)
    hist_df_processed = (
        hist_df_raw
        .withColumn("event_time", to_timestamp(col("timestamp")))     # safer than cast
        .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))
        .na.drop(subset=["event_time", "fare_amount"])
    )

    # Windowed aggregation -> average fare per time window
    hist_windowed_df = (
        hist_df_processed
        .groupBy(window(col("event_time"), WINDOW_DURATION))
        .agg(avg("fare_amount").alias("avg_fare"))
    )

    # Time-based features (from window start)
    hist_features = (
        hist_windowed_df
        .withColumn("hour_of_day", hour(col("window.start")))
        .withColumn("minute_of_hour", minute(col("window.start")))
    )

    # Assemble features
    assembler = VectorAssembler(
        inputCols=["hour_of_day", "minute_of_hour"],
        outputCol="features"
    )
    train_df = assembler.transform(hist_features).select("features", "avg_fare")

    # Train linear regression model
    lr = LinearRegression(featuresCol="features", labelCol="avg_fare")
    model = lr.fit(train_df)

    # Save the model
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Model Saved] -> {MODEL_PATH}")
else:
    print(f"[Model Found] Using existing model at {MODEL_PATH}")

# ------------------- STREAMING INFERENCE ------------------- #
print("\n[Inference Phase] Starting real-time trend prediction stream...")

# Define the schema for incoming data
schema = StructType([
    StructField("trip_id", StringType()),
    StructField("driver_id", IntegerType()),
    StructField("distance_km", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("timestamp", StringType()),
])

# Read socket stream and parse JSON
raw_stream = (
    spark.readStream
    .format("socket")
    .option("host", "localhost")
    .option("port", 9999)
    .load()
)

parsed_stream = (
    raw_stream
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
    .withColumn("event_time", to_timestamp(col("timestamp")))
    .withWatermark("event_time", WATERMARK_DELAY)
)

# Windowed aggregation on the stream
windowed_df = (
    parsed_stream
    .groupBy(window(col("event_time"), WINDOW_DURATION, SLIDE_DURATION))
    .agg(avg("fare_amount").alias("avg_fare"))
)

# Stream features
windowed_features = (
    windowed_df
    .withColumn("hour_of_day", hour(col("window.start")))
    .withColumn("minute_of_hour", minute(col("window.start")))
)

assembler_inference = VectorAssembler(
    inputCols=["hour_of_day", "minute_of_hour"],
    outputCol="features"
)
feature_df = assembler_inference.transform(windowed_features)

# Load trained model and predict
trend_model = LinearRegressionModel.load(MODEL_PATH)
predictions = trend_model.transform(feature_df)

# Final output selection
output_df = predictions.select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    col("avg_fare"),
    col("prediction").alias("predicted_next_avg_fare")
)

# Write to console frequently so you can see updates
query = (
    output_df.writeStream
    .format("console")
    .outputMode("update")  # show partial aggregates as the window evolves
    .option("truncate", False)
    .trigger(processingTime=TRIGGER_INTERVAL)
    .start()
)

query.awaitTermination()