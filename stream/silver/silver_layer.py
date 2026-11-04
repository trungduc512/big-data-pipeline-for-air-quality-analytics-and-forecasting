"""Silver layer processing for air quality stream."""

from pyspark.sql.functions import col, from_json, to_timestamp, year, month, dayofmonth, isnan, current_timestamp
from pyspark.sql.types import StructType


def process_silver_layer(kafka_df, s3_path, kafka_message_schema: StructType):
    """Process Silver layer: Clean and standardize data."""
    print("Setting up Silver layer processing...")

    silver_df = kafka_df.select(
        from_json(col("value").cast("string"), kafka_message_schema).alias("data")
    ).select(
        to_timestamp(col("data.datetime")).alias("datetime"),
        col("data.location_id").alias("location_id"),
        col("data.location_name").alias("location_name"),
        col("data.country").alias("country"),
        col("data.latitude").alias("latitude"),
        col("data.longitude").alias("longitude"),
        col("data.parameter").alias("parameter"),
        col("data.value_standard").alias("value"),
        col("data.aqi").alias("aqi"),
        col("data.aqi_category").alias("aqi_category"),
        current_timestamp().alias("processing_timestamp"),
    ).filter(
        col("datetime").isNotNull()
        & col("location_id").isNotNull()
        & col("parameter").isNotNull()
        & col("value").isNotNull()
        & (col("value") >= 0)
        & ~isnan(col("value"))
    )

    silver_df = silver_df.withColumn("year", year(col("datetime"))) \
        .withColumn("month", month(col("datetime"))) \
        .withColumn("day", dayofmonth(col("datetime")))

    silver_query = silver_df.writeStream \
        .format("parquet") \
        .outputMode("append") \
        .option("path", f"{s3_path}/silver") \
        .option("checkpointLocation", f"{s3_path}/checkpoints/silver") \
        .partitionBy("year", "month", "day") \
        .trigger(processingTime="10 seconds") \
        .start()

    print("Silver layer streaming started")
    return silver_query
