"""Bronze layer processing for air quality stream."""

from pyspark.sql.functions import col, from_json, to_timestamp, year, month, dayofmonth
from pyspark.sql.types import StructType


def process_bronze_layer(kafka_df, s3_path, kafka_message_schema: StructType):
    """Process Bronze layer: Save raw data as-is."""
    print("Setting up Bronze layer processing...")

    bronze_df = kafka_df.select(
        from_json(col("value").cast("string"), kafka_message_schema).alias("data")
    ).select(
        to_timestamp(col("data.datetime")).alias("datetime"),
        col("data.location_id").alias("location_id"),
        col("data.location_name").alias("location_name"),
        col("data.country").alias("country"),
        col("data.latitude").alias("latitude"),
        col("data.longitude").alias("longitude"),
        col("data.parameter").alias("parameter"),
        col("data.value").alias("value"),
        col("data.value_standard").alias("value_standard"),
        col("data.unit").alias("unit"),
        col("data.aqi").alias("aqi"),
        col("data.aqi_category").alias("aqi_category"),
        to_timestamp(col("data.ingestion_timestamp")).alias("ingestion_timestamp"),
    ).filter(
        col("datetime").isNotNull()
        & col("location_id").isNotNull()
        & col("parameter").isNotNull()
    )

    bronze_df = bronze_df.withColumn("year", year(col("datetime"))) \
        .withColumn("month", month(col("datetime"))) \
        .withColumn("day", dayofmonth(col("datetime")))

    bronze_query = bronze_df.writeStream \
        .format("parquet") \
        .outputMode("append") \
        .option("path", f"{s3_path}/bronze") \
        .option("checkpointLocation", f"{s3_path}/checkpoints/bronze") \
        .partitionBy("year", "month", "day") \
        .trigger(processingTime="10 seconds") \
        .start()

    print("Bronze layer streaming started")
    return bronze_query
