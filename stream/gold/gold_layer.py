"""Gold layer processing for air quality stream."""

from pyspark.sql.functions import (
    col, from_json, to_timestamp, year, month, dayofmonth, isnan,
    max as spark_max, collect_list, window
)
from pyspark.sql.types import StructType


def process_gold_layer(kafka_df, s3_path, kafka_message_schema: StructType):
    """Process Gold layer: Aggregate by location and hour."""
    print("Setting up Gold layer processing...")

    gold_df = kafka_df.select(
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
    ).filter(
        col("datetime").isNotNull()
        & col("location_id").isNotNull()
        & col("parameter").isNotNull()
        & col("value").isNotNull()
        & (col("value") >= 0)
        & ~isnan(col("value"))
        & col("aqi").isNotNull()
    )

    gold_df = gold_df.withWatermark("datetime", "1 hour")

    gold_aggregated = gold_df \
        .groupBy(
            window(col("datetime"), "1 hour").alias("hour_window"),
            col("location_id"),
            col("location_name"),
            col("country"),
            col("latitude"),
            col("longitude"),
        ) \
        .agg(
            spark_max("aqi").alias("aqi"),
            collect_list("parameter").alias("parameters"),
            collect_list("value").alias("values"),
            spark_max("aqi_category").alias("aqi_category"),
        ) \
        .select(
            col("hour_window.start").alias("datetime"),
            col("location_id"),
            col("location_name"),
            col("country"),
            col("latitude"),
            col("longitude"),
            col("aqi"),
            col("parameters"),
            col("values"),
            col("aqi_category"),
        )

    gold_aggregated = gold_aggregated.withColumn("year", year(col("datetime"))) \
        .withColumn("month", month(col("datetime"))) \
        .withColumn("day", dayofmonth(col("datetime")))

    gold_query = gold_aggregated.writeStream \
        .format("parquet") \
        .outputMode("append") \
        .option("path", f"{s3_path}/gold") \
        .option("checkpointLocation", f"{s3_path}/checkpoints/gold") \
        .partitionBy("year", "month", "day") \
        .trigger(processingTime="1 minute") \
        .start()

    print("Gold layer streaming started")
    return gold_query
