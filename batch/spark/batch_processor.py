"""
Spark Batch Processing job for ML Training Data Pipeline
Pipeline: Bronze → Silver → Gold (sequential, not parallel)
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, isnan, max as spark_max, collect_list,
    date_trunc
)

# Configuration from environment
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "historical-air-quality-data")

# MinIO S3 path
S3_PATH = f"s3a://{MINIO_BUCKET}"


def get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        print(f"[Spark] Invalid {name}={value!r}; using {default}")
        return default
    return max(1, parsed)


def create_spark_session():
    """Create Spark session with MinIO S3 configuration."""
    import multiprocessing
    available_cores = multiprocessing.cpu_count()
    num_cores = min(get_int_env("SPARK_LOCAL_CORES", 4), available_cores)
    shuffle_partitions = get_int_env("SPARK_SQL_SHUFFLE_PARTITIONS", max(8, num_cores * 2))
    driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "3g")
    executor_memory = os.environ.get("SPARK_EXECUTOR_MEMORY", "3g")
    shuffle_dir = os.environ.get("SPARK_LOCAL_DIR", "/tmp/spark-shuffle")
    
    spark = SparkSession.builder \
        .appName("AirQualityBatchProcessor") \
        .master(f"local[{num_cores}]") \
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .config("spark.local.dir", shuffle_dir) \
        .config("spark.executor.cores", str(num_cores)) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1") \
        .config("spark.sql.adaptive.coalescePartitions.initialPartitionNum", str(shuffle_partitions)) \
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
        .config("spark.sql.adaptive.shuffle.targetPostShuffleInputSize", "128MB") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print(
        "[Spark] Initialized "
        f"local[{num_cores}], driver={driver_memory}, executor={executor_memory}, "
        f"shuffle_partitions={shuffle_partitions}"
    )
    return spark


def process_bronze_to_silver(spark):
    """Job 1: Read from Bronze → Write to Silver (Cleaned Data)"""
    print("[Job 1] Processing Bronze → Silver...")
    
    # Read from Bronze
    bronze_path = f"{S3_PATH}/bronze"
    bronze_df = spark.read.parquet(bronze_path)
    
    # Clean and standardize
    silver_df = bronze_df.select(
        col("datetime"),
        col("location_id"),
        col("location_name"),
        col("country"),
        col("latitude"),
        col("longitude"),
        col("parameter"),
        col("value_standard").alias("value"),  # Use value_standard only
        col("aqi"),
        col("aqi_category"),
        col("year"),
        col("month"),
        col("day")
    ).filter(
        col("datetime").isNotNull() &
        col("location_id").isNotNull() &
        col("parameter").isNotNull() &
        col("value").isNotNull() &
        (col("value") >= 0) &  # Only valid values
        ~isnan(col("value"))
    )
    
    # Write to MinIO
    silver_path = f"{S3_PATH}/silver"
    print(f"[Job 1] Writing to {silver_path}...")
    silver_df.write \
        .mode("overwrite") \
        .format("parquet") \
        .partitionBy("year", "month", "day") \
        .save(silver_path)
    
    count = silver_df.count()
    print(f"[Job 1] ✓ Completed: {count} records written to Silver")
    return silver_path


def process_silver_to_gold(spark):
    """Job 2: Read from Silver → Write to Gold (Aggregated by Hour)"""
    print("[Job 2] Processing Silver → Gold...")
    
    # Read from Silver
    silver_path = f"{S3_PATH}/silver"
    silver_df = spark.read.parquet(silver_path)
    
    # Aggregate by location and hour
    gold_df = silver_df \
        .withColumn("hour_datetime", date_trunc("hour", col("datetime"))) \
        .groupBy(
            col("hour_datetime").alias("datetime"),
            col("location_id"),
            col("location_name"),
            col("country"),
            col("latitude"),
            col("longitude"),
            col("year"),
            col("month"),
            col("day")
        ) \
        .agg(
            spark_max("aqi").alias("aqi"),
            collect_list("parameter").alias("parameters"),
            collect_list("value").alias("values"),
            spark_max("aqi_category").alias("aqi_category")
        )
    
    # Write to MinIO
    gold_path = f"{S3_PATH}/gold"
    print(f"[Job 2] Writing to {gold_path}...")
    gold_df.write \
        .mode("overwrite") \
        .format("parquet") \
        .partitionBy("year", "month", "day") \
        .save(gold_path)
    
    count = gold_df.count()
    print(f"[Job 2] ✓ Completed: {count} records written to Gold")
    return gold_path


def main():
    """Main function: Run batch processing pipeline sequentially."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch processing pipeline for ML training data")
    parser.add_argument("--layer", choices=["silver", "gold", "all"], 
                       default="all", help="Which layer to process")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Spark Batch Processing Pipeline for ML Training")
    print("=" * 60)
    
    spark = create_spark_session()
    
    try:
        if args.layer == "silver" or args.layer == "all":
            process_bronze_to_silver(spark)
        
        if args.layer == "gold" or args.layer == "all":
            process_silver_to_gold(spark)
        
        print("=" * 60)
        print("✓ All batch processing jobs completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
