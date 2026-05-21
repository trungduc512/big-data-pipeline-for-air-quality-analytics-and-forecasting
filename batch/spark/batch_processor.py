"""
Spark Batch Processing job for ML Training Data Pipeline
Pipeline: Bronze → Silver → Gold (sequential, not parallel)
"""

import os
import sys
from io import BytesIO
from urllib.parse import urlparse

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
BRONZE_REPAIR_SCHEMA = os.environ.get(
    "BRONZE_REPAIR_SCHEMA", "true"
).lower() in {"1", "true", "yes", "on"}


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


def create_minio_client():
    """Create a MinIO client for pre-Spark object repairs."""
    parsed = urlparse(MINIO_ENDPOINT)
    if parsed.scheme:
        endpoint = parsed.netloc
        secure = parsed.scheme == "https"
    else:
        endpoint = MINIO_ENDPOINT
        secure = False

    from minio import Minio

    return Minio(
        endpoint,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=secure,
    )


def bronze_target_type(field_name: str):
    import pyarrow as pa

    target_types = {
        "datetime": pa.timestamp("us"),
        "ingestion_timestamp": pa.timestamp("us"),
        "location_id": pa.int64(),
        "location_name": pa.string(),
        "country": pa.string(),
        "latitude": pa.float64(),
        "longitude": pa.float64(),
        "parameter": pa.string(),
        "value": pa.float64(),
        "value_standard": pa.float64(),
        "unit": pa.string(),
        "aqi": pa.float64(),
        "aqi_category": pa.string(),
    }
    return target_types.get(field_name)


def normalize_bronze_table(table):
    """Normalize Bronze Parquet physical types so Spark can read all files together."""
    import pyarrow as pa

    fields = []
    changed = False
    for field in table.schema:
        target_type = bronze_target_type(field.name)
        if target_type is not None and field.type != target_type:
            fields.append(
                pa.field(
                    field.name,
                    target_type,
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
            changed = True
        else:
            fields.append(field)

    if not changed:
        return table, False

    return table.cast(pa.schema(fields, metadata=table.schema.metadata), safe=False), True


def repair_bronze_schema() -> None:
    """Rewrite existing Bronze Parquet files with Spark-compatible physical types."""
    if not BRONZE_REPAIR_SCHEMA:
        print("[Repair] Bronze schema repair disabled")
        return

    print("[Repair] Scanning Bronze Parquet files for Spark-incompatible schemas...")
    import pyarrow.parquet as pq

    client = create_minio_client()
    objects = client.list_objects(MINIO_BUCKET, prefix="bronze/", recursive=True)
    scanned = 0
    repaired = 0

    for obj in objects:
        object_name = obj.object_name
        if not object_name.endswith(".parquet"):
            continue

        scanned += 1
        response = client.get_object(MINIO_BUCKET, object_name)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()

        table = pq.read_table(BytesIO(data))
        table, changed = normalize_bronze_table(table)
        if not changed:
            continue

        output = BytesIO()
        pq.write_table(
            table,
            output,
            coerce_timestamps="us",
            allow_truncated_timestamps=True,
        )
        output.seek(0)
        client.put_object(
            MINIO_BUCKET,
            object_name,
            output,
            length=output.getbuffer().nbytes,
            content_type="application/octet-stream",
        )
        repaired += 1
        if repaired % 25 == 0:
            print(f"[Repair] Rewritten {repaired} Bronze Parquet files so far...")

    print(f"[Repair] Completed: scanned={scanned}, rewritten={repaired}")


def process_bronze_to_silver(spark):
    """Job 1: Read from Bronze → Write to Silver (Cleaned Data)"""
    print("[Job 1] Processing Bronze → Silver...")
    repair_bronze_schema()
    
    # Read from Bronze
    bronze_path = f"{S3_PATH}/bronze"
    bronze_df = spark.read.parquet(bronze_path)
    
    # Clean and standardize
    silver_df = bronze_df.select(
        col("datetime").cast("timestamp").alias("datetime"),
        col("location_id").cast("long").alias("location_id"),
        col("location_name"),
        col("country"),
        col("latitude").cast("double").alias("latitude"),
        col("longitude").cast("double").alias("longitude"),
        col("parameter"),
        col("value_standard").cast("double").alias("value"),  # Use value_standard only
        col("aqi").cast("double").alias("aqi"),
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
