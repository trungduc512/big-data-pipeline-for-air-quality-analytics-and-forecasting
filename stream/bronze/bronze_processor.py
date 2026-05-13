"""Standalone Spark streaming processor for Bronze layer."""

from bronze_layer import process_bronze_layer
from config import (
    KAFKA_MESSAGE_SCHEMA,
    S3_PATH,
    create_spark_session,
    read_kafka_stream,
)


def main():
    """Start Bronze layer streaming job."""
    print("Initializing Spark session...")
    spark = create_spark_session()

    print("Reading from Kafka...")
    kafka_df = read_kafka_stream(spark)

    print("Starting Bronze streaming query...")
    query = process_bronze_layer(kafka_df, S3_PATH, KAFKA_MESSAGE_SCHEMA)

    print("Bronze query started. Waiting for termination...")
    query.awaitTermination()


if __name__ == "__main__":
    main()
