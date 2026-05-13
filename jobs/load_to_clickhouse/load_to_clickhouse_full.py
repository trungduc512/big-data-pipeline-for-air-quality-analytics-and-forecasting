"""
Script to load Gold layer data from MinIO (S3) to ClickHouse without date filtering.
"""

import argparse
import os

import clickhouse_connect
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from tqdm import tqdm

# Configuration
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "air-quality-data")
CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "air_quality")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "default123")
# Use native protocol port if HTTP fails
CLICKHOUSE_NATIVE_PORT = int(os.environ.get("CLICKHOUSE_NATIVE_PORT", "9002"))


def get_s3_filesystem():
    """Create S3 filesystem connection to MinIO."""
    return s3fs.S3FileSystem(
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        use_ssl=False,
    )


def get_clickhouse_client():
    """Create ClickHouse client connection."""
    # clickhouse-connect requires username and password parameters
    # For empty password, pass empty string explicitly
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        database=CLICKHOUSE_DATABASE,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )


def load_gold_to_clickhouse(client, fs):
    """Load Gold layer data from MinIO to ClickHouse."""
    print("Loading Gold layer data...")

    gold_path = f"{MINIO_BUCKET}/gold/"
    parquet_files = fs.glob(f"{gold_path}**/*.parquet")

    if not parquet_files:
        print(f"No Gold files found in {gold_path}")
        return

    print(f"Found {len(parquet_files)} Parquet files")

    total_records = 0
    for file_path in tqdm(parquet_files, desc="Loading Gold files"):
        try:
            with fs.open(file_path, "rb") as f:
                table = pq.read_table(f)
                df = table.to_pandas()

            if len(df) == 0:
                continue

            # Convert datetime columns
            df["datetime"] = pd.to_datetime(df["datetime"])
            if "aggregation_timestamp" in df.columns:
                df["aggregation_timestamp"] = pd.to_datetime(
                    df["aggregation_timestamp"], errors="coerce"
                )

            # Fill None values for non-nullable columns
            string_cols = ["location_name", "country", "aqi_category"]
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].fillna("")

            float_cols = ["latitude", "longitude", "aqi"]
            for col in float_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)

            # Ensure arrays are proper lists for ClickHouse
            if "parameters" in df.columns:
                df["parameters"] = df["parameters"].apply(
                    lambda x: x
                    if isinstance(x, list)
                    else (list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else [])
                )
                df["parameters"] = df["parameters"].apply(lambda x: x if x is not None else [])
            if "values" in df.columns:
                df["values"] = df["values"].apply(
                    lambda x: x
                    if isinstance(x, list)
                    else (list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else [])
                )
                df["values"] = df["values"].apply(lambda x: x if x is not None else [])

            client.insert_df("gold_hourly_aqi", df)
            total_records += len(df)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {total_records} records to Gold table")


def main():
    parser = argparse.ArgumentParser(
        description="Load Gold layer data from MinIO to ClickHouse (no date filter)"
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=["gold"],
        default="gold",
        help="Which layer to load (default: gold)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MinIO to ClickHouse Gold Data Loader")
    print("=" * 80)
    print(f"MinIO endpoint: {MINIO_ENDPOINT}")
    print(f"ClickHouse: {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}")
    print(f"Layer: {args.layer}")
    print("=" * 80)

    try:
        fs = get_s3_filesystem()
        client = get_clickhouse_client()
        print("Connected to MinIO and ClickHouse successfully")
    except Exception as e:
        print(f"Error connecting to services: {e}")
        return

    load_gold_to_clickhouse(client, fs)

    print("\nData loading complete!")


if __name__ == "__main__":
    main()
