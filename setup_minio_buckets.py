"""
Script to create MinIO buckets and set up initial structure.
"""

from minio import Minio
from minio.error import S3Error
import os

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
BUCKET_NAME = os.environ.get("MINIO_BUCKET", "air-quality-data")


def setup_minio():
    """Create MinIO bucket if it doesn't exist."""
    try:
        # Create MinIO client
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        # Check if bucket exists
        found = client.bucket_exists(BUCKET_NAME)
        
        if not found:
            client.make_bucket(BUCKET_NAME)
            print(f"Created bucket: {BUCKET_NAME}")
        else:
            print(f"Bucket {BUCKET_NAME} already exists")
        
        print("MinIO setup complete!")
        print(f"Bucket: {BUCKET_NAME}")
        print(f"Endpoint: http://{MINIO_ENDPOINT}")
        
    except S3Error as e:
        print(f"Error setting up MinIO: {e}")


if __name__ == "__main__":
    setup_minio()

