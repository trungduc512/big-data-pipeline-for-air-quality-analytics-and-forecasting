FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    clickhouse-connect \
    s3fs \
    pyarrow \
    pandas \
    tqdm

COPY load_to_clickhouse_full.py /app/load_to_clickhouse_full.py

ENTRYPOINT ["python", "/app/load_to_clickhouse_full.py"]
