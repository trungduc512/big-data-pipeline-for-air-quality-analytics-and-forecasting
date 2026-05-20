FROM python:3.11-slim

WORKDIR /app

COPY batch/backfill/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY batch/backfill/backfill_to_bronze.py .
COPY batch/backfill/aqi_calculator.py .

CMD ["python", "backfill_to_bronze.py"]
