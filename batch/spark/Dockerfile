FROM apache/spark-py:latest

WORKDIR /opt/spark/work-dir

COPY batch/spark/batch_processor.py .
COPY batch/spark/spark.conf .

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PATH="/opt/spark/bin:${PATH}"

RUN mkdir -p /tmp/.ivy/ && chmod -R 777 /tmp/.ivy

CMD ["spark-submit", "--properties-file", "spark.conf", "batch_processor.py"]
