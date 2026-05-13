FROM apache/spark-py:latest

WORKDIR /opt/spark/work-dir

# COPY hadoop-aws-3.3.4.jar /opt/spark/jars/

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PATH="/opt/spark/bin:${PATH}"

RUN mkdir -p /tmp/.ivy/ && chmod -R 777 /tmp/.ivy

# Pre-download the required packages to avoid downloading them at runtime
# RUN spark-submit \
#     --conf spark.jars.ivy=/tmp/.ivy \
#     --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
#     --jars /opt/spark/jars/hadoop-aws-3.3.4.jar \
#     --class dummy dummy.jar || true

COPY . .

CMD ["spark-submit","--properties-file","spark.conf","gold_processor.py"]
