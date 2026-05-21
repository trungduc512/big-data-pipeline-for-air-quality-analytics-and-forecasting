FROM apache/spark-py:latest

USER root

WORKDIR /opt/spark/work-dir

COPY batch/spark/batch_processor.py .
COPY batch/spark/spark.conf .

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV HADOOP_USER_NAME=spark
ENV SPARK_USER=spark
ENV USER=spark
ENV LOGNAME=spark
ENV HOME=/home/spark
ENV JAVA_TOOL_OPTIONS="-Duser.name=spark"
ENV SPARK_SUBMIT_OPTS="-Duser.name=spark"
ENV HADOOP_OPTS="-Duser.name=spark"
ENV PATH="/opt/spark/bin:${PATH}"

RUN set -eux; \
    python3 -m pip install --no-cache-dir pyarrow minio; \
    if ! getent group spark >/dev/null; then \
        groupadd -g 185 spark || groupadd spark; \
    fi; \
    if ! id -u spark >/dev/null 2>&1; then \
        useradd -u 185 -g spark -m -s /bin/bash spark || useradd -g spark -m -s /bin/bash spark; \
    fi; \
    mkdir -p /tmp/.ivy /opt/spark/work-dir /home/spark; \
    chown -R spark:spark /tmp/.ivy /opt/spark/work-dir /home/spark; \
    chmod -R 775 /tmp/.ivy /opt/spark/work-dir

USER spark

CMD ["spark-submit", "--properties-file", "spark.conf", "batch_processor.py"]
