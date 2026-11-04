FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY collect_data_realtime.py aqi_calculator.py ./

RUN pip install --no-cache-dir \
    requests \
    tqdm \
    kafka-python \
    pandas

CMD ["python", "collect_data_realtime.py","--interval", "300"]
