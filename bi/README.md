# Metabase BI

Metabase reads from ClickHouse after the `load-to-clickhouse-full` job has copied Gold Parquet data from MinIO into ClickHouse.

```text
Kafka -> Gold Spark -> MinIO /gold -> load-to-clickhouse-full -> ClickHouse -> Metabase
```

## Deploy

Apply ClickHouse first, then Metabase:

```bash
kubectl apply -n bigdata -f clickhouse/
kubectl apply -n bigdata -f bi/
```

After Gold data exists in MinIO, load it into ClickHouse:

```bash
kubectl delete job -n bigdata load-to-clickhouse-full --ignore-not-found
kubectl apply -n bigdata -f jobs/load_to_clickhouse/load-to-clickhouse-full-job.yaml
kubectl logs -n bigdata job/load-to-clickhouse-full -f
```

Open Metabase locally:

```bash
kubectl port-forward -n bigdata svc/metabase 3000:3000
```

Then browse to `http://localhost:3000`.

Metabase stores only its application metadata in `metabase-pvc`; BI queries still run directly against ClickHouse.

Metabase is a JVM application. The local manifest requests `768Mi` and limits memory to `2Gi`, with the Java heap capped at `1Gi` to leave room for non-heap JVM memory and startup overhead.

## ClickHouse Connection

Use these values when adding the database in Metabase:

```text
Database type: ClickHouse
Display name: Air Quality ClickHouse
Host: clickhouse
Port: 8123
Username: default
Password: default123
Databases: air_quality
SSL: off
```

## Useful Starting Questions

```sql
SELECT
    datetime,
    country,
    location_name,
    aqi,
    parameters,
    values,
    aqi_category
FROM air_quality.gold_hourly_aqi
ORDER BY datetime DESC
LIMIT 100;
```

```sql
SELECT
    country,
    location_name,
    max(datetime) AS latest_window,
    argMax(aqi, datetime) AS latest_aqi,
    argMax(aqi_category, datetime) AS latest_category
FROM air_quality.gold_hourly_aqi
GROUP BY country, location_name
ORDER BY latest_aqi DESC
LIMIT 50;
```
