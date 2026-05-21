"""
Historical OpenAQ backfill job that writes Bronze Parquet directly to MinIO.

This path is intended for historical ML backfills where message replay semantics
are not needed. It preserves the Bronze schema consumed by batch/spark/batch_processor.py.
"""

import argparse
import io
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from typing import Iterable
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
import requests
from minio import Minio
from minio.deleteobjects import DeleteObject
from tqdm import tqdm

from aqi_calculator import (
    calculate_aqi_for_pollutant,
    convert_unit_to_standard,
    get_aqi_category,
)


BASE_URL = "https://api.openaq.org/v3"
SOUTHEAST_ASIA_COUNTRIES = ["BN", "KH", "ID", "LA", "MY", "MM", "PH", "SG", "TH", "TL", "VN"]
PARAMETERS = ["pm25", "pm10", "o3", "co", "so2", "no2"]
AUTH_STATUS_CODES = {401, 403}
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_TRANSIENT_RETRIES = 5

api_key_index = 0
api_key_lock = threading.Lock()
rate_limit_counts = defaultdict(int)
rate_limit_lock = threading.Lock()


@dataclass(frozen=True)
class Sensor:
    sensor_id: int
    parameter: str
    unit: str


@dataclass(frozen=True)
class Station:
    location_id: int
    name: str
    country: str
    latitude: float | None
    longitude: float | None
    sensors: tuple[Sensor, ...]


@dataclass(frozen=True)
class StationFetchResult:
    rows: list[dict]
    errors: int = 0


def load_api_keys() -> list[str]:
    """Load OpenAQ API keys from env."""
    env_value = os.environ.get("OPENAQ_API_KEYS", "").strip()
    if env_value:
        keys = [key.strip() for key in env_value.split(",") if key.strip()]
        if keys:
            return keys
    return []


API_KEYS = load_api_keys()


def get_next_api_key() -> tuple[int, str]:
    """Get the next OpenAQ API key in rotation."""
    global api_key_index
    if not API_KEYS:
        raise RuntimeError("No OpenAQ API keys provided. Set OPENAQ_API_KEYS.")
    with api_key_lock:
        key_idx = api_key_index
        key = API_KEYS[key_idx]
        api_key_index = (api_key_index + 1) % len(API_KEYS)
        return key_idx, key


def record_rate_limit(key_idx: int) -> None:
    with rate_limit_lock:
        rate_limit_counts[key_idx] += 1


def get_rate_limit_counts_snapshot() -> dict[int, int]:
    with rate_limit_lock:
        return dict(rate_limit_counts)


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD") from exc


def utc_start(day: date) -> datetime:
    return datetime.combine(day, datetime_time.min, tzinfo=timezone.utc)


def iter_days(start_day: date, end_day: date) -> Iterable[date]:
    current = start_day
    while current <= end_day:
        yield current
        current += timedelta(days=1)


def iter_date_chunks(start_day: date, end_day: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    current = start_day
    while current <= end_day:
        chunk_end = min(end_day, current + timedelta(days=chunk_days - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def resolve_date_range(args: argparse.Namespace) -> tuple[date, date]:
    if args.date_from and args.date_to:
        start_day = args.date_from
        end_day = args.date_to
    elif args.date_from:
        start_day = args.date_from
        end_day = start_day + timedelta(days=args.days - 1)
    elif args.date_to:
        end_day = args.date_to
        start_day = end_day - timedelta(days=args.days - 1)
    else:
        end_day = datetime.now(timezone.utc).date()
        start_day = end_day - timedelta(days=args.days - 1)

    if args.days < 1:
        raise ValueError("--days must be greater than zero")
    if start_day > end_day:
        raise ValueError("--date-from must be on or before --date-to")
    return start_day, end_day


def create_openaq_session() -> tuple[requests.Session, int]:
    key_idx, key = get_next_api_key()
    session = requests.Session()
    session.headers.update({"X-API-Key": key})
    return session, key_idx


def rotate_session_key(session: requests.Session) -> int:
    key_idx, key = get_next_api_key()
    session.headers.update({"X-API-Key": key})
    return key_idx


def get_openaq(
    session: requests.Session,
    current_key_idx: int,
    url: str,
    params: dict,
    timeout: int,
    context: str,
) -> tuple[requests.Response, int]:
    attempts = 0
    auth_failures = 0

    while True:
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            attempts += 1
            if attempts > MAX_TRANSIENT_RETRIES:
                raise RuntimeError(f"OpenAQ request failed while {context}: {exc}") from exc
            sleep_seconds = min(60, 2**attempts)
            print(
                f"Retrying OpenAQ request while {context} after request error: {exc}; "
                f"attempt {attempts}/{MAX_TRANSIENT_RETRIES}"
            )
            time.sleep(sleep_seconds)
            continue

        if response.status_code in AUTH_STATUS_CODES:
            print(
                f"OpenAQ API key index {current_key_idx} was rejected with "
                f"HTTP {response.status_code}"
            )
            auth_failures += 1
            if auth_failures >= len(API_KEYS):
                raise RuntimeError(
                    f"OpenAQ authentication failed for all {len(API_KEYS)} configured API key(s) "
                    f"while {context}. Refresh the openaq-api-keys Secret."
                )
            print("Rotating OpenAQ API key")
            current_key_idx = rotate_session_key(session)
            continue

        if response.status_code == 429:
            record_rate_limit(current_key_idx)
            current_key_idx = rotate_session_key(session)

        if response.status_code not in TRANSIENT_STATUS_CODES:
            return response, current_key_idx

        attempts += 1
        if attempts > MAX_TRANSIENT_RETRIES:
            return response, current_key_idx

        sleep_seconds = min(60, 2**attempts)
        print(
            f"Retrying OpenAQ request while {context} after HTTP {response.status_code}; "
            f"attempt {attempts}/{MAX_TRANSIENT_RETRIES}"
        )
        time.sleep(sleep_seconds)


def parse_openaq_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def location_overlaps_range(location: dict, start_day: date, end_day: date) -> bool:
    """Skip stations whose advertised data range does not overlap the backfill."""
    first_value = (location.get("datetimeFirst") or {}).get("utc")
    last_value = (location.get("datetimeLast") or {}).get("utc")
    first_dt = parse_openaq_datetime(first_value)
    last_dt = parse_openaq_datetime(last_value)

    start_dt = utc_start(start_day)
    end_dt = utc_start(end_day + timedelta(days=1))

    if last_dt and last_dt < start_dt:
        return False
    if first_dt and first_dt >= end_dt:
        return False
    return True


def sensor_from_payload(sensor: dict) -> Sensor | None:
    parameter = sensor.get("parameter") or {}
    parameter_name = parameter.get("name") or parameter.get("displayName")
    if parameter_name not in PARAMETERS:
        return None

    sensor_id = sensor.get("id")
    if sensor_id is None:
        return None

    unit = parameter.get("units") or parameter.get("unit") or ""
    return Sensor(sensor_id=int(sensor_id), parameter=parameter_name, unit=unit)


def station_from_payload(location: dict, start_day: date, end_day: date) -> Station | None:
    if not location_overlaps_range(location, start_day, end_day):
        return None

    sensors = tuple(
        parsed
        for parsed in (sensor_from_payload(sensor) for sensor in location.get("sensors", []))
        if parsed is not None
    )
    if not sensors:
        return None

    coordinates = location.get("coordinates") or {}
    country = location.get("country") or {}
    country_code = country.get("code") if isinstance(country, dict) else country
    location_id = location.get("id")
    if location_id is None:
        return None

    return Station(
        location_id=int(location_id),
        name=location.get("name") or "Unknown",
        country=country_code or "",
        latitude=coordinates.get("latitude"),
        longitude=coordinates.get("longitude"),
        sensors=sensors,
    )


def fetch_locations(start_day: date, end_day: date) -> list[Station]:
    """Fetch Southeast Asia locations that have at least one supported sensor."""
    stations: list[Station] = []
    print("Fetching OpenAQ locations for Southeast Asia...")

    for country_code in tqdm(SOUTHEAST_ASIA_COUNTRIES, desc="Scanning countries"):
        session, current_key_idx = create_openaq_session()
        page = 1

        while True:
            response, current_key_idx = get_openaq(
                session,
                current_key_idx,
                f"{BASE_URL}/locations",
                {"iso": country_code, "limit": 1000, "page": page},
                30,
                f"fetching locations for {country_code}, page {page}",
            )

            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                break

            for location in results:
                station = station_from_payload(location, start_day, end_day)
                if station:
                    stations.append(station)

            if len(results) < 1000:
                break
            page += 1
            time.sleep(0.1)

    print(f"Found {len(stations)} locations with supported sensors")
    return stations


def fetch_measurements_for_sensor(
    sensor: Sensor, start_day: date, end_day: date
) -> tuple[list[dict], int]:
    session, current_key_idx = create_openaq_session()
    page = 1
    rows: list[dict] = []
    errors = 0

    date_from = utc_start(start_day)
    date_to = utc_start(end_day + timedelta(days=1))

    while True:
        context = f"fetching sensor {sensor.sensor_id} from {start_day} to {end_day}, page {page}"
        try:
            response, current_key_idx = get_openaq(
                session,
                current_key_idx,
                f"{BASE_URL}/sensors/{sensor.sensor_id}/measurements/hourly",
                {
                    "datetime_from": date_from.isoformat(),
                    "datetime_to": date_to.isoformat(),
                    "limit": 1000,
                    "page": page,
                },
                60,
                context,
            )
        except RuntimeError as exc:
            print(f"Error {context}: {exc}")
            return rows, errors + 1

        if response.status_code != 200:
            print(
                f"Error fetching sensor {sensor.sensor_id} from {start_day} to {end_day}: "
                f"HTTP {response.status_code}"
            )
            return rows, errors + 1

        results = response.json().get("results", [])
        if not results:
            break

        for measurement in results:
            period = measurement.get("period") or {}
            datetime_from = period.get("datetimeFrom") or {}
            measurement_time = datetime_from.get("utc")
            if measurement_time:
                rows.append({"datetime": measurement_time, "value": measurement.get("value")})

        if len(results) < 1000:
            break
        page += 1
        time.sleep(0.1)

    return rows, errors


def process_station_for_range(
    station: Station,
    start_day: date,
    end_day: date,
    ingestion_timestamp: str,
    request_chunk_days: int,
) -> StationFetchResult:
    rows: list[dict] = []
    errors = 0

    for sensor in station.sensors:
        measurements: list[dict] = []
        for chunk_start, chunk_end in iter_date_chunks(start_day, end_day, request_chunk_days):
            chunk_measurements, sensor_errors = fetch_measurements_for_sensor(
                sensor, chunk_start, chunk_end
            )
            measurements.extend(chunk_measurements)
            errors += sensor_errors

        for measurement in measurements:
            value = measurement["value"]
            value_standard = (
                convert_unit_to_standard(value, sensor.parameter, sensor.unit)
                if value is not None
                else None
            )
            aqi = calculate_aqi_for_pollutant(value_standard, sensor.parameter)
            aqi_category = get_aqi_category(aqi) if aqi is not None else None

            rows.append(
                {
                    "datetime": measurement["datetime"],
                    "location_id": station.location_id,
                    "location_name": station.name,
                    "country": station.country,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "parameter": sensor.parameter,
                    "value": value,
                    "value_standard": value_standard,
                    "unit": sensor.unit,
                    "aqi": aqi,
                    "aqi_category": aqi_category,
                    "ingestion_timestamp": ingestion_timestamp,
                }
            )

    return StationFetchResult(rows=rows, errors=errors)


def collect_range_rows(
    stations: list[Station],
    start_day: date,
    end_day: date,
    workers: int,
    request_chunk_days: int,
) -> tuple[list[dict], int]:
    ingestion_timestamp = datetime.now(timezone.utc).isoformat()
    worker_count = max(1, min(workers, len(stations)))
    all_rows: list[dict] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                process_station_for_range,
                station,
                start_day,
                end_day,
                ingestion_timestamp,
                request_chunk_days,
            )
            for station in stations
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Backfilling {start_day} to {end_day}",
        ):
            result = future.result()
            all_rows.extend(result.rows)
            errors += result.errors

    return all_rows, errors


def collect_day_rows(stations: list[Station], day: date, workers: int) -> tuple[list[dict], int]:
    return collect_range_rows(stations, day, day, workers, request_chunk_days=1)


def rows_by_day(rows: list[dict]) -> dict[date, list[dict]]:
    grouped: dict[date, list[dict]] = defaultdict(list)
    for row in rows:
        row_datetime = parse_openaq_datetime(row.get("datetime"))
        if row_datetime:
            grouped[row_datetime.date()].append(row)
    return grouped


def parse_minio_endpoint(endpoint: str) -> tuple[str, bool]:
    parsed = urlparse(endpoint)
    if parsed.scheme:
        return parsed.netloc or parsed.path, parsed.scheme == "https"
    return endpoint, False


def create_minio_client(endpoint: str, access_key: str, secret_key: str) -> Minio:
    minio_endpoint, secure = parse_minio_endpoint(endpoint)
    return Minio(minio_endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"Created bucket: {bucket}")


def bronze_partition_prefix(day: date) -> str:
    return f"bronze/year={day.year}/month={day.month}/day={day.day}/"


def delete_partition(client: Minio, bucket: str, prefix: str) -> int:
    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    if not objects:
        return 0

    delete_errors = list(
        client.remove_objects(bucket, [DeleteObject(obj.object_name) for obj in objects])
    )
    if delete_errors:
        messages = ", ".join(f"{err.object_name}: {err.message}" for err in delete_errors)
        raise RuntimeError(f"Failed deleting partition objects under {prefix}: {messages}")
    return len(objects)


def bronze_parquet_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("datetime", pa.timestamp("us")),
            pa.field("location_id", pa.int64()),
            pa.field("location_name", pa.string()),
            pa.field("country", pa.string()),
            pa.field("latitude", pa.float64()),
            pa.field("longitude", pa.float64()),
            pa.field("parameter", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("value_standard", pa.float64()),
            pa.field("unit", pa.string()),
            pa.field("aqi", pa.float64()),
            pa.field("aqi_category", pa.string()),
            pa.field("ingestion_timestamp", pa.timestamp("us")),
        ]
    )


def rows_to_parquet_bytes(rows: list[dict]) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = bronze_parquet_schema()
    df = pd.DataFrame(rows)
    df = df.reindex(columns=schema.names)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    df["ingestion_timestamp"] = pd.to_datetime(
        df["ingestion_timestamp"], utc=True
    ).dt.tz_convert(None)
    df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").astype("float64")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").astype("float64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    df["value_standard"] = pd.to_numeric(df["value_standard"], errors="coerce").astype("float64")
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").astype("float64")

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    return buffer.getvalue()


def write_day_partition(client: Minio, bucket: str, day: date, rows: list[dict]) -> tuple[int, str | None]:
    prefix = bronze_partition_prefix(day)
    if not rows:
        deleted = delete_partition(client, bucket, prefix)
        return deleted, None

    parquet_bytes = rows_to_parquet_bytes(rows)
    object_name = f"{prefix}part-{day.strftime('%Y%m%d')}-{uuid4().hex}.parquet"
    deleted = delete_partition(client, bucket, prefix)
    client.put_object(
        bucket,
        object_name,
        io.BytesIO(parquet_bytes),
        length=len(parquet_bytes),
        content_type="application/octet-stream",
    )
    return deleted, object_name


def print_run_summary(start_day: date, end_day: date, args: argparse.Namespace) -> None:
    print("=" * 80)
    print("OpenAQ Historical Backfill to MinIO Bronze")
    print("=" * 80)
    print(f"Date range: {start_day} to {end_day} inclusive")
    print(f"MinIO endpoint: {args.minio_endpoint}")
    print(f"Bucket: {args.bucket}")
    print(f"Workers: {args.workers}")
    print(f"Fetch mode: {args.fetch_mode}")
    print(f"Request chunk days: {args.request_chunk_days}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill OpenAQ historical data directly to MinIO Bronze Parquet"
    )
    parser.add_argument("--date-from", type=parse_date, default=None, help="Start date, YYYY-MM-DD")
    parser.add_argument("--date-to", type=parse_date, default=None, help="End date, YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=1, help="Number of days to backfill")
    parser.add_argument("--sample", type=int, default=None, help="Use only the first N stations")
    parser.add_argument("--workers", type=int, default=8, help="Max worker threads")
    parser.add_argument(
        "--request-chunk-days",
        type=int,
        default=int(os.environ.get("BACKFILL_REQUEST_CHUNK_DAYS", "7")),
        help="Number of days per OpenAQ measurement request in range mode",
    )
    parser.add_argument(
        "--fetch-mode",
        choices=["range", "day"],
        default=os.environ.get("BACKFILL_FETCH_MODE", "range"),
        help="Fetch each sensor once for the whole range, or one day at a time",
    )
    parser.add_argument(
        "--minio-endpoint",
        default=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        help="MinIO endpoint, with or without http(s) scheme",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("MINIO_BUCKET", "historical-air-quality-data"),
        help="MinIO bucket name",
    )
    parser.add_argument(
        "--minio-access-key",
        default=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        help="MinIO access key",
    )
    parser.add_argument(
        "--minio-secret-key",
        default=os.environ.get("MINIO_SECRET_KEY", "minioadmin123"),
        help="MinIO secret key",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and count rows without writing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if not API_KEYS:
            raise RuntimeError(
                "No OpenAQ API keys configured. Set OPENAQ_API_KEYS in the openaq-api-keys Secret."
            )

        start_day, end_day = resolve_date_range(args)
        if args.request_chunk_days < 1:
            raise ValueError("--request-chunk-days must be greater than zero")
        print_run_summary(start_day, end_day, args)

        stations = fetch_locations(start_day, end_day)
        if args.sample:
            stations = stations[: args.sample]
            print(f"Sampling first {len(stations)} station(s)")

        if not stations:
            print("No matching stations found; nothing to backfill.")
            return 0

        minio_client = None
        if not args.dry_run:
            minio_client = create_minio_client(
                args.minio_endpoint,
                args.minio_access_key,
                args.minio_secret_key,
            )
            ensure_bucket(minio_client, args.bucket)

        total_rows = 0
        if args.fetch_mode == "range":
            rows, errors = collect_range_rows(
                stations,
                start_day,
                end_day,
                args.workers,
                args.request_chunk_days,
            )
            if errors:
                raise RuntimeError(
                    f"Encountered {errors} fetch error(s) for {start_day} to {end_day}; "
                    "partitions were not overwritten."
                )
            total_rows += len(rows)
            grouped_rows = rows_by_day(rows)

            for day in iter_days(start_day, end_day):
                day_rows = grouped_rows.get(day, [])
                prefix = bronze_partition_prefix(day)

                if args.dry_run:
                    print(f"[dry-run] {day}: {len(day_rows)} rows for s3a://{args.bucket}/{prefix}")
                    continue

                assert minio_client is not None
                deleted, object_name = write_day_partition(minio_client, args.bucket, day, day_rows)
                if object_name:
                    print(f"{day}: wrote {len(day_rows)} rows to s3a://{args.bucket}/{object_name}")
                else:
                    print(
                        f"{day}: no rows; cleared {deleted} object(s) "
                        f"from s3a://{args.bucket}/{prefix}"
                    )
        else:
            for day in iter_days(start_day, end_day):
                rows, errors = collect_day_rows(stations, day, args.workers)
                if errors:
                    raise RuntimeError(
                        f"Encountered {errors} fetch error(s) for {day}; "
                        "partition was not overwritten."
                    )

                total_rows += len(rows)
                prefix = bronze_partition_prefix(day)

                if args.dry_run:
                    print(f"[dry-run] {day}: {len(rows)} rows for s3a://{args.bucket}/{prefix}")
                    continue

                assert minio_client is not None
                deleted, object_name = write_day_partition(minio_client, args.bucket, day, rows)
                if object_name:
                    print(f"{day}: wrote {len(rows)} rows to s3a://{args.bucket}/{object_name}")
                else:
                    print(
                        f"{day}: no rows; cleared {deleted} object(s) "
                        f"from s3a://{args.bucket}/{prefix}"
                    )

        rate_limits = get_rate_limit_counts_snapshot()
        print("=" * 80)
        print(f"Backfill complete. Total rows: {total_rows}")
        if rate_limits:
            print(f"OpenAQ 429 counts by API key index: {rate_limits}")
        print("=" * 80)
        return 0
    except Exception as exc:
        print(f"Backfill failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
