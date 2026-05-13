"""
Script to collect air quality data from OpenAQ API v3
for alive monitoring stations in Southeast Asia and send to Kafka (real-time).

Usage:
    python collect_data_realtime.py --interval 300
"""

import requests
import argparse
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import time
import sys
import json
import os
from collections import defaultdict
from kafka import KafkaProducer
from kafka.errors import KafkaError
from aqi_calculator import calculate_aqi_for_pollutant, convert_unit_to_standard, get_aqi_category
import threading

# --- Constants ---
API_KEYS = []
BASE_URL = "https://api.openaq.org/v3"
ALIVE_THRESHOLD_DAYS = 7
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RAW = "openaq-raw-measurements"

# Southeast Asia countries
SOUTHEAST_ASIA_COUNTRIES = ["BN", "KH", "ID", "LA", "MY", "MM", "PH", "SG", "TH", "TL", "VN"]

# Parameters we're interested in
PARAMETERS = ["pm25", "pm10", "o3", "co", "so2", "no2"]

# API key rotation for load balancing
api_key_index = 0
api_key_lock = threading.Lock()
rate_limit_counts = defaultdict(int)
rate_limit_lock = threading.Lock()


def get_api_keys_from_env():
    """Load OpenAQ API keys from environment."""
    raw = os.environ.get("OPENAQ_API_KEYS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def get_next_api_key():
    """Get next API key in rotation."""
    global api_key_index
    with api_key_lock:
        if not API_KEYS:
            raise RuntimeError("No OpenAQ API keys provided. Set OPENAQ_API_KEYS.")
        key_idx = api_key_index
        key = API_KEYS[key_idx]
        api_key_index = (api_key_index + 1) % len(API_KEYS)
        return key_idx, key


def record_rate_limit(key_idx):
    """Track HTTP 429 occurrences by API key index."""
    with rate_limit_lock:
        rate_limit_counts[key_idx] += 1


def reset_rate_limit_counts():
    """Reset 429 counters for a new real-time cycle."""
    with rate_limit_lock:
        rate_limit_counts.clear()


def get_rate_limit_counts_snapshot():
    """Get a snapshot of 429 counters."""
    with rate_limit_lock:
        return dict(rate_limit_counts)


def get_alive_stations():
    """Get list of alive monitoring stations in Southeast Asia."""
    print("Fetching alive stations in Southeast Asia...")

    alive_threshold = datetime.now(timezone.utc) - timedelta(days=ALIVE_THRESHOLD_DAYS)
    alive_stations = []

    for country_code in tqdm(SOUTHEAST_ASIA_COUNTRIES, desc="Scanning countries"):
        page = 1
        session = requests.Session()
        current_key_idx, current_key = get_next_api_key()
        session.headers.update({"X-API-Key": current_key})

        while True:
            try:
                response = session.get(
                    f"{BASE_URL}/locations",
                    params={"iso": country_code, "limit": 1000, "page": page}
                )

                if response.status_code == 429:
                    record_rate_limit(current_key_idx)
                    time.sleep(2)
                    current_key_idx, current_key = get_next_api_key()
                    session.headers.update({"X-API-Key": current_key})
                    continue

                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                for station in results:
                    datetime_last = station.get("datetimeLast")
                    if datetime_last and datetime_last.get("utc"):
                        last_update = datetime.fromisoformat(datetime_last["utc"])
                        if last_update > alive_threshold:
                            sensors = station.get("sensors", [])
                            station_params = [s["parameter"]["name"] for s in sensors]

                            if any(p in station_params for p in ["pm25", "pm10"]):
                                alive_stations.append({
                                    "location_id": station["id"],
                                    "name": station.get("name", "Unknown"),
                                    "country": station["country"]["code"],
                                    "latitude": station["coordinates"]["latitude"],
                                    "longitude": station["coordinates"]["longitude"],
                                    "sensors": sensors
                                })

                if len(results) < 1000:
                    break
                page += 1
                time.sleep(0.1)

            except Exception as e:
                print(f"\nError fetching {country_code}: {e}")
                break

    print(f"\nFound {len(alive_stations)} alive stations with PM data")
    return alive_stations


def fetch_measurements_for_sensor(sensor_id, date_from, date_to):
    """Fetch hourly measurements for a specific sensor."""
    session = requests.Session()
    current_key_idx, current_key = get_next_api_key()
    session.headers.update({"X-API-Key": current_key})

    all_measurements = []
    page = 1
    max_pages = 100

    while page <= max_pages:
        try:
            response = session.get(
                f"{BASE_URL}/sensors/{sensor_id}/measurements/hourly",
                params={
                    "datetime_from": date_from.isoformat(),
                    "datetime_to": date_to.isoformat(),
                    "limit": 1000,
                    "page": page
                }
            )

            if response.status_code == 429:
                record_rate_limit(current_key_idx)
                time.sleep(1)
                current_key_idx, current_key = get_next_api_key()
                session.headers.update({"X-API-Key": current_key})
                continue

            if response.status_code != 200:
                break

            data = response.json()
            results = data.get("results", [])

            if not results:
                break

            for measurement in results:
                period = measurement.get("period", {})
                datetime_from = period.get("datetimeFrom", {})
                if datetime_from and datetime_from.get("utc"):
                    all_measurements.append({
                        "datetime": datetime_from["utc"],
                        "value": measurement.get("value")
                    })

            if len(results) < 1000:
                break
            page += 1
            time.sleep(0.1)

        except Exception as e:
            print(f"Error fetching sensor {sensor_id}: {e}")
            break

    return all_measurements


def send_to_kafka(producer, data, key=None):
    """Send data to Kafka topic, optionally partitioned by key."""
    try:
        message = json.dumps(data).encode("utf-8")
        key_bytes = None if key is None else str(key).encode("utf-8")
        producer.send(KAFKA_TOPIC_RAW, key=key_bytes, value=message)
    except KafkaError as e:
        print(f"Error sending to Kafka: {e}")


def process_station_data(station, producer, date_from, date_to):
    """Process all supported sensors for one station and send records to Kafka."""
    location_id = station["location_id"]
    records_sent = 0

    for sensor in station["sensors"]:
        param_name = sensor["parameter"]["name"]

        if param_name not in PARAMETERS:
            continue

        sensor_id = sensor["id"]
        measurements = fetch_measurements_for_sensor(
            sensor_id,
            date_from=date_from,
            date_to=date_to,
        )

        for m in measurements:
            value = m["value"]
            unit = sensor["parameter"]["units"]
            value_standard = convert_unit_to_standard(value, param_name, unit)

            aqi = calculate_aqi_for_pollutant(value_standard, param_name)
            aqi_category = get_aqi_category(aqi) if aqi else None

            message = {
                "datetime": m["datetime"],
                "location_id": location_id,
                "location_name": station["name"],
                "country": station["country"],
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "parameter": param_name,
                "value": value,
                "value_standard": value_standard,
                "unit": unit,
                "aqi": aqi,
                "aqi_category": aqi_category,
                "ingestion_timestamp": datetime.now(timezone.utc).isoformat()
            }

            send_to_kafka(producer, message, key=location_id)
            records_sent += 1

    return records_sent


def collect_realtime_data(stations, producer, interval_seconds=300):
    """Collect real-time data continuously."""
    print(f"\nStarting real-time data collection (interval: {interval_seconds}s)...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            reset_rate_limit_counts()
            date_to = datetime.now(timezone.utc)
            date_from = date_to - timedelta(hours=1)
            cycle_start = time.time()

            total_records = 0
            for station in stations:
                try:
                    total_records += process_station_data(
                        station,
                        producer,
                        date_from=date_from,
                        date_to=date_to,
                    )
                except Exception as e:
                    print(f"Error processing station: {e}")

            producer.flush()
            elapsed = max(time.time() - cycle_start, 1e-9)
            throughput = total_records / elapsed
            rate_limit_snapshot = get_rate_limit_counts_snapshot()
            affected_key_indexes = sorted(rate_limit_snapshot.keys())

            print(
                f"[{datetime.now()}] Sent {total_records} records in {elapsed:.2f}s "
                f"({throughput:.2f} records/s)."
            )
            if affected_key_indexes:
                print(
                    "429 detected on key indexes: "
                    f"{affected_key_indexes} | counts={rate_limit_snapshot}"
                )
            else:
                print("No 429 in this cycle.")

            print(f"Waiting {interval_seconds}s...")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nStopping real-time collection...")
        producer.flush()
        producer.close()


def main():
    parser = argparse.ArgumentParser(description="Collect air quality data from OpenAQ and send to Kafka (real-time)")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval in seconds for real-time mode (default: 300)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample only N stations for testing (optional)"
    )
    parser.add_argument(
        "--kafka-bootstrap",
        type=str,
        default=KAFKA_BOOTSTRAP_SERVERS,
        help=f"Kafka bootstrap servers (default: {KAFKA_BOOTSTRAP_SERVERS})"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OpenAQ Data Collection Script (Kafka Producer, Real-time)")
    print("=" * 80)
    print("Target region: Southeast Asia")
    print(f"Collection interval: {args.interval}s")
    print(f"Kafka broker: {args.kafka_bootstrap}")
    print(f"Topic: {KAFKA_TOPIC_RAW}")
    print("=" * 80)

    global API_KEYS
    API_KEYS = get_api_keys_from_env()
    if not API_KEYS:
        print("Missing OpenAQ API keys. Set OPENAQ_API_KEYS (comma-separated).")
        sys.exit(1)

    try:
        producer = KafkaProducer(
            bootstrap_servers=args.kafka_bootstrap,
            value_serializer=lambda v: v,
            acks="all",
            retries=3
        )
        print("Connected to Kafka successfully")
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        print("Make sure Kafka is running: docker-compose up -d kafka")
        sys.exit(1)

    stations = get_alive_stations()

    if not stations:
        print("No alive stations found!")
        sys.exit(1)

    if args.sample:
        stations = stations[:args.sample]
        print(f"\nSampling {args.sample} stations for testing")

    collect_realtime_data(stations, producer, args.interval)

    producer.close()
    print("\nCollection complete!")


if __name__ == "__main__":
    main()
