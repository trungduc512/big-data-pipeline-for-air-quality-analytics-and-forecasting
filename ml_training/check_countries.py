"""
Check countries, stations, sensors, and data statistics in Gold layer.
"""

from data_loader import AirQualityDataLoader
import pandas as pd
from datetime import datetime, timedelta

def main():
    print("=" * 80)
    print("THỐNG KÊ DỮ LIỆU - GOLD LAYER")
    print("=" * 80)
    
    loader = AirQualityDataLoader()
    
    # Load ALL data without date filter
    print("\nĐang tải dữ liệu từ Gold layer (không filter ngày)...")
    df = loader.load_gold_layer(
        start_date=None,
        end_date=None
    )
    
    print(f"\n{'='*80}")
    print("1. TỔNG SỐ BẢN GHI")
    print(f"{'='*80}")
    total_records = len(df)
    print(f"Tổng số bản ghi: {total_records:,}")
    print(f"Khoảng thời gian: {df['datetime'].min()} đến {df['datetime'].max()}")
    
    # Estimate size
    estimated_size_mb = (total_records * 200) / (1024**2)  # ~200 bytes per record
    estimated_size_gb = estimated_size_mb / 1024
    compressed_size_gb = estimated_size_gb * 0.2  # Parquet compression ~20%
    print(f"\nƯớc lượng kích thước:")
    print(f"  - Uncompressed: ~{estimated_size_mb:.2f} MB (~{estimated_size_gb:.2f} GB)")
    print(f"  - Compressed (Parquet): ~{compressed_size_gb:.2f} GB")
    
    # 5V Big Data Analysis
    print(f"\n{'='*80}")
    print("2. PHÂN TÍCH 5V BIG DATA")
    print(f"{'='*80}")
    print(f"Volume (Khối lượng): {total_records:,} records (~{compressed_size_gb:.2f} GB compressed)")
    print(f"Velocity (Tốc độ): Real-time ingestion từ OpenAQ API, batch processing hàng ngày")
    
    # Variety
    countries = sorted(df['country'].unique())
    locations = df['location_id'].nunique()
    parameters_count = 0
    for params in df['parameters'].dropna():
        if isinstance(params, list):
            parameters_count = max(parameters_count, len(params))
    
    print(f"Variety (Đa dạng):")
    print(f"  - {len(countries)} quốc gia: {', '.join(countries)}")
    print(f"  - {locations:,} trạm đo")
    print(f"  - {parameters_count} parameters (PM2.5, PM10, O3, CO, SO2, NO2)")
    
    print(f"Veracity (Độ tin cậy): Có nhiều vấn đề cần xử lý (missing values, invalid values, inconsistent units)")
    print(f"Value (Giá trị): ML training để dự đoán AQI, phân tích xu hướng chất lượng không khí")
    
    # Countries
    print(f"\n{'='*80}")
    print("3. THỐNG KÊ THEO QUỐC GIA")
    print(f"{'='*80}")
    country_counts = df.groupby('country').size().sort_values(ascending=False)
    for country, count in country_counts.items():
        country_data = df[df['country'] == country]
        date_min = country_data['datetime'].min()
        date_max = country_data['datetime'].max()
        locations_count = country_data['location_id'].nunique()
        print(f"{country}: {count:,} records, {locations_count:,} trạm, từ {date_min} đến {date_max}")
    
    # Expected countries
    expected = {'BN', 'KH', 'ID', 'LA', 'MY', 'MM', 'PH', 'SG', 'TH', 'TL', 'VN'}
    missing = expected - set(countries)
    if missing:
        print(f"\n⚠️  Thiếu quốc gia: {sorted(missing)}")
    
    # Alive stations (có data đến 2025-12-24 - latest data trong dataset)
    print(f"\n{'='*80}")
    print("4. TRẠM CÒN SỐNG (ALIVE STATIONS)")
    print(f"{'='*80}")
    print("Quy tắc lọc trạm sống:")
    print("  1. Trạm phải có dữ liệu trong 7 ngày gần nhất (từ datetimeLast)")
    print("  2. Trạm phải có ít nhất 1 sensor PM2.5 hoặc PM10")
    print("  3. Chỉ lấy từ 11 quốc gia Đông Nam Á")
    
    # Use latest date in dataset (2025-12-24) as reference
    latest_date = df['datetime'].max()
    seven_days_ago = latest_date - timedelta(days=7)
    
    # Get latest datetime per location
    latest_per_location = df.groupby('location_id')['datetime'].max()
    alive_locations = latest_per_location[latest_per_location >= seven_days_ago].index.tolist()
    
    print(f"\nTổng số trạm còn sống: {len(alive_locations):,}")
    print(f"(Có dữ liệu từ {seven_days_ago.strftime('%Y-%m-%d')} đến {latest_date.strftime('%Y-%m-%d')})")
    
    # Count sensors per location (from Gold layer - count unique parameters)
    print(f"\n{'='*80}")
    print("5. SENSORS CỦA CÁC TRẠM")
    print(f"{'='*80}")
    
    # Count sensors (parameters) per location
    location_sensors = {}
    for loc_id in alive_locations:
        loc_data = df[df['location_id'] == loc_id]
        if len(loc_data) > 0:
            # Get unique parameters from all records
            all_params = set()
            for params in loc_data['parameters'].dropna():
                if isinstance(params, list):
                    all_params.update(params)
            location_sensors[loc_id] = {
                'name': loc_data['location_name'].iloc[0] if 'location_name' in loc_data.columns else 'Unknown',
                'country': loc_data['country'].iloc[0],
                'sensors': sorted(all_params),
                'sensor_count': len(all_params)
            }
    
    total_sensors = sum(info['sensor_count'] for info in location_sensors.values())
    print(f"Tổng số trạm còn sống: {len(alive_locations):,}")
    if len(alive_locations) > 0:
        print(f"Tổng số sensors: {total_sensors:,}")
        print(f"(Trung bình: {total_sensors/len(alive_locations):.1f} sensors/trạm)")
    else:
        print(f"Tổng số sensors: {total_sensors:,}")
        print("(Không có trạm nào có dữ liệu trong 7 ngày gần nhất)")
    
    # Example: Vietnam stations
    print(f"\n{'='*80}")
    print("6. VÍ DỤ: VIỆT NAM (VN)")
    print(f"{'='*80}")
    
    vn_alive = [loc_id for loc_id in alive_locations 
                if location_sensors.get(loc_id, {}).get('country') == 'VN']
    
    print(f"Số trạm còn sống ở VN: {len(vn_alive)}")
    
    if vn_alive:
        print(f"\nDanh sách 5 trạm đầu tiên ở VN:")
        for i, loc_id in enumerate(vn_alive[:15], 1):
            info = location_sensors[loc_id]
            print(f"\n{i}. {info['name']} (Location ID: {loc_id})")
            print(f"   Số sensors: {info['sensor_count']}")
            print(f"   Danh sách sensors: {', '.join(info['sensors'])}")
            
            # Get sensor IDs from location data (if available in Gold layer)
            # Note: Gold layer doesn't have sensor IDs, only parameters
            # So we show parameters instead
            loc_data = df[df['location_id'] == loc_id]
            if len(loc_data) > 0:
                latest_record = loc_data.loc[loc_data['datetime'].idxmax()]
                params = latest_record.get('parameters', [])
                values = latest_record.get('values', [])
                if isinstance(params, list) and isinstance(values, list):
                    print(f"   Parameters và values mới nhất:")
                    for param, val in zip(params[:3], values[:3]):  # Show first 3
                        print(f"     - {param}: {val}")
    
    # Sample record
    print(f"\n{'='*80}")
    print("7. VÍ DỤ 1 BẢN GHI")
    print(f"{'='*80}")
    if len(df) > 0:
        sample = df.iloc[0]
        print(f"Location ID: {sample.get('location_id', 'N/A')}")
        print(f"Location Name: {sample.get('location_name', 'N/A')}")
        print(f"Country: {sample.get('country', 'N/A')}")
        print(f"Datetime: {sample.get('datetime', 'N/A')}")
        print(f"AQI: {sample.get('aqi', 'N/A')}")
        print(f"AQI Category: {sample.get('aqi_category', 'N/A')}")
        print(f"Parameters: {sample.get('parameters', 'N/A')}")
        print(f"Values: {sample.get('values', 'N/A')}")
    
    print(f"\n{'='*80}")
    print("HOÀN TẤT")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
