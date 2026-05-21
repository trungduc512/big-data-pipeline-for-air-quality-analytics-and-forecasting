"""
Data Loader for ML Training
Loads Gold layer data from MinIO (Parquet) and prepares for training.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import s3fs
import pyarrow.parquet as pq
from typing import Optional, Tuple
import os
from sklearn.preprocessing import LabelEncoder


DEFAULT_DATA_END = "2026-02-28"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
POLLUTANTS = ['pm25', 'pm10', 'o3', 'co', 'so2', 'no2']


def end_date_exclusive(value: str) -> pd.Timestamp:
    """Return an exclusive end boundary for an inclusive date string."""
    timestamp = pd.to_datetime(value)
    if timestamp == timestamp.normalize():
        return timestamp + pd.Timedelta(days=1)
    return timestamp


class AirQualityDataLoader:
    """Load and preprocess air quality data for ML training from MinIO."""
    
    def __init__(self, minio_endpoint: str = None):
        """
        Initialize data loader.
        
        Args:
            minio_endpoint: MinIO endpoint URL (auto-detect if None)
        """
        # Auto-detect endpoint: try minio (Docker) first, then localhost
        if minio_endpoint is None:
            import socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('minio', 9000))
                sock.close()
                minio_endpoint = "http://minio:9000" if result == 0 else "http://localhost:9000"
            except:
                minio_endpoint = "http://localhost:9000"
        
        self.minio_endpoint = minio_endpoint
        print(f"Connecting to MinIO at: {minio_endpoint}")
        
        self.fs = s3fs.S3FileSystem(
            key='minioadmin',
            secret='minioadmin123',
            endpoint_url=minio_endpoint,
            use_ssl=False
        )
        self.country_encoder = LabelEncoder()
    
    def load_gold_layer(self, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       countries: Optional[list] = None,
                       sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Load Gold layer data (hourly aggregated AQI) from MinIO.
        
        Args:
            start_date: Start date (YYYY-MM-DD), default: all
            end_date: End date (YYYY-MM-DD), default: all
            countries: List of country codes to filter, default: all
            sample_frac: Fraction to sample (for testing), default: None (all data)
        
        Returns:
            DataFrame with columns: datetime, location_id, country, aqi, etc.
        """
        bucket = 'historical-air-quality-data'
        gold_path = f'{bucket}/gold/'
        
        # Get all parquet files
        files = sorted(self.fs.glob(f'{gold_path}**/*.parquet'))
        
        if not files:
            raise ValueError("No parquet files found in MinIO gold layer")
        
        print(f"Found {len(files)} parquet files. Loading...")
        
        # Load files with progress tracking
        dfs = []
        for i, file_path in enumerate(files):
            if sample_frac and np.random.random() > sample_frac:
                continue
            
            try:
                with self.fs.open(file_path, 'rb') as f:
                    df = pq.read_table(f).to_pandas()
                    dfs.append(df)
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
                continue
            
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{len(files)} files...")
        
        if not dfs:
            raise ValueError("No data loaded from parquet files")
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Filter by date
        if start_date:
            df = df[df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['datetime'] < end_date_exclusive(end_date)]
        
        # Filter by country
        if countries:
            df = df[df['country'].isin(countries)]
        
        # Sort by datetime and location (critical for lag features)
        df = df.sort_values(['location_id', 'datetime']).reset_index(drop=True)
        
        print(f"Loaded {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Countries: {sorted(df['country'].unique())}")
        print(f"Locations: {df['location_id'].nunique()} unique locations")
        
        return df
    
    def create_features(
        self,
        df: pd.DataFrame,
        include_pollutant_features: bool = False,
        drop_array_columns: bool = True,
    ) -> pd.DataFrame:
        """
        Create features for ML training.
        
        Args:
            df: Raw DataFrame from load_gold_layer()
            include_pollutant_features: Extract pollutant columns from parameters/values arrays.
            drop_array_columns: Drop raw array columns after optional extraction to save memory.
        
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        print("Creating time features...")
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
        
        print("Creating lag features...")
        df['aqi_lag_1h'] = df.groupby('location_id')['aqi'].shift(1)
        df['aqi_lag_24h'] = df.groupby('location_id')['aqi'].shift(24)
        df['aqi_lag_168h'] = df.groupby('location_id')['aqi'].shift(168)
        
        print("Creating rolling statistics...")
        df['aqi_mean_7d'] = df.groupby('location_id')['aqi'].transform(
            lambda x: x.rolling(168, min_periods=1).mean()
        )
        df['aqi_std_7d'] = df.groupby('location_id')['aqi'].transform(
            lambda x: x.rolling(168, min_periods=1).std()
        )
        df['aqi_max_7d'] = df.groupby('location_id')['aqi'].transform(
            lambda x: x.rolling(168, min_periods=1).max()
        )
        df['aqi_min_7d'] = df.groupby('location_id')['aqi'].transform(
            lambda x: x.rolling(168, min_periods=1).min()
        )
        df['aqi_mean_30d'] = df.groupby('location_id')['aqi'].transform(
            lambda x: x.rolling(720, min_periods=1).mean()
        )
        
        print("Encoding country...")
        df['country_encoded'] = self.country_encoder.fit_transform(df['country'])

        if include_pollutant_features:
            print("Extracting pollutant values from arrays...")
            pollutant_df = df.apply(
                self._extract_pollutant_values,
                axis=1,
                result_type='expand',
            )
            pollutant_df = pollutant_df.reindex(columns=POLLUTANTS, fill_value=0.0)
            pollutant_df = pollutant_df.fillna(0.0).astype('float32')
            df = pd.concat([df, pollutant_df], axis=1)
        else:
            print("Skipping pollutant array extraction (use --include-pollutant-features to enable)")

        if drop_array_columns:
            df = df.drop(columns=['parameters', 'values'], errors='ignore')
        
        return df

    def _extract_pollutant_values(self, row) -> dict[str, float]:
        """Extract all pollutant values from one parameters/values row."""
        params = row.get('parameters')
        values = row.get('values')
        extracted = {pollutant: 0.0 for pollutant in POLLUTANTS}

        if not isinstance(params, (list, tuple, np.ndarray)) or not isinstance(
            values, (list, tuple, np.ndarray)
        ):
            return extracted

        for param, value in zip(params, values):
            if param in extracted:
                try:
                    extracted[param] = float(value)
                except (TypeError, ValueError):
                    extracted[param] = 0.0

        return extracted
    
    def split_train_val_test(self, 
                            df: pd.DataFrame,
                            train_end: Optional[str] = None,
                            val_end: Optional[str] = None,
                            train_ratio: float = DEFAULT_TRAIN_RATIO,
                            val_ratio: float = DEFAULT_VAL_RATIO,
                            test_ratio: float = DEFAULT_TEST_RATIO,
                            data_end: str = DEFAULT_DATA_END) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets (time-based).
        
        Args:
            df: DataFrame with datetime column
            train_end: End date for training set (YYYY-MM-DD). If omitted, computed from ratios.
            val_end: End date for validation set (YYYY-MM-DD). If omitted, computed from ratios.
            train_ratio: Chronological share for training when dates are not provided.
            val_ratio: Chronological share for validation when dates are not provided.
            test_ratio: Chronological share for testing when dates are not provided.
            data_end: Last available data date, inclusive.
        
        Returns:
            train_df, val_df, test_df
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'] < end_date_exclusive(data_end)]

        if df.empty:
            raise ValueError("No records available before the configured data_end date")

        if train_end is None or val_end is None:
            ratio_total = train_ratio + val_ratio + test_ratio
            if ratio_total <= 0:
                raise ValueError("Split ratios must sum to a positive number")

            train_share = train_ratio / ratio_total
            val_share = val_ratio / ratio_total
            available_days = pd.date_range(
                df['datetime'].min().normalize(),
                pd.to_datetime(data_end).normalize(),
                freq='D',
            )
            if len(available_days) < 3:
                raise ValueError("Need at least 3 days of data to create train/val/test splits")

            train_days = max(1, int(len(available_days) * train_share))
            val_days = max(1, int(len(available_days) * val_share))
            if train_days + val_days >= len(available_days):
                val_days = max(1, len(available_days) - train_days - 1)

            train_end_ts = available_days[train_days - 1]
            val_end_ts = available_days[train_days + val_days - 1]
        else:
            train_end_ts = pd.to_datetime(train_end)
            val_end_ts = pd.to_datetime(val_end)
        
        train_df = df[df['datetime'] < end_date_exclusive(train_end_ts.strftime('%Y-%m-%d'))]
        val_df = df[
            (df['datetime'] >= end_date_exclusive(train_end_ts.strftime('%Y-%m-%d'))) &
            (df['datetime'] < end_date_exclusive(val_end_ts.strftime('%Y-%m-%d')))
        ]
        test_df = df[
            (df['datetime'] >= end_date_exclusive(val_end_ts.strftime('%Y-%m-%d'))) &
            (df['datetime'] < end_date_exclusive(data_end))
        ]

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                "Empty train/validation/test split. "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )
        
        print(
            "Split policy: chronological "
            f"{train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}, data end {data_end}"
        )
        print(f"Train: {len(train_df)} records ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
        print(f"Val:   {len(val_df)} records ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
        print(f"Test:  {len(test_df)} records ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
        
        return train_df, val_df, test_df


def main():
    """Example usage."""
    loader = AirQualityDataLoader()
    
    # Load data
    df = loader.load_gold_layer(
        start_date="2020-01-01",
        end_date=DEFAULT_DATA_END
    )
    
    # Create features
    df_features = loader.create_features(df)
    
    # Split data
    train_df, val_df, test_df = loader.split_train_val_test(df_features)
    
    print("\nFeature columns:", list(df_features.columns))
    print("\nSample data:")
    print(df_features.head())


if __name__ == "__main__":
    main()
