"""
Train LightGBM model for AQI prediction.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
from datetime import datetime
import argparse

from data_loader import (
    DEFAULT_DATA_END,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    AirQualityDataLoader,
)


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE: Only calculate for non-zero values to avoid division issues
    mask = (y_true != 0) & (y_true > 0.1)  # Ignore values <= 0.1
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def train_model(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                target_col: str = 'aqi_next',
                model_type: str = 'global',
                country: str = None):
    """
    Train LightGBM model.
    
    Args:
        train_df: Training data
        val_df: Validation data
        target_col: Target column name
        model_type: 'global' (all countries) or 'country' (per country)
        country: Country code if model_type == 'country'
    
    Returns:
        Trained LightGBM model
    """
    # Feature columns (exclude target and metadata)
    exclude_cols = [
        target_col, 'aqi', 'datetime', 'location_id', 'location_name',
        'country', 'aqi_category', 'parameters', 'values',
        'year', 'month', 'day'  # Partition columns
    ]
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Ensure we have valid features
    feature_cols = [col for col in feature_cols if col in train_df.columns]
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col].fillna(0)
    
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[target_col].fillna(0)
    
    print(f"\nTraining features ({len(feature_cols)}):")
    for i, feat in enumerate(feature_cols[:15]):
        print(f"  {i+1}. {feat}")
    if len(feature_cols) > 15:
        print(f"  ... and {len(feature_cols) - 15} more")
    
    # LightGBM parameters (optimized for best performance)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 128,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'num_iterations': 5000,
        'early_stopping_rounds': 150,
        'verbose': -1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.3,
        'max_bin': 255,
        'force_row_wise': True,
        'num_threads': -1,
        'min_gain_to_split': 0.1
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    print(f"\nTraining {model_type} model" + (f" for {country}" if country else "") + "...")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=100)]
    )
    
    # Evaluate on validation set
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    metrics = calculate_metrics(y_val, y_pred)
    
    print(f"\nValidation Metrics:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE:  {metrics['mae']:.2f}")
    if not np.isnan(metrics['mape']):
        print(f"  MAPE: {metrics['mape']:.2f}%")
    else:
        print(f"  MAPE: N/A (insufficient non-zero values)")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    return model, metrics, feature_cols, params


def main():
    parser = argparse.ArgumentParser(description='Train LightGBM global model for AQI prediction')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD), default: None (all data)')
    parser.add_argument('--end-date', type=str, default=DEFAULT_DATA_END,
                       help=f'End date (YYYY-MM-DD), default: {DEFAULT_DATA_END}')
    parser.add_argument('--train-ratio', type=float, default=DEFAULT_TRAIN_RATIO,
                       help=f'Chronological training ratio, default: {DEFAULT_TRAIN_RATIO}')
    parser.add_argument('--val-ratio', type=float, default=DEFAULT_VAL_RATIO,
                       help=f'Chronological validation ratio, default: {DEFAULT_VAL_RATIO}')
    parser.add_argument('--test-ratio', type=float, default=DEFAULT_TEST_RATIO,
                       help=f'Chronological test ratio, default: {DEFAULT_TEST_RATIO}')
    parser.add_argument('--sample', type=float, default=None,
                       help='Sample fraction (for testing)')
    parser.add_argument('--include-pollutant-features', action='store_true',
                       help='Extract pollutant columns from parameters/values arrays')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--target', type=str, default='aqi_next',
                       help='Target column: aqi_next (next hour) or aqi (current)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LightGBM AQI Prediction Training (Global Model)")
    print("=" * 80)
    
    # Load data
    loader = AirQualityDataLoader()
    df = loader.load_gold_layer(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_frac=args.sample
    )
    
    # Create features
    print("\nCreating features...")
    df_features = loader.create_features(
        df,
        include_pollutant_features=args.include_pollutant_features,
    )
    
    # Create target: next hour AQI
    if args.target == 'aqi_next':
        print("Creating target: AQI next hour...")
        df_features['aqi_next'] = df_features.groupby('location_id')['aqi'].shift(-1)
        # Remove rows where target is NaN (last hour of each location)
        df_features = df_features.dropna(subset=['aqi_next'])
        target_col = 'aqi_next'
    else:
        target_col = 'aqi'
    
    # Split data
    print("\nSplitting data...")
    train_df, val_df, test_df = loader.split_train_val_test(
        df_features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        data_end=args.end_date,
    )
    
    # Remove rows with NaN target
    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col])
    
    # Train global model
    print("\nTraining global LightGBM model...")
    model, metrics, feature_cols, params = train_model(
        train_df, val_df, target_col=target_col, model_type='global'
    )
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'lightgbm_global.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics,
            'target_col': target_col,
            'train_date': datetime.now().isoformat(),
            'country_encoder': loader.country_encoder,
            'include_pollutant_features': args.include_pollutant_features,
        }, f)
    print(f"\n✓ Model saved to {model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  MAE:  {test_metrics['mae']:.2f}")
    if not np.isnan(test_metrics['mape']):
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
    else:
        print(f"  MAPE: N/A (insufficient non-zero values)")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    
    # Save evaluation results to file
    results_dir = os.path.join(args.output_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'lightgbm_evaluation.txt')
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LightGBM Model Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target: {target_col}\n\n")
        
        f.write("Validation Set Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"MAE:  {metrics['mae']:.4f}\n")
        if not np.isnan(metrics['mape']):
            f.write(f"MAPE: {metrics['mape']:.4f}%\n")
        else:
            f.write(f"MAPE: N/A\n")
        f.write(f"R²:   {metrics['r2']:.4f}\n\n")
        
        f.write("Test Set Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"MAE:  {test_metrics['mae']:.4f}\n")
        if not np.isnan(test_metrics['mape']):
            f.write(f"MAPE: {test_metrics['mape']:.4f}%\n")
        else:
            f.write(f"MAPE: N/A\n")
        f.write(f"R²:   {test_metrics['r2']:.4f}\n\n")
        
        f.write("Data Split:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Validation samples: {len(val_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n\n")
        
        f.write("Model Parameters:\n")
        f.write("-" * 40 + "\n")
        for key, value in params.items():
            if key != 'verbose':
                f.write(f"{key}: {value}\n")
        f.write(f"\nNumber of features: {len(feature_cols)}\n")
    
    print(f"\n✓ Evaluation results saved to {results_file}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
