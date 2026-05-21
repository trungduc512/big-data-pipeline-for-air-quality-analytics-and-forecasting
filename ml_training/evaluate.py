"""
Evaluate trained models and generate visualizations.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import os
from data_loader import DEFAULT_DATA_END, AirQualityDataLoader


DEFAULT_TEST_START = "2025-08-20"


def load_model(model_path: str):
    """Load trained model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def evaluate_model(model_data: dict, test_df: pd.DataFrame, output_dir: str = "results"):
    """
    Evaluate model on test set and generate visualizations.
    
    Args:
        model_data: Dictionary with 'model', 'feature_cols', 'metrics'
        test_df: Test DataFrame
        output_dir: Output directory for plots
    """
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    target_col = model_data.get('target_col', 'aqi')

    missing_features = [col for col in feature_cols if col not in test_df.columns]
    if missing_features:
        print(f"Adding {len(missing_features)} missing feature column(s) with 0.0 defaults")
        for feature in missing_features:
            test_df[feature] = 0.0
    
    # Prepare test data
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]
    
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    else:
        y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    mask = (y_test != 0) & (y_test > 0.1)
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = np.nan
    
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    if not np.isnan(mape):
        print(f"MAPE: {mape:.2f}%")
    else:
        print(f"MAPE: N/A (insufficient non-zero values)")
    print(f"R²:   {r2:.4f}")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title(f'Predicted vs Actual AQI (R² = {r2:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=150)
    plt.close()
    
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted AQI')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=150)
    plt.close()
    
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        importances = model.feature_importance(importance_type='gain')
    else:
        try:
            booster = model.get_booster()
            importance_dict = booster.get_score(importance_type='gain')
            importances = [importance_dict.get(f'f{i}', 0) for i, f in enumerate(feature_cols)]
        except:
            importances = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.xlabel('Importance (Gain)')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    
    if 'location_id' in test_df.columns:
        sample_location = test_df['location_id'].iloc[0]
        location_data = test_df[test_df['location_id'] == sample_location].sort_values('datetime')
        
        if len(location_data) > 0:
            plt.figure(figsize=(14, 6))
            plt.plot(location_data['datetime'], location_data['aqi'], label='Actual', alpha=0.7)
            
            location_indices = location_data.index
            location_pred = y_pred[test_df.index.isin(location_indices)]
            if len(location_pred) == len(location_data):
                plt.plot(location_data['datetime'], location_pred, label='Predicted', alpha=0.7)
            
            plt.xlabel('Date')
            plt.ylabel('AQI')
            plt.title(f'Time Series: Location {sample_location}')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'time_series_sample.png'), dpi=150)
            plt.close()
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
    
    with open(os.path.join(output_dir, 'test_metrics.txt'), 'w') as f:
        f.write("Test Set Metrics\n")
        f.write("=" * 40 + "\n")
        for key, value in metrics.items():
            if key == 'mape' and np.isnan(value):
                f.write(f"{key.upper()}: N/A\n")
            else:
                f.write(f"{key.upper()}: {value:.4f}\n")
    
    print(f"\nResults saved to {output_dir}/")
    print(f"- predicted_vs_actual.png")
    print(f"- residuals.png")
    print(f"- feature_importance.png")
    print(f"- time_series_sample.png")
    print(f"- test_metrics.txt")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pkl)')
    parser.add_argument('--test-start', type=str, default=DEFAULT_TEST_START,
                       help=f'Test set start date, default: {DEFAULT_TEST_START}')
    parser.add_argument('--test-end', type=str, default=DEFAULT_DATA_END,
                       help=f'Test set end date, default: {DEFAULT_DATA_END}')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--include-pollutant-features', action='store_true',
                       help='Extract pollutant columns from parameters/values arrays')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model_data = load_model(args.model)
    
    # Load test data
    loader = AirQualityDataLoader()
    test_df = loader.load_gold_layer(
        start_date=args.test_start,
        end_date=args.test_end
    )
    
    # Create features
    print("Creating features...")
    include_pollutant_features = (
        args.include_pollutant_features
        or model_data.get('include_pollutant_features', False)
    )
    test_df = loader.create_features(
        test_df,
        include_pollutant_features=include_pollutant_features,
    )
    
    # Create target if needed
    target_col = model_data.get('target_col', 'aqi')
    if target_col == 'aqi_next' and 'aqi_next' not in test_df.columns:
        print("Creating target: AQI next hour...")
        test_df['aqi_next'] = test_df.groupby('location_id')['aqi'].shift(-1)
        test_df = test_df.dropna(subset=[target_col])
    
    # Evaluate
    evaluate_model(model_data, test_df, args.output_dir)


if __name__ == "__main__":
    main()
