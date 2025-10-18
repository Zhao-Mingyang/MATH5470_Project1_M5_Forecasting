"""
m5_LGBM_cycle_encoding_conformal_prediction.py
============================================================

Enhanced version combining preprocessing.py features with the 
cycle_encoding_conformal_prediction.py implementation
- Added mean and std encodings from preprocessing.py
- Updated lag structure to match preprocessing.py (28-42)
- Added specific rolling window combinations from preprocessing.py
- Changed to Tweedie objective with variance_power=1.1
- Kept all enhancements: Conformal Prediction, Meta-Models, Recursive Prediction
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Dict, List
import gc
import warnings
import pickle
import os
warnings.filterwarnings('ignore')


class ConformalPredictor:
    """Conformal Prediction for uncertainty quantification"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantiles = {}
        
    def calibrate(self, residuals: np.ndarray, horizon: int = 1):
        """Calibrate on residuals"""
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        quantile = np.quantile(np.abs(residuals), q_level)
        self.quantiles[horizon] = quantile
        
    def predict_interval(self, point_predictions: np.ndarray, horizon: int = 1):
        """Construct prediction intervals"""
        if horizon not in self.quantiles:
            q = self.quantiles.get(1, 0.0)
        else:
            q = self.quantiles[horizon]
        
        lower = np.maximum(0, point_predictions - q)
        upper = point_predictions + q
        return lower, upper
    
    def get_uncertainty(self, point_predictions: np.ndarray, horizon: int = 1):
        """Get uncertainty measure (interval width)"""
        lower, upper = self.predict_interval(point_predictions, horizon)
        return upper - lower


class M5ForecasterEnhanced:
    """
    M5 Forecaster with enhanced preprocessing matching preprocessing.py
    + Conformal Prediction, Meta-Models, and Recursive Prediction
    """
    
    def __init__(self, data_path='./data/', alpha=0.1):
        self.data_path = data_path
        self.alpha = alpha
        self.pred_horizon = 28
        self.max_lags = 57
        
        # Model storage directory
        self.model_dir = f'{data_path}models_full/'
        self.results_dir = f'{data_path}results_full/'
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.processed_dir = f'{data_path}processed_full/'
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Data containers
        self.calendar = None
        self.prices = None
        self.sales = None
        self.submission = None
        
        # Model storage - 113 models total
        self.models = {
            'state': {},           # 3 models (CA, TX, Others)
            'store': {},           # 10 models
            'store_cat': {},       # 30 models  
            'store_dept': {}       # 70 models
        }
        
        # Meta-models for aggregate predictions
        self.meta_models = {
            'max': None,    # Predict max sale of goods on next day
            'mean': None,   # Predict mean sale of all goods on next day
            'std': None     # Predict std of sales on next day
        }
        
        # Conformal predictors
        self.conformal_predictors = {
            'state': {},
            'store': {},
            'store_cat': {},
            'store_dept': {}
        }
        
        # RMSSE scaling factors per series
        self.scaling_factors = {}
        
        # Preprocessing parameters from preprocessing.py
        self.SHIFT_DAY = 28
        self.N_LAGS = 15
        self.LAGS_SPLIT = [col for col in range(self.SHIFT_DAY, self.SHIFT_DAY + self.N_LAGS)]
        self.ROLS_SPLIT = []
        for i in [1, 7, 14]:
            for j in [7, 14, 30, 60]:
                self.ROLS_SPLIT.append([i, j])
    
    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")
        
        self.calendar = pd.read_csv(f'{self.data_path}calendar.csv')
        self.calendar['date'] = pd.to_datetime(self.calendar['date'])
        
        self.prices = pd.read_csv(f'{self.data_path}sell_prices.csv')
        self.sales = pd.read_csv(f'{self.data_path}sales_train_evaluation.csv')
        self.submission = pd.read_csv(f'{self.data_path}sample_submission.csv')
        
        print(f"Sales shape: {self.sales.shape}")
        print(f"Calendar shape: {self.calendar.shape}")
        print(f"Prices shape: {self.prices.shape}")
    
    def save_models(self):
        """Save all trained models and artifacts to disk"""
        print("\n" + "="*70)
        print("SAVING MODELS AND ARTIFACTS")
        print("="*70)
        
        # Save meta-models
        print("Saving meta-models...")
        with open(f'{self.model_dir}meta_models.pkl', 'wb') as f:
            pickle.dump(self.meta_models, f)
        
        # Save hierarchical models
        print("Saving hierarchical models...")
        for level in ['state', 'store', 'store_cat', 'store_dept']:
            with open(f'{self.model_dir}models_{level}.pkl', 'wb') as f:
                pickle.dump(self.models[level], f)
        
        # Save conformal predictors
        print("Saving conformal predictors...")
        with open(f'{self.model_dir}conformal_predictors.pkl', 'wb') as f:
            pickle.dump(self.conformal_predictors, f)
        
        # Save scaling factors
        print("Saving scaling factors...")
        with open(f'{self.model_dir}scaling_factors.pkl', 'wb') as f:
            pickle.dump(self.scaling_factors, f)
        
        print(f"✓ All models saved to: {self.model_dir}")
        print("="*70)
    
    def load_models(self):
        """Load all trained models and artifacts from disk"""
        print("\n" + "="*70)
        print("LOADING MODELS AND ARTIFACTS")
        print("="*70)
        
        try:
            # Load meta-models
            print("Loading meta-models...")
            with open(f'{self.model_dir}meta_models.pkl', 'rb') as f:
                self.meta_models = pickle.load(f)
            
            # Load hierarchical models
            print("Loading hierarchical models...")
            for level in ['state', 'store', 'store_cat', 'store_dept']:
                with open(f'{self.model_dir}models_{level}.pkl', 'rb') as f:
                    self.models[level] = pickle.load(f)
            
            # Load conformal predictors
            print("Loading conformal predictors...")
            with open(f'{self.model_dir}conformal_predictors.pkl', 'rb') as f:
                self.conformal_predictors = pickle.load(f)
            
            # Load scaling factors
            print("Loading scaling factors...")
            with open(f'{self.model_dir}scaling_factors.pkl', 'rb') as f:
                self.scaling_factors = pickle.load(f)
            
            print(f"✓ All models loaded from: {self.model_dir}")
            print("="*70)
            return True
        except FileNotFoundError as e:
            print(f"✗ Model files not found: {e}")
            print("="*70)
            return False
    
    def predict_from_saved_models(self):
        """Generate predictions using pre-trained models"""
        print("\n" + "="*70)
        print("GENERATING PREDICTIONS FROM SAVED MODELS")
        print("="*70)
        
        # Load models if not already loaded
        if not self.models['state']:
            if not self.load_models():
                print("Error: Cannot load models. Please train first.")
                return None
        
        # Load and prepare data
        self.load_data()
        df = self.prepare_training_data()
        
        # Add meta-model predictions as features
        df = self.add_meta_predictions_as_features(df)
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df)
        
        # Generate predictions
        predictions = self.predict_hierarchical(df, feature_cols)
        
        # Create submission
        submission = self.create_submission(predictions)
        
        # Save
        submission.to_csv(f'{self.data_path}submission_from_saved_models.csv', index=False)
        print(f"\n✓ Submission saved: {self.data_path}submission_from_saved_models.csv")
        
        return submission
    
    def compute_scaling_factors(self, df, train_end_day):
        """
        Compute RMSSE scaling factors for each time series
        Formula: scaling_factor = (1/(n-1)) * sum_{t=2}^{n} (y_t - y_{t-1})^2
        
        This is the denominator in RMSSE formula - the MSE of naive forecast
        """
        print("\nComputing RMSSE scaling factors...")
        
        scaling_factors = {}
        
        # Use only training data for computing scaling factors
        train_df = df[df['d_numeric'] <= train_end_day].copy()
        
        # Group by series ID
        for series_id in train_df['id'].unique():
            series_data = train_df[train_df['id'] == series_id].sort_values('d_numeric')
            sales = series_data['sales'].values
            
            # Calculate naive forecast errors: (y_t - y_{t-1})^2
            if len(sales) > 1:
                # Compute differences: y_t - y_{t-1} for t=2 to n
                naive_errors_squared = np.diff(sales) ** 2
                
                # Mean of squared naive errors: (1/(n-1)) * sum(...)
                # np.diff already gives us n-1 values
                scaling_factor = np.mean(naive_errors_squared)
                
                # Avoid division by zero - use small epsilon
                scaling_factors[series_id] = max(scaling_factor, 1e-10)
            else:
                # For series with only 1 point, use 1.0 as default
                scaling_factors[series_id] = 1.0
        
        print(f"Computed scaling factors for {len(scaling_factors)} series")
        print(f"  Min scaling: {min(scaling_factors.values()):.6f}")
        print(f"  Max scaling: {max(scaling_factors.values()):.6f}")
        print(f"  Mean scaling: {np.mean(list(scaling_factors.values())):.6f}")
        print(f"  Median scaling: {np.median(list(scaling_factors.values())):.6f}")
        
        return scaling_factors
    
    def create_rmsse_metric(self, scaling_factors):
        """Create custom RMSSE evaluation metric for LightGBM"""
        
        def rmsse_eval(y_pred, lgb_train):
            y_true = lgb_train.get_label()
            
            # Get series IDs from the dataset (need to be passed during training)
            # For now, compute simple RMSSE across all samples
            mse = np.mean((y_true - y_pred) ** 2)
            
            # Use average scaling factor as approximation
            avg_scaling = np.mean(list(scaling_factors.values()))
            rmsse = np.sqrt(mse / avg_scaling)
            
            return 'rmsse', rmsse, False  # False means lower is better
        
        return rmsse_eval
    
    def prepare_sales_data(self):
        """Transform sales data from wide to long format"""
        print("\nPreparing sales data...")
        
        id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        sales_long = pd.melt(
            self.sales,
            id_vars=id_columns,
            var_name='d',
            value_name='sales'
        )
        
        print(f"Sales long format shape: {sales_long.shape}")
        return sales_long
    
    def create_cyclical_encoding(self, df):
        """Apply sine/cosine transformation to periodic features"""
        cyclical_features = {
            'dayofweek': 7,
            'month': 12,
            'day': 31,
            'quarter': 4,
            'week': 52
        }
        
        for feature, period in cyclical_features.items():
            if feature in df.columns:
                df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / period).astype(np.float32)
                df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / period).astype(np.float32)
        
        return df

    def create_calendar_features(self, df):
        """Create time-based features with cyclical encoding"""
        print("\nCreating calendar features...")
        
        df = df.merge(self.calendar, on='d', how='left')
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)
        
        # Cyclical encoding
        df = self.create_cyclical_encoding(df)
        
        # SNAP benefits
        df['snap_CA'] = df['snap_CA'].fillna(0).astype(np.int8)
        df['snap_TX'] = df['snap_TX'].fillna(0).astype(np.int8)
        df['snap_WI'] = df['snap_WI'].fillna(0).astype(np.int8)
        
        # Events
        df['event_name_1'] = df['event_name_1'].fillna('none')
        df['event_type_1'] = df['event_type_1'].fillna('none')
        df['event_name_2'] = df['event_name_2'].fillna('none')
        df['event_type_2'] = df['event_type_2'].fillna('none')
        
        df['event_count'] = (
            (df['event_name_1'] != 'none').astype(int) +
            (df['event_name_2'] != 'none').astype(int)
        )
        
        return df
    
    def create_price_features(self, df):
        """Create price-related features - enhanced for first place"""
        print("\nCreating price features...")
        
        df['wm_yr_wk'] = df['wm_yr_wk'].astype(np.int16)
        df = df.merge(self.prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        
        # Forward/backward fill missing prices
        df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        
        # Price statistics across items on same day
        price_stats = df.groupby(['item_id', 'd'])['sell_price'].agg(['mean', 'std', 'max', 'min'])
        price_stats.columns = ['price_mean', 'price_std', 'price_max', 'price_min']
        df = df.merge(price_stats, left_on=['item_id', 'd'], right_index=True, how='left')
        
        df['price_norm'] = df['sell_price'] / (df['price_mean'] + 1e-5)
        df['price_nunique'] = df.groupby(['item_id', 'd'])['sell_price'].transform('nunique')
        
        # Price momentum
        df = df.sort_values(['store_id', 'item_id', 'd'])
        df['price_change'] = df.groupby(['store_id', 'item_id'])['sell_price'].pct_change()
        df['price_change'] = df['price_change'].fillna(0).clip(-1, 1)
        
        # Price moving average
        df['price_rolling_mean'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform(
            lambda x: x.rolling(window=28, min_periods=1).mean()
        )
        
        # *** FIRST PLACE FEATURES: Price momentum indicators ***
        df['price_momentum'] = (
            df['sell_price'] - df['price_rolling_mean']
        ).astype(np.float32)
        
        return df
    
    def create_lag_features(self, df):
        """Create lag features matching preprocessing.py structure"""
        print(f"\nCreating lag features: {self.LAGS_SPLIT}...")
        
        df = df.sort_values(['id', 'd'])
        
        # Create lags from 28 to 42 (as per preprocessing.py)
        for lag in self.LAGS_SPLIT:
            col_name = f'sales_lag_{lag}'
            df[col_name] = df.groupby('id')['sales'].shift(lag).astype(np.float32)
        
        # Also keep some additional useful lags from original code
        additional_lags = [7, 14, 21, 49, 56]
        for lag in additional_lags:
            if lag not in self.LAGS_SPLIT:
                col_name = f'sales_lag_{lag}'
                df[col_name] = df.groupby('id')['sales'].shift(lag).astype(np.float32)
        
        return df
    
    def create_rolling_features(self, df):
        """Create rolling window features matching preprocessing.py structure"""
        print("\nCreating rolling window features...")
        
        df = df.sort_values(['id', 'd'])
        
        # Create rolling features as per preprocessing.py ROLS_SPLIT
        # Format: [shift_day, roll_window]
        for shift_day, roll_wind in self.ROLS_SPLIT:
            col_name = f'rolling_mean_tmp_{shift_day}_{roll_wind}'
            df[col_name] = df.groupby(['id'])['sales'].transform(
                lambda x: x.shift(shift_day).rolling(roll_wind, min_periods=1).mean()
            ).astype(np.float32)
        
        # Also keep the original rolling features for backwards compatibility
        windows = [7, 14, 28, 56]
        for window in windows:
            sales_shifted = df.groupby('id')['sales'].shift(1)
            
            # Mean
            col_name = f'rolling_mean_{window}'
            if col_name not in df.columns:
                df[col_name] = sales_shifted.groupby(df['id']).transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                ).astype(np.float32)
            
            # Std
            df[f'rolling_std_{window}'] = sales_shifted.groupby(df['id']).transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            ).astype(np.float32)
            
            # Max
            df[f'rolling_max_{window}'] = sales_shifted.groupby(df['id']).transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            ).astype(np.float32)
            
            # Min
            df[f'rolling_min_{window}'] = sales_shifted.groupby(df['id']).transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            ).astype(np.float32)
        
        return df
    
    def create_encoding_features(self, df):
        """Create encoded categorical features with mean and std (matching preprocessing.py)"""
        print("\nCreating encoding features...")
        
        # Mean encodings (as per preprocessing.py)
        for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
            # Mean encoding
            df[f'enc_{col}_mean'] = df.groupby(col)['sales'].transform('mean').astype(np.float32)
            # Std encoding (from preprocessing.py)
            df[f'enc_{col}_std'] = df.groupby(col)['sales'].transform('std').fillna(0).astype(np.float32)
        
        # Interaction encodings
        df['item_store_sales_mean'] = df.groupby(['item_id', 'store_id'])['sales'].transform('mean').astype(np.float32)
        df['cat_store_sales_mean'] = df.groupby(['cat_id', 'store_id'])['sales'].transform('mean').astype(np.float32)
        df['dept_store_sales_mean'] = df.groupby(['dept_id', 'store_id'])['sales'].transform('mean').astype(np.float32)
        
        return df
    
    def create_aggregate_features(self, df):
        """
        *** NEW: Create aggregate features per day ***
        These will be targets for meta-models
        """
        print("\nCreating aggregate daily features...")
        
        df = df.sort_values(['d_numeric'])
        
        # Compute daily aggregates: max, mean, std of all sales on that day
        daily_agg = df.groupby('d_numeric')['sales'].agg(['max', 'mean', 'std']).reset_index()
        daily_agg.columns = ['d_numeric', 'daily_max_sales', 'daily_mean_sales', 'daily_std_sales']
        daily_agg['daily_std_sales'] = daily_agg['daily_std_sales'].fillna(0)
        
        # Merge back to main dataframe
        df = df.merge(daily_agg, on='d_numeric', how='left')
        
        # Create lagged versions (what were the aggregates yesterday, 7 days ago, etc.)
        for lag in [1, 7, 14, 28]:
            df[f'daily_max_sales_lag_{lag}'] = df.groupby('id')['daily_max_sales'].shift(lag).astype(np.float32)
            df[f'daily_mean_sales_lag_{lag}'] = df.groupby('id')['daily_mean_sales'].shift(lag).astype(np.float32)
            df[f'daily_std_sales_lag_{lag}'] = df.groupby('id')['daily_std_sales'].shift(lag).astype(np.float32)
        
        print(f"Added daily aggregate features: max, mean, std (current + lags)")
        return df
    
    def reduce_mem_usage(self, df):
        """Reduce memory usage"""
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Memory: {start_mem:.2f}MB → {end_mem:.2f}MB ({100*(start_mem-end_mem)/start_mem:.1f}% ↓)')
        
        return df
    
    def prepare_training_data(self):
        """Main feature engineering pipeline with enhanced preprocessing"""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING - ENHANCED PREPROCESSING")
        print("="*70)
        
        df = self.prepare_sales_data()
        df = self.create_calendar_features(df)
        df = self.create_price_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_encoding_features(df)
        
        # Additional features
        df['sales_velocity'] = (
            df.get('rolling_mean_7', 0) / (df.get('rolling_mean_28', 1) + 1e-5)
        ).astype(np.float32)
        
        df['demand_change'] = (
            df.get('rolling_mean_7', 0) - df.get('rolling_mean_28', 0)
        ).astype(np.float32)
        
        df = self.reduce_mem_usage(df)
        df['d_numeric'] = df['d'].str.replace('d_', '').astype(np.int16)
        
        # *** NEW: Add aggregate features before meta-model training ***
        df = self.create_aggregate_features(df)
        
        # Coerce dtypes
        cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'id', 'd']
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype('category')
        
        print(f"\nFinal dataset shape: {df.shape}")
        print("\nEnhanced features from preprocessing.py:")
        print("  • Mean and Std encodings for cat_id, dept_id, item_id")
        print(f"  • Lag features from {self.SHIFT_DAY} to {self.SHIFT_DAY + self.N_LAGS - 1}")
        print(f"  • {len(self.ROLS_SPLIT)} rolling window combinations")

        if hasattr(self, 'save_by_store') and self.save_by_store:
            for store_id in df['store_id'].cat.categories:
                store_df = df[df['store_id'] == store_id]
                store_df.to_pickle(f'{self.data_path}processed/test_{store_id}.pkl')
                print(f"  Saved: test_{store_id}.pkl")
        
        return df
    
    def save_preprocessed_data(self, df, filename='preprocessed_data.pkl'):
        """Save preprocessed data to pickle file"""
        filepath = f'{self.data_path}processed/{filename}'
        os.makedirs(f'{self.data_path}processed/', exist_ok=True)
        df.to_pickle(filepath)
        print(f"✓ Preprocessed data saved to: {filepath}")
        return filepath

    def load_preprocessed_data(self, filename='preprocessed_data.pkl'):
        """Load preprocessed data from pickle file"""
        filepath = f'{self.data_path}processed/{filename}'
        if os.path.exists(filepath):
            print(f"Loading preprocessed data from: {filepath}")
            df = pd.read_pickle(filepath)
            print(f"✓ Loaded preprocessed data: shape {df.shape}")
            return df
        else:
            print(f"File not found: {filepath}")
            return None
    
    def get_feature_columns(self, df):
        """Define feature columns for modeling"""
        exclude_cols = [
            'id', 'date', 'd', 'sales', 'wm_yr_wk',
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
            'weekday', 'd_numeric',
            # Exclude current day aggregates (only use lagged versions)
            'daily_max_sales', 'daily_mean_sales', 'daily_std_sales'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def get_meta_feature_columns(self, df):
        """
        *** NEW: Define feature columns for meta-models ***
        Meta-models predict daily aggregates based on temporal and aggregate lag features
        """
        # Use temporal features and lagged aggregate features
        meta_features = [
            'year', 'month', 'week', 'day', 'dayofweek', 'quarter', 'is_weekend',
            'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
            'day_sin', 'day_cos', 'quarter_sin', 'quarter_cos', 'week_sin', 'week_cos',
            'event_count'
        ]
        
        # Add lagged aggregate features
        for lag in [1, 7, 14, 28]:
            meta_features.extend([
                f'daily_max_sales_lag_{lag}',
                f'daily_mean_sales_lag_{lag}',
                f'daily_std_sales_lag_{lag}'
            ])
        
        # Filter to only existing columns
        meta_features = [col for col in meta_features if col in df.columns]
        return meta_features
    
    def train_meta_models(self, df, train_days=730):
        """
        *** NEW: Train meta-models to predict daily aggregates ***
        Three regression models: max, mean, std of next day sales
        """
        print("\n" + "="*70)
        print("TRAINING META-MODELS FOR AGGREGATE PREDICTIONS")
        print("="*70)
        print("\nMeta-Models:")
        print("  • Model 1: Predict max sale of goods on next day")
        print("  • Model 2: Predict mean sale of all goods on next day")
        print("  • Model 3: Predict std of sales on next day")
        print("="*70)
        
        max_day = df['d_numeric'].max()
        train_end = max_day - self.pred_horizon
        train_start = max(1, train_end - train_days)
        val_start = train_end + 1
        val_end = max_day
        
        print(f"\nMeta-model Train period: d_{train_start} to d_{train_end}")
        print(f"Meta-model Validation period: d_{val_start} to d_{val_end}")
        
        meta_feature_cols = self.get_meta_feature_columns(df)
        print(f"\nMeta-model features: {len(meta_feature_cols)}")
        
        # Prepare daily aggregated data (one row per day)
        daily_data = df.groupby('d_numeric').first().reset_index()
        
        # LightGBM parameters for meta-models (using Tweedie)
        meta_params = {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'rmse',
            'num_leaves': 31,
            'min_child_samples': 20,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbosity': -1,
            'num_threads': -1
        }
        
        # Train three meta-models
        for target_name in ['max', 'mean', 'std']:
            target_col = f'daily_{target_name}_sales'
            print(f"\n[Meta-Model] Training for: {target_name.upper()}")
            
            # Prepare train/val splits
            train_mask = (daily_data['d_numeric'] >= train_start) & (daily_data['d_numeric'] <= train_end)
            val_mask = (daily_data['d_numeric'] >= val_start) & (daily_data['d_numeric'] <= val_end)
            
            X_train = daily_data.loc[train_mask, meta_feature_cols]
            y_train = daily_data.loc[train_mask, target_col]
            X_val = daily_data.loc[val_mask, meta_feature_cols]
            y_val = daily_data.loc[val_mask, target_col]
            
            print(f"  Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")
            
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            model = lgb.train(
                meta_params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_val],
                valid_names=['valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            self.meta_models[target_name] = model
            
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            val_rmse = np.sqrt(np.mean((y_val.values - val_preds) ** 2))
            print(f"  ✓ Trained (best_iter={model.best_iteration}, val_rmse={val_rmse:.4f})")
            
            del X_train, y_train, X_val, y_val
            gc.collect()
        
        print("\n" + "="*70)
        print("META-MODELS TRAINING COMPLETE")
        print("="*70)
    
    def add_meta_predictions_as_features(self, df):
        """
        *** NEW: Add meta-model predictions as features to main dataframe ***
        For each day, predict the next day's max, mean, std using meta-models
        """
        print("\nAdding meta-model predictions as features...")
        
        meta_feature_cols = self.get_meta_feature_columns(df)
        
        # Get unique days and prepare features
        daily_data = df.groupby('d_numeric').first().reset_index()
        
        # Predict for each day
        for target_name in ['max', 'mean', 'std']:
            model = self.meta_models[target_name]
            X_pred = daily_data[meta_feature_cols]
            predictions = model.predict(X_pred, num_iteration=model.best_iteration)
            daily_data[f'meta_pred_{target_name}'] = predictions
        
        # Shift predictions: prediction made on day t is for day t+1
        # So we shift back to align with the day it's predicting
        for target_name in ['max', 'mean', 'std']:
            daily_data[f'meta_pred_{target_name}_next'] = daily_data[f'meta_pred_{target_name}'].shift(1)
        
        # Merge back to main dataframe
        merge_cols = ['d_numeric'] + [f'meta_pred_{t}_next' for t in ['max', 'mean', 'std']]
        df = df.merge(daily_data[merge_cols], on='d_numeric', how='left')
        
        # Fill any NaN values (first day won't have predictions)
        for target_name in ['max', 'mean', 'std']:
            col_name = f'meta_pred_{target_name}_next'
            df[col_name] = df[col_name].fillna(df[col_name].mean()).astype(np.float32)
        
        print(f"Added 3 meta-prediction features: meta_pred_max_next, meta_pred_mean_next, meta_pred_std_next")
        return df
    
    def train_hierarchical_models(self, df, train_days=730):
        """
        *** HIERARCHICAL MODELING WITH TWEEDIE OBJECTIVE ***
        
        Train 113 models with Tweedie loss (matching preprocessing.py)
        """
        print("\n" + "="*70)
        print("HIERARCHICAL MODEL TRAINING - ENHANCED PREPROCESSING")
        print("="*70)
        print("\nModel Structure:")
        print("  • State-level: 3 models (CA, TX, Others)")
        print("  • Store-level: 10 models")
        print("  • Store-Category: 30 models")
        print("  • Store-Department: 70 models")
        print("  • Total: 113 models")
        print("  • Loss Function: Tweedie (variance_power=1.1)")
        print("  • Calibration: Conformal Prediction on validation set")
        print("  • Enhanced with: Meta-model predictions as features")
        print("="*70)
        
        # Compute RMSSE scaling factors
        max_day = df['d_numeric'].max()
        train_end = max_day - self.pred_horizon
        self.scaling_factors = self.compute_scaling_factors(df, train_end)
        rmsse_metric = self.create_rmsse_metric(self.scaling_factors)
        
        train_start = max(1, train_end - train_days)
        val_start = train_end + 1
        val_end = max_day
        
        print(f"\nTrain period: d_{train_start} to d_{train_end}")
        print(f"Validation period: d_{val_start} to d_{val_end}")
        
        feature_cols = self.get_feature_columns(df)
        cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year']
        
        # LightGBM parameters (matching preprocessing.py with Tweedie)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'rmse',
            'subsample': 0.5,
            'subsample_freq': 1,
            'learning_rate': 0.03,  # Slightly higher than preprocessing.py for faster convergence
            'num_leaves': 2**8-1,    # Compromise between original codes
            'min_data_in_leaf': 2**8-1,
            'feature_fraction': 0.6,
            'max_bin': 100,
            'max_depth': 8,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'boost_from_average': False,
            'verbose': -1,
            'num_threads': -1
        }
        
        # 0. STATE-LEVEL MODELS (3 models: CA, TX, Others)
        print("\n" + "="*70)
        print("Training State-Level Models (3 models)")
        print("="*70)
        
        # Define state groups: CA, TX, Others (which includes WI)
        state_groups = {
            'CA': ['CA'],
            'TX': ['TX'],
            'Others': ['WI']
        }
        
        for i, (state_name, states) in enumerate(state_groups.items(), 1):
            print(f"\n[{i}/3] Training model for state group: {state_name}")
            
            # Filter data for this state group
            state_data = df[df['state_id'].isin(states)].copy()
            
            train_mask = (state_data['d_numeric'] >= train_start) & (state_data['d_numeric'] <= train_end)
            val_mask = (state_data['d_numeric'] >= val_start) & (state_data['d_numeric'] <= val_end)
            
            X_train = state_data.loc[train_mask, feature_cols]
            y_train = state_data.loc[train_mask, 'sales']
            X_val = state_data.loc[val_mask, feature_cols]
            y_val = state_data.loc[val_mask, 'sales']
            
            if len(X_train) == 0 or len(X_val) == 0:
                print(f"  Skipping {state_name} - insufficient data")
                continue
            
            print(f"  Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")
            
            lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=cat_features)
            
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_val],
                valid_names=['valid'],
                feval=rmsse_metric,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            self.models['state'][state_name] = model
            
            # Calibrate conformal predictor
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            residuals = y_val.values - val_preds
            cp = ConformalPredictor(alpha=self.alpha)
            cp.calibrate(residuals)
            self.conformal_predictors['state'][state_name] = cp
            
            print(f"  ✓ Trained (best_iter={model.best_iteration}), Conformal calibrated")
            
            del X_train, y_train, X_val, y_val, state_data
            gc.collect()
        
        # 1. STORE-LEVEL MODELS (10 models)
        print("\n" + "="*70)
        print("Training Store-Level Models (10 models)")
        print("="*70)
        
        stores = df['store_id'].cat.categories
        for i, store in enumerate(stores, 1):
            print(f"\n[{i}/10] Training model for store: {store}")
            
            # Filter data for this store
            store_data = df[df['store_id'] == store].copy()
            
            train_mask = (store_data['d_numeric'] >= train_start) & (store_data['d_numeric'] <= train_end)
            val_mask = (store_data['d_numeric'] >= val_start) & (store_data['d_numeric'] <= val_end)
            
            X_train = store_data.loc[train_mask, feature_cols]
            y_train = store_data.loc[train_mask, 'sales']
            X_val = store_data.loc[val_mask, feature_cols]
            y_val = store_data.loc[val_mask, 'sales']
            
            if len(X_train) == 0 or len(X_val) == 0:
                print(f"  Skipping {store} - insufficient data")
                continue
            
            lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=cat_features)
            
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_val],
                valid_names=['valid'],
                feval=rmsse_metric,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            self.models['store'][store] = model
            
            # Calibrate conformal predictor
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            residuals = y_val.values - val_preds
            cp = ConformalPredictor(alpha=self.alpha)
            cp.calibrate(residuals)
            self.conformal_predictors['store'][store] = cp
            
            print(f"  ✓ Trained (best_iter={model.best_iteration}), Conformal calibrated")
            
            del X_train, y_train, X_val, y_val, store_data
            gc.collect()
        
        # 2. STORE-CATEGORY MODELS (30 models)
        print("\n" + "="*70)
        print("Training Store-Category Models (30 models)")
        print("="*70)
        
        categories = df['cat_id'].cat.categories
        model_count = 0
        for store in stores:
            for cat in categories:
                model_count += 1
                print(f"\n[{model_count}/30] Training: {store} × {cat}")
                
                # Filter data
                mask = (df['store_id'] == store) & (df['cat_id'] == cat)
                subset = df[mask].copy()
                
                train_mask = (subset['d_numeric'] >= train_start) & (subset['d_numeric'] <= train_end)
                val_mask = (subset['d_numeric'] >= val_start) & (subset['d_numeric'] <= val_end)
                
                X_train = subset.loc[train_mask, feature_cols]
                y_train = subset.loc[train_mask, 'sales']
                X_val = subset.loc[val_mask, feature_cols]
                y_val = subset.loc[val_mask, 'sales']
                
                if len(X_train) == 0 or len(X_val) == 0:
                    print(f"  Skipping - insufficient data")
                    continue
                
                lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
                lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=cat_features)
                
                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=[lgb_val],
                    valid_names=['valid'],
                    feval=rmsse_metric,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                key = f"{store}_{cat}"
                self.models['store_cat'][key] = model
                
                # Calibrate conformal predictor
                val_preds = model.predict(X_val, num_iteration=model.best_iteration)
                residuals = y_val.values - val_preds
                cp = ConformalPredictor(alpha=self.alpha)
                cp.calibrate(residuals)
                self.conformal_predictors['store_cat'][key] = cp
                
                print(f"  ✓ Trained (best_iter={model.best_iteration}), Conformal calibrated")
                
                del X_train, y_train, X_val, y_val, subset
                gc.collect()
        
        # 3. STORE-DEPARTMENT MODELS (70 models)
        print("\n" + "="*70)
        print("Training Store-Department Models (70 models)")
        print("="*70)
        
        departments = df['dept_id'].cat.categories
        model_count = 0
        for store in stores:
            for dept in departments:
                model_count += 1
                print(f"\n[{model_count}/70] Training: {store} × {dept}")
                
                # Filter data
                mask = (df['store_id'] == store) & (df['dept_id'] == dept)
                subset = df[mask].copy()
                
                train_mask = (subset['d_numeric'] >= train_start) & (subset['d_numeric'] <= train_end)
                val_mask = (subset['d_numeric'] >= val_start) & (subset['d_numeric'] <= val_end)
                
                X_train = subset.loc[train_mask, feature_cols]
                y_train = subset.loc[train_mask, 'sales']
                X_val = subset.loc[val_mask, feature_cols]
                y_val = subset.loc[val_mask, 'sales']
                
                if len(X_train) == 0 or len(X_val) == 0:
                    print(f"  Skipping - insufficient data")
                    continue
                
                lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
                lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=cat_features)
                
                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=[lgb_val],
                    valid_names=['valid'],
                    feval=rmsse_metric,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                key = f"{store}_{dept}"
                self.models['store_dept'][key] = model
                
                # Calibrate conformal predictor
                val_preds = model.predict(X_val, num_iteration=model.best_iteration)
                residuals = y_val.values - val_preds
                cp = ConformalPredictor(alpha=self.alpha)
                cp.calibrate(residuals)
                self.conformal_predictors['store_dept'][key] = cp
                
                print(f"  ✓ Trained (best_iter={model.best_iteration}), Conformal calibrated")
                
                del X_train, y_train, X_val, y_val, subset
                gc.collect()
        
        print("\n" + "="*70)
        print("HIERARCHICAL TRAINING COMPLETE")
        print("="*70)
        print(f"State models: {len(self.models['state'])}")
        print(f"Store models: {len(self.models['store'])}")
        print(f"Store-Category models: {len(self.models['store_cat'])}")
        print(f"Store-Department models: {len(self.models['store_dept'])}")
        total_models = (len(self.models['state']) + len(self.models['store']) + 
                       len(self.models['store_cat']) + len(self.models['store_dept']))
        print(f"Total models: {total_models}")
        
        return feature_cols
    
    def update_recursive_features(self, current_data, new_sales):
        """
        *** NEW: Update lag and rolling features with new predictions ***
        This is the key to recursive/auto-regressive forecasting
        """
        # Store new sales in a temporary column
        current_data['new_sales'] = new_sales
        
        # Update lag features (including preprocessing.py lags)
        all_lags = list(self.LAGS_SPLIT) + [7, 14, 21, 49, 56]
        
        for lag in all_lags:
            col_name = f'sales_lag_{lag}'
            if col_name in current_data.columns:
                # Shift the existing lags
                if lag == 7:
                    # Use the new prediction for the smallest lag
                    if 'sales_lag_6' in current_data.columns:
                        current_data[col_name] = current_data['sales_lag_6']
                else:
                    # Shift from the next smaller lag
                    prev_lag = lag - 1
                    prev_col = f'sales_lag_{prev_lag}'
                    if prev_col in current_data.columns:
                        current_data[col_name] = current_data[prev_col]
        
        # Update rolling features from preprocessing.py
        for shift_day, roll_wind in self.ROLS_SPLIT:
            col_name = f'rolling_mean_tmp_{shift_day}_{roll_wind}'
            if col_name in current_data.columns:
                # Approximate update for rolling features
                old_value = current_data[col_name].values
                # Incorporate new sales (approximate moving average update)
                current_data[col_name] = (old_value * (roll_wind - 1) + new_sales) / roll_wind
                current_data[col_name] = current_data[col_name].astype(np.float32)
        
        # Update standard rolling window features
        windows = [7, 14, 28, 56]
        for window in windows:
            # Update rolling mean
            mean_col = f'rolling_mean_{window}'
            if mean_col in current_data.columns:
                old_mean = current_data[mean_col].values
                current_data[mean_col] = (old_mean * (window - 1) + new_sales) / window
                current_data[mean_col] = current_data[mean_col].astype(np.float32)
            
            # Update rolling std (approximate)
            std_col = f'rolling_std_{window}'
            if std_col in current_data.columns:
                current_data[std_col] = current_data[std_col].astype(np.float32)
            
            # Update rolling max
            max_col = f'rolling_max_{window}'
            if max_col in current_data.columns:
                current_data[max_col] = np.maximum(current_data[max_col].values, new_sales).astype(np.float32)
            
            # Update rolling min
            min_col = f'rolling_min_{window}'
            if min_col in current_data.columns:
                current_data[min_col] = np.minimum(current_data[min_col].values, new_sales).astype(np.float32)
        
        # Update velocity and demand change
        if 'sales_velocity' in current_data.columns:
            current_data['sales_velocity'] = (
                current_data.get('rolling_mean_7', 0) / (current_data.get('rolling_mean_28', 1) + 1e-5)
            ).astype(np.float32)
        
        if 'demand_change' in current_data.columns:
            current_data['demand_change'] = (
                current_data.get('rolling_mean_7', 0) - current_data.get('rolling_mean_28', 0)
            ).astype(np.float32)
        
        # Clean up temporary column
        current_data.drop('new_sales', axis=1, inplace=True)
        
        return current_data
    
    def predict_hierarchical(self, df, feature_cols):
        """
        *** RECURSIVE AUTO-REGRESSIVE PREDICTION ***
        
        Weight predictions from all 113 models based on conformal prediction uncertainty
        Each time step uses predictions from previous steps as input features
        Lower uncertainty = higher weight
        Uses meta-model predictions as features
        """
        print("\n" + "="*70)
        print("GENERATING RECURSIVE AUTO-REGRESSIVE PREDICTIONS")
        print("="*70)
        print("Each model uses ensemble predictions from previous steps")
        print("="*70)
        
        max_day = df['d_numeric'].max()
        
        # Storage for predictions from each day
        all_predictions = []
        
        # Start with the last day of historical data
        last_data = df[df['d_numeric'] == max_day].copy()
        
        # Prepare meta-model daily data for predictions
        meta_feature_cols = self.get_meta_feature_columns(df)
        
        for day in range(1, self.pred_horizon + 1):
            print(f"[RECURSIVE] Predicting day {day}/{self.pred_horizon}...", end='\r')
            
            current_data = last_data.copy()
            current_data['d_numeric'] = max_day + day
            
            # Update temporal features
            future_date = self.calendar['date'].max() + pd.Timedelta(days=day)
            current_data['date'] = future_date
            current_data['year'] = future_date.year
            current_data['month'] = future_date.month
            current_data['day'] = future_date.day
            current_data['dayofweek'] = future_date.dayofweek
            current_data['quarter'] = future_date.quarter
            current_data['week'] = future_date.isocalendar().week
            current_data['is_weekend'] = int(future_date.dayofweek >= 5)
            
            # Recalculate cyclical features
            current_data['dayofweek_sin'] = np.sin(2 * np.pi * current_data['dayofweek'] / 7).astype(np.float32)
            current_data['dayofweek_cos'] = np.cos(2 * np.pi * current_data['dayofweek'] / 7).astype(np.float32)
            current_data['month_sin'] = np.sin(2 * np.pi * current_data['month'] / 12).astype(np.float32)
            current_data['month_cos'] = np.cos(2 * np.pi * current_data['month'] / 12).astype(np.float32)
            current_data['day_sin'] = np.sin(2 * np.pi * current_data['day'] / 31).astype(np.float32)
            current_data['day_cos'] = np.cos(2 * np.pi * current_data['day'] / 31).astype(np.float32)
            current_data['quarter_sin'] = np.sin(2 * np.pi * current_data['quarter'] / 4).astype(np.float32)
            current_data['quarter_cos'] = np.cos(2 * np.pi * current_data['quarter'] / 4).astype(np.float32)
            current_data['week_sin'] = np.sin(2 * np.pi * current_data['week'] / 52).astype(np.float32)
            current_data['week_cos'] = np.cos(2 * np.pi * current_data['week'] / 52).astype(np.float32)
            
            # Update meta-model predictions for this future day
            future_meta_row = current_data[meta_feature_cols].iloc[0:1].copy()
            
            # Predict next day aggregates using meta-models
            for target_name in ['max', 'mean', 'std']:
                model = self.meta_models[target_name]
                pred = model.predict(future_meta_row, num_iteration=model.best_iteration)[0]
                current_data[f'meta_pred_{target_name}_next'] = pred
            
            # Initialize predictions and weights arrays
            weighted_predictions = np.zeros(len(current_data))
            total_weights = np.zeros(len(current_data))
            
            # 0. State-level predictions with uncertainty weighting
            for state_name, model in self.models['state'].items():
                if state_name == 'CA':
                    mask = current_data['state_id'] == 'CA'
                elif state_name == 'TX':
                    mask = current_data['state_id'] == 'TX'
                elif state_name == 'Others':
                    mask = current_data['state_id'] == 'WI'
                else:
                    continue
                    
                if mask.sum() > 0:
                    X_pred = current_data.loc[mask, feature_cols]
                    preds = np.maximum(0, model.predict(X_pred, num_iteration=model.best_iteration))
                    
                    # Get uncertainty and compute weight (inverse uncertainty)
                    cp = self.conformal_predictors['state'][state_name]
                    uncertainty = cp.get_uncertainty(preds, horizon=day)
                    weights = 1.0 / (uncertainty + 1e-6)
                    
                    weighted_predictions[mask] += preds * weights
                    total_weights[mask] += weights
            
            # 1. Store-level predictions with uncertainty weighting
            for store, model in self.models['store'].items():
                mask = current_data['store_id'] == store
                if mask.sum() > 0:
                    X_pred = current_data.loc[mask, feature_cols]
                    preds = np.maximum(0, model.predict(X_pred, num_iteration=model.best_iteration))
                    
                    # Get uncertainty and compute weight
                    cp = self.conformal_predictors['store'][store]
                    uncertainty = cp.get_uncertainty(preds, horizon=day)
                    weights = 1.0 / (uncertainty + 1e-6)
                    
                    weighted_predictions[mask] += preds * weights
                    total_weights[mask] += weights
            
            # 2. Store-Category predictions with uncertainty weighting
            for key, model in self.models['store_cat'].items():
                store, cat = key.split('_')[0], '_'.join(key.split('_')[1:])
                mask = (current_data['store_id'] == store) & (current_data['cat_id'] == cat)
                if mask.sum() > 0:
                    X_pred = current_data.loc[mask, feature_cols]
                    preds = np.maximum(0, model.predict(X_pred, num_iteration=model.best_iteration))
                    
                    # Get uncertainty and compute weight
                    cp = self.conformal_predictors['store_cat'][key]
                    uncertainty = cp.get_uncertainty(preds, horizon=day)
                    weights = 1.0 / (uncertainty + 1e-6)
                    
                    weighted_predictions[mask] += preds * weights
                    total_weights[mask] += weights
            
            # 3. Store-Department predictions with uncertainty weighting
            for key, model in self.models['store_dept'].items():
                parts = key.split('_')
                store = parts[0]
                dept = '_'.join(parts[1:])
                mask = (current_data['store_id'] == store) & (current_data['dept_id'] == dept)
                if mask.sum() > 0:
                    X_pred = current_data.loc[mask, feature_cols]
                    preds = np.maximum(0, model.predict(X_pred, num_iteration=model.best_iteration))
                    
                    # Get uncertainty and compute weight
                    cp = self.conformal_predictors['store_dept'][key]
                    uncertainty = cp.get_uncertainty(preds, horizon=day)
                    weights = 1.0 / (uncertainty + 1e-6)
                    
                    weighted_predictions[mask] += preds * weights
                    total_weights[mask] += weights
            
            # Normalize by total weights to get final weighted average
            valid_mask = total_weights > 0
            final_preds = np.zeros(len(current_data))
            final_preds[valid_mask] = weighted_predictions[valid_mask] / total_weights[valid_mask]
            
            # Store predictions
            pred_df = current_data[['id']].copy()
            pred_df[f'F{day}'] = final_preds
            all_predictions.append(pred_df)
            
            # *** KEY RECURSIVE STEP: Update features with new predictions ***
            # These predictions will be used as input for the next time step
            current_data = self.update_recursive_features(current_data, final_preds)
            
            # Update last_data for next iteration (recursive step)
            last_data = current_data.copy()
        
        print("\n[RECURSIVE] Auto-regressive predictions complete!" + " "*30)
        
        # Combine all days
        final_predictions = all_predictions[0]
        for i in range(1, len(all_predictions)):
            final_predictions = final_predictions.merge(all_predictions[i], on='id', how='left')
        
        # Save predictions
        print("\nSaving prediction results...")
        final_predictions.to_csv(f'{self.results_dir}predictions_recursive.csv', index=False)
        print(f"✓ Predictions saved to: {self.results_dir}predictions_recursive.csv")
        
        return final_predictions
    
    def create_submission(self, predictions):
        """Create submission file"""
        print("\n" + "="*70)
        print("CREATING SUBMISSION")
        print("="*70)
        
        validation = self.submission[self.submission['id'].str.contains('validation')]
        validation = validation[['id']].merge(predictions, on='id', how='left')
        
        evaluation = self.submission[self.submission['id'].str.contains('evaluation')]
        evaluation['id'] = evaluation['id'].str.replace('evaluation', 'validation')
        evaluation = evaluation[['id']].merge(predictions, on='id', how='left')
        evaluation['id'] = evaluation['id'].str.replace('validation', 'evaluation')
        
        submission = pd.concat([validation, evaluation], axis=0)
        submission = submission[self.submission.columns]
        
        print(f"Submission shape: {submission.shape}")
        return submission
    
    def run_full_pipeline(self):
        """Execute complete pipeline with enhanced preprocessing"""
        print("\n" + "="*80)
        print(" M5 FORECASTING - ENHANCED PREPROCESSING + RECURSIVE")
        print("="*80)
        print("\nMETHODOLOGY:")
        print("  • Enhanced preprocessing matching preprocessing.py features")
        print("  • Hierarchical modeling (State, Store, Store-Cat, Store-Dept)")
        print("  • 113 LightGBM models with Tweedie loss (variance_power=1.1)")
        print("  • Mean and Std encodings for categorical features")
        print("  • Specific lag structure (28-42) from preprocessing.py")
        print("  • Rolling window combinations from preprocessing.py")
        print("  • 3 Meta-models (max, mean, std predictions as features)")
        print("  • Uncertainty-weighted ensemble using Conformal Prediction")
        print("  • RECURSIVE PREDICTION: Each step uses previous predictions")
        print("  • Cyclical encoding for temporal features")
        print("="*80 + "\n")
        
        # Load and prepare data
        self.load_data()

        # Check if preprocessed data exists
        df = self.load_preprocessed_data()
        if df is None:
            df = self.prepare_training_data()
            # Save preprocessed data for future use
            self.save_preprocessed_data(df)
        # Train meta-models first
        self.train_meta_models(df)
        
        # Add meta-model predictions as features
        df = self.add_meta_predictions_as_features(df)
        
        # Train hierarchical models (now with meta-model features)
        feature_cols = self.train_hierarchical_models(df)
        
        # Save all trained models
        self.save_models()
        
        # Generate recursive ensemble predictions
        predictions = self.predict_hierarchical(df, feature_cols)
        
        # Create submission
        submission = self.create_submission(predictions)
        
        # Save
        submission.to_csv(f'{self.data_path}submission_enhanced_recursive.csv', index=False)
        print(f"\nSubmission saved: {self.data_path}submission_enhanced_recursive.csv")
        
        # Save summary statistics
        summary = {
            'total_models': (len(self.models['state']) + len(self.models['store']) + 
                           len(self.models['store_cat']) + len(self.models['store_dept'])),
            'state_models': len(self.models['state']),
            'store_models': len(self.models['store']),
            'store_cat_models': len(self.models['store_cat']),
            'store_dept_models': len(self.models['store_dept']),
            'meta_models': len(self.meta_models),
            'prediction_method': 'enhanced_recursive_auto_regressive',
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'lag_structure': f'{self.SHIFT_DAY} to {self.SHIFT_DAY + self.N_LAGS - 1}',
            'rolling_windows': len(self.ROLS_SPLIT),
            'prediction_stats': {
                'mean': float(submission.iloc[:, 1:].mean().mean()),
                'std': float(submission.iloc[:, 1:].std().mean()),
                'min': float(submission.iloc[:, 1:].min().min()),
                'max': float(submission.iloc[:, 1:].max().max())
            }
        }
        
        with open(f'{self.results_dir}training_summary_enhanced.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        # Save as readable text file
        with open(f'{self.results_dir}training_summary_enhanced.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("M5 FORECASTING - TRAINING SUMMARY (ENHANCED)\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Models Trained: {summary['total_models']}\n")
            f.write(f"  - State-level: {summary['state_models']}\n")
            f.write(f"  - Store-level: {summary['store_models']}\n")
            f.write(f"  - Store-Category: {summary['store_cat_models']}\n")
            f.write(f"  - Store-Department: {summary['store_dept_models']}\n")
            f.write(f"  - Meta-models: {summary['meta_models']}\n\n")
            f.write(f"Prediction Method: {summary['prediction_method']}\n")
            f.write(f"Objective: {summary['objective']} (variance_power={summary['tweedie_variance_power']})\n")
            f.write(f"Lag Structure: {summary['lag_structure']}\n")
            f.write(f"Rolling Windows: {summary['rolling_windows']} combinations\n\n")
            f.write("Prediction Statistics:\n")
            f.write(f"  - Mean: {summary['prediction_stats']['mean']:.4f}\n")
            f.write(f"  - Std: {summary['prediction_stats']['std']:.4f}\n")
            f.write(f"  - Min: {summary['prediction_stats']['min']:.4f}\n")
            f.write(f"  - Max: {summary['prediction_stats']['max']:.4f}\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Training summary saved to: {self.results_dir}training_summary_enhanced.txt")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        
        return submission


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" M5 FORECASTING - ENHANCED PREPROCESSING + RECURSIVE")
    print("="*80)
    print("\nKEY ENHANCEMENTS FROM preprocessing.py:")
    print("  • Tweedie objective with variance_power=1.1")
    print("  • Mean and Std encodings for categorical features")
    print("  • Specific lag features from 28 to 42")
    print("  • 12 rolling window combinations: [1,7,14] × [7,14,30,60]")
    
    print("\nADDITIONAL FEATURES KEPT:")
    print("  • Cyclical encoding for temporal patterns")
    print("  • Price momentum and advanced price features")
    print("  • Meta-models for aggregate predictions")
    print("  • Conformal Prediction for uncertainty")
    print("  • Recursive auto-regressive prediction")
    
    print("\n" + "="*80 + "\n")
    
    # Initialize and run
    forecaster = M5ForecasterEnhanced(data_path='./data/', alpha=0.1)
    submission = forecaster.run_full_pipeline()
    
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE")
    print("="*70)
    print("Based on enhanced preprocessing + recursive prediction:")
    print("  • Combines best of both approaches")
    print("  • Tweedie loss for better handling of zero sales")
    print("  • Rich feature set with mean/std encodings")
    print("  • Recursive prediction captures temporal dependencies")
    print("  • Uncertainty-weighted ensemble for robustness")
    print("="*70 + "\n")
    
    print("\n" + "="*70)
    print("SAVED ARTIFACTS")
    print("="*70)
    print(f"Models saved to: {forecaster.model_dir}")
    print(f"Results saved to: {forecaster.results_dir}")
    print("\nTo use saved models:")
    print("  forecaster = M5ForecasterEnhanced(data_path='./data/')")
    print("  forecaster.predict_from_saved_models()")
    print("="*70 + "\n")
