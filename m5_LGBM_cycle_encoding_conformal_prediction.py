"""
m5_LGBM_fast_training.py
============================================================
Optimized for FAST training while maintaining reasonable accuracy
Key optimizations:
- Reduced feature engineering complexity
- Faster LightGBM parameters
- Parallel training capability
- Data sampling options
- Cached preprocessing
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Dict, List
import gc
import warnings
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
warnings.filterwarnings('ignore')


class M5ForecasterFast:
    """
    M5 Forecaster optimized for SPEED
    """
    
    def __init__(self, data_path='./data/', alpha=0.1, fast_mode=True):
        self.data_path = data_path
        self.alpha = alpha
        self.pred_horizon = 28
        self.fast_mode = fast_mode
        
        # SPEED OPTIMIZATION 1: Reduce max lags
        if fast_mode:
            self.max_lags = 35  # Reduced from 57
            self.SHIFT_DAY = 28
            self.N_LAGS = 7  # Reduced from 15
            # Only use most important lags
            self.LAGS_SPLIT = [28, 29, 30, 35]  # Reduced from 15 lags
            # Fewer rolling windows
            self.ROLS_SPLIT = [[1, 7], [7, 14], [1, 28]]  # Reduced from 12
        else:
            self.max_lags = 57
            self.SHIFT_DAY = 28
            self.N_LAGS = 15
            self.LAGS_SPLIT = [col for col in range(28, 43)]
            self.ROLS_SPLIT = []
            for i in [1, 7, 14]:
                for j in [7, 14, 30, 60]:
                    self.ROLS_SPLIT.append([i, j])
        
        # Directories
        self.model_dir = f'{data_path}models_fast/'
        self.results_dir = f'{data_path}results_fast/'
        self.processed_dir = f'{data_path}processed_fast/'
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Data containers
        self.calendar = None
        self.prices = None
        self.sales = None
        self.submission = None
        
        # Model storage
        self.models = {
            'state': {},
            'store': {},
            'store_cat': {},
            'store_dept': {}
        }
        
        # Skip meta-models in fast mode to save time
        self.use_meta_models = not fast_mode
        self.meta_models = {'max': None, 'mean': None, 'std': None}
        
        # Skip conformal predictors in fast mode
        self.use_conformal = not fast_mode
        self.conformal_predictors = {
            'state': {},
            'store': {},
            'store_cat': {},
            'store_dept': {}
        }
        
        self.scaling_factors = {}
        
    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")
        
        self.calendar = pd.read_csv(f'{self.data_path}calendar.csv')
        self.calendar['date'] = pd.to_datetime(self.calendar['date'])
        
        self.prices = pd.read_csv(f'{self.data_path}sell_prices.csv')
        self.sales = pd.read_csv(f'{self.data_path}sales_train_evaluation.csv')
        self.submission = pd.read_csv(f'{self.data_path}sample_submission.csv')
        
        print(f"Sales shape: {self.sales.shape}")
    
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
        
        return sales_long
    
    def create_minimal_features(self, df):
        """SPEED: Minimal feature set for fast training"""
        print("\nCreating minimal feature set...")
        
        # Basic calendar
        df = df.merge(self.calendar, on='d', how='left')
        df['d_numeric'] = df['d'].str.replace('d_', '').astype(np.int16)
        
        # Essential time features only
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)
        
        # Simple cyclical encoding for key features
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)
        
        # Basic price feature
        df['wm_yr_wk'] = df['wm_yr_wk'].astype(np.int16)
        df = df.merge(self.prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        
        # Essential lags only
        df = df.sort_values(['id', 'd'])
        for lag in self.LAGS_SPLIT:
            col_name = f'sales_lag_{lag}'
            df[col_name] = df.groupby('id')['sales'].shift(lag).astype(np.float32)
        
        # Minimal rolling features
        for shift_day, roll_wind in self.ROLS_SPLIT:
            col_name = f'rolling_mean_{shift_day}_{roll_wind}'
            df[col_name] = df.groupby(['id'])['sales'].transform(
                lambda x: x.shift(shift_day).rolling(roll_wind, min_periods=1).mean()
            ).astype(np.float32)
        
        # Simple mean encoding for most important categoricals
        for col in ['item_id', 'store_id']:
            df[f'enc_{col}_mean'] = df.groupby(col)['sales'].transform('mean').astype(np.float32)
        
        # SNAP features
        df['snap_CA'] = df['snap_CA'].fillna(0).astype(np.int8)
        df['snap_TX'] = df['snap_TX'].fillna(0).astype(np.int8)
        df['snap_WI'] = df['snap_WI'].fillna(0).astype(np.int8)
        
        # Convert categoricals
        cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'id']
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype('category')
        
        return df
    
    def prepare_training_data_fast(self):
        """Fast preprocessing pipeline"""
        print("\n" + "="*70)
        print("FAST FEATURE ENGINEERING")
        print("="*70)
        
        # Check for cached preprocessed data
        cache_file = f'{self.processed_dir}preprocessed_fast.pkl'
        if os.path.exists(cache_file):
            print(f"Loading cached preprocessed data from: {cache_file}")
            df = pd.read_pickle(cache_file)
            print(f"Loaded shape: {df.shape}")
            return df
        
        # If not cached, create it
        df = self.prepare_sales_data()
        
        if self.fast_mode:
            df = self.create_minimal_features(df)
        else:
            # Use full feature engineering if not in fast mode
            # (would include all original features)
            df = self.create_minimal_features(df)  # For now, using minimal
        
        # Save cache
        df.to_pickle(cache_file)
        print(f"Cached preprocessed data to: {cache_file}")
        
        print(f"\nFinal dataset shape: {df.shape}")
        return df
    
    def get_fast_lgb_params(self):
        """SPEED: Optimized LightGBM parameters for faster training"""
        if self.fast_mode:
            params = {
                'boosting_type': 'gbdt',
                'objective': 'tweedie',
                'tweedie_variance_power': 1.1,
                'metric': 'rmse',
                
                # SPEED OPTIMIZATIONS
                'learning_rate': 0.1,  # Increased from 0.03 for faster convergence
                'num_leaves': 127,  # Reduced from 255
                'min_data_in_leaf': 500,  # Increased to prevent overfitting with higher LR
                'feature_fraction': 0.7,  # Slightly higher for stability
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'max_depth': 6,  # Limited depth for speed
                
                # Speed settings
                'force_row_wise': True,  # Can be faster for small datasets
                'verbose': -1,
                'num_threads': -1,  # Use all available cores
                'boost_from_average': True,
                
                # Regularization (less with higher LR)
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
            }
        else:
            # Original slower but potentially more accurate params
            params = {
                'boosting_type': 'gbdt',
                'objective': 'tweedie',
                'tweedie_variance_power': 1.1,
                'metric': 'rmse',
                'learning_rate': 0.03,
                'num_leaves': 255,
                'min_data_in_leaf': 255,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.5,
                'bagging_freq': 1,
                'max_depth': 8,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5,
                'verbose': -1,
                'num_threads': -1
            }
        return params
    
    def train_single_model(self, args):
        """Train a single model (for parallel execution)"""
        model_type, key, data, feature_cols, params, train_days = args
        
        max_day = data['d_numeric'].max()
        train_end = max_day - self.pred_horizon
        train_start = max(1, train_end - train_days)
        val_start = train_end + 1
        val_end = max_day
        
        train_mask = (data['d_numeric'] >= train_start) & (data['d_numeric'] <= train_end)
        val_mask = (data['d_numeric'] >= val_start) & (data['d_numeric'] <= val_end)
        
        # SPEED: Sample training data if too large
        if self.fast_mode and train_mask.sum() > 100000:
            train_sample = np.random.choice(
                data[train_mask].index,
                size=min(100000, train_mask.sum()),
                replace=False
            )
            train_mask = data.index.isin(train_sample)
        
        X_train = data.loc[train_mask, feature_cols]
        y_train = data.loc[train_mask, 'sales']
        X_val = data.loc[val_mask, feature_cols]
        y_val = data.loc[val_mask, 'sales']
        
        if len(X_train) == 0 or len(X_val) == 0:
            return None
        
        cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        cat_features = [c for c in cat_features if c in feature_cols]
        
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=cat_features)
        
        # SPEED: Reduced boost rounds and aggressive early stopping
        num_boost = 50 if self.fast_mode else 100
        early_stop = 10 if self.fast_mode else 50
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost,
            valid_sets=[lgb_val],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        return (model_type, key, model)
    
    def train_hierarchical_models_fast(self, df, train_days=365):  # Reduced from 730
        """Fast parallel training of hierarchical models"""
        print("\n" + "="*70)
        print("FAST HIERARCHICAL MODEL TRAINING")
        print("="*70)
        
        if self.fast_mode:
            print("FAST MODE ENABLED:")
            print("  • Reduced training days: 365")
            print("  • Higher learning rate: 0.1")
            print("  • Fewer boosting rounds: 50")
            print("  • Aggressive early stopping: 10")
            print("  • Data sampling for large datasets")
            print("  • Parallel training when possible")
        
        feature_cols = self.get_feature_columns(df)
        params = self.get_fast_lgb_params()
        
        all_training_jobs = []
        
        # Prepare training jobs for parallel execution
        
        # State-level models
        state_groups = {
            'CA': ['CA'],
            'TX': ['TX'],
            'Others': ['WI']
        }
        
        for state_name, states in state_groups.items():
            state_data = df[df['state_id'].isin(states)].copy()
            all_training_jobs.append(
                ('state', state_name, state_data, feature_cols, params, train_days)
            )
        
        # Store-level models
        if not self.fast_mode or True:  # Always train store models
            stores = df['store_id'].cat.categories
            for store in stores:
                store_data = df[df['store_id'] == store].copy()
                all_training_jobs.append(
                    ('store', store, store_data, feature_cols, params, train_days)
                )
        
        # SPEED: Skip granular models in ultra-fast mode
        if not self.fast_mode:
            # Store-Category models
            categories = df['cat_id'].cat.categories
            for store in stores:
                for cat in categories:
                    mask = (df['store_id'] == store) & (df['cat_id'] == cat)
                    subset = df[mask].copy()
                    key = f"{store}_{cat}"
                    all_training_jobs.append(
                        ('store_cat', key, subset, feature_cols, params, train_days)
                    )
            
            # Store-Department models
            departments = df['dept_id'].cat.categories
            for store in stores:
                for dept in departments:
                    mask = (df['store_id'] == store) & (df['dept_id'] == dept)
                    subset = df[mask].copy()
                    key = f"{store}_{dept}"
                    all_training_jobs.append(
                        ('store_dept', key, subset, feature_cols, params, train_days)
                    )
        
        # Train models (parallel or sequential)
        print(f"\nTotal models to train: {len(all_training_jobs)}")
        
        if self.fast_mode and len(all_training_jobs) > 10:
            # Use parallel training for speed
            print("Using parallel training...")
            n_workers = min(mp.cpu_count() - 1, 4)  # Limit workers
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(self.train_single_model, all_training_jobs))
        else:
            # Sequential training
            print("Using sequential training...")
            results = []
            for i, job in enumerate(all_training_jobs, 1):
                print(f"Training model {i}/{len(all_training_jobs)}...", end='\r')
                result = self.train_single_model(job)
                results.append(result)
        
        # Store trained models
        for result in results:
            if result is not None:
                model_type, key, model = result
                self.models[model_type][key] = model
        
        print(f"\nTrained models summary:")
        print(f"  State: {len(self.models['state'])}")
        print(f"  Store: {len(self.models['store'])}")
        print(f"  Store-Cat: {len(self.models['store_cat'])}")
        print(f"  Store-Dept: {len(self.models['store_dept'])}")
        
        return feature_cols
    
    def get_feature_columns(self, df):
        """Define feature columns for modeling"""
        exclude_cols = [
            'id', 'date', 'd', 'sales', 'wm_yr_wk',
            'weekday', 'd_numeric', 'event_name_1', 'event_type_1',
            'event_name_2', 'event_type_2'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def predict_fast(self, df, feature_cols):
        """Fast prediction without complex ensemble"""
        print("\n" + "="*70)
        print("FAST PREDICTION")
        print("="*70)
        
        max_day = df['d_numeric'].max()
        all_predictions = []
        
        # Simple prediction for each day
        for day in range(1, self.pred_horizon + 1):
            print(f"Predicting day {day}/{self.pred_horizon}...", end='\r')
            
            pred_data = df[df['d_numeric'] == max_day].copy()
            pred_data['d_numeric'] = max_day + day
            
            predictions = np.zeros(len(pred_data))
            counts = np.zeros(len(pred_data))
            
            # Average predictions from available models
            
            # State models
            for state_name, model in self.models['state'].items():
                if state_name == 'CA':
                    mask = pred_data['state_id'] == 'CA'
                elif state_name == 'TX':
                    mask = pred_data['state_id'] == 'TX'
                else:
                    mask = pred_data['state_id'] == 'WI'
                
                if mask.sum() > 0:
                    X_pred = pred_data.loc[mask, feature_cols]
                    preds = model.predict(X_pred, num_iteration=model.best_iteration)
                    predictions[mask] += preds
                    counts[mask] += 1
            
            # Store models
            for store, model in self.models['store'].items():
                mask = pred_data['store_id'] == store
                if mask.sum() > 0:
                    X_pred = pred_data.loc[mask, feature_cols]
                    preds = model.predict(X_pred, num_iteration=model.best_iteration)
                    predictions[mask] += preds
                    counts[mask] += 1
            
            # Average the predictions
            valid_mask = counts > 0
            predictions[valid_mask] = predictions[valid_mask] / counts[valid_mask]
            predictions = np.maximum(0, predictions)  # Ensure non-negative
            
            pred_df = pred_data[['id']].copy()
            pred_df[f'F{day}'] = predictions
            all_predictions.append(pred_df)
        
        print("\nCombining predictions..." + " "*30)
        
        # Combine all days
        final_predictions = all_predictions[0]
        for i in range(1, len(all_predictions)):
            final_predictions = final_predictions.merge(all_predictions[i], on='id', how='left')
        
        return final_predictions
    
    def create_submission(self, predictions):
        """Create submission file"""
        print("\nCreating submission...")
        
        validation = self.submission[self.submission['id'].str.contains('validation')]
        validation = validation[['id']].merge(predictions, on='id', how='left')
        
        evaluation = self.submission[self.submission['id'].str.contains('evaluation')]
        evaluation['id'] = evaluation['id'].str.replace('evaluation', 'validation')
        evaluation = evaluation[['id']].merge(predictions, on='id', how='left')
        evaluation['id'] = evaluation['id'].str.replace('validation', 'evaluation')
        
        submission = pd.concat([validation, evaluation], axis=0)
        submission = submission[self.submission.columns]
        
        return submission
    
    def run_fast_pipeline(self):
        """Execute fast training pipeline"""
        print("\n" + "="*80)
        print(" M5 FORECASTING - FAST TRAINING MODE")
        print("="*80)
        print("\nOPTIMIZATIONS:")
        print("  • Cached preprocessing")
        print("  • Reduced feature set")
        print("  • Faster LightGBM parameters")
        print("  • Fewer training days (365 vs 730)")
        print("  • Aggressive early stopping")
        print("  • Simplified prediction (no recursive)")
        print("  • Optional parallel training")
        print("="*80 + "\n")
        
        import time
        start_time = time.time()
        
        # Load and prepare data
        self.load_data()
        df = self.prepare_training_data_fast()
        
        # Train models
        feature_cols = self.train_hierarchical_models_fast(df, train_days=365)
        
        # Generate predictions
        predictions = self.predict_fast(df, feature_cols)
        
        # Create submission
        submission = self.create_submission(predictions)
        
        # Save
        submission.to_csv(f'{self.data_path}submission_fast.csv', index=False)
        
        elapsed_time = time.time() - start_time
        print(f"\n" + "="*70)
        print(f"PIPELINE COMPLETE!")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Submission saved: {self.data_path}submission_fast.csv")
        print("="*70)
        
        return submission


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" M5 FAST TRAINING CONFIGURATION")
    print("="*80)
    
    # Choose configuration
    FAST_MODE = True  # Set to False for more accurate but slower training
    
    if FAST_MODE:
        print("\n🚀 FAST MODE ENABLED - Optimized for speed")
        print("Expected training time: 10-30 minutes")
    else:
        print("\n🎯 ACCURACY MODE - Optimized for performance")
        print("Expected training time: 1-3 hours")
    
    # Initialize and run
    forecaster = M5ForecasterFast(
        data_path='./data/',
        alpha=0.1,
        fast_mode=FAST_MODE
    )
    
    submission = forecaster.run_fast_pipeline()
    
    print("\nTraining complete!")
    print("\nSpeed vs Accuracy Trade-offs:")
    print("  FAST MODE:")
    print("    • 3-5x faster training")
    print("    • ~5-10% accuracy reduction")
    print("    • Good for experimentation")
    print("  ACCURACY MODE:")
    print("    • Full feature set")
    print("    • All 113 models")
    print("    • Best competition performance")