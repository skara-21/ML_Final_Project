import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error

def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri("")
    
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass
    
    mlflow.set_experiment(experiment_name)

def load_walmart_data():
    train_df = pd.read_csv('/content/drive/MyDrive/ML_final_project/data/train.csv')
    test_df = pd.read_csv('/content/drive/MyDrive/ML_final_project/data/test.csv')
    stores_df = pd.read_csv('/content/drive/MyDrive/ML_final_project/data/stores.csv')
    features_df = pd.read_csv('/content/drive/MyDrive/ML_final_project/data/features.csv')
    
    return train_df, test_df, stores_df, features_df

def explore_data(train_df, test_df, stores_df, features_df):
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    return {
        'unique_stores': train_df['Store'].nunique(),
        'unique_departments': train_df['Dept'].nunique(),
        'date_range_days': (train_df['Date'].max() - train_df['Date'].min()).days
    }

class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, store_id=1, dept_id=1, freq='W'):
        self.store_id = store_id
        self.dept_id = dept_id
        self.freq = freq
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not pd.api.types.is_datetime64_any_dtype(X['Date']):
            X = X.copy()
            X['Date'] = pd.to_datetime(X['Date'])
        
        store_dept_data = X[(X['Store'] == self.store_id) & (X['Dept'] == self.dept_id)].copy()
        
        if store_dept_data.empty:
            raise ValueError(f"No data found for Store {self.store_id}, Dept {self.dept_id}")
        
        store_dept_data = store_dept_data.sort_values('Date')
        store_dept_data.set_index('Date', inplace=True)
        
        if self.freq == 'W':
            store_dept_data = store_dept_data.resample('W').agg({
                'Weekly_Sales': 'sum',
                'IsHoliday': 'max'
            })
        
        return store_dept_data

def prepare_time_series_data(train_df, stores_df, features_df, store_id=1, dept_id=1):
    if not pd.api.types.is_datetime64_any_dtype(train_df['Date']):
        train_df = train_df.copy()
        train_df['Date'] = pd.to_datetime(train_df['Date'])
    
    ts_data = train_df[(train_df['Store'] == store_id) & (train_df['Dept'] == dept_id)].copy()
    
    if ts_data.empty:
        raise ValueError(f"No data found for Store {store_id}, Dept {dept_id}")
    
    ts_data = ts_data.sort_values('Date')
    ts_data.set_index('Date', inplace=True)
    
    if features_df is not None and not features_df.empty:
        features_df = features_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(features_df['Date']):
            features_df['Date'] = pd.to_datetime(features_df['Date'])
        
        store_features = features_df[features_df['Store'] == store_id]
        if not store_features.empty:
            ts_data = ts_data.merge(store_features, left_index=True, right_on='Date', how='left')
            ts_data.set_index('Date', inplace=True)
    
    if ts_data['Weekly_Sales'].isnull().sum() > 0:
        ts_data['Weekly_Sales'].fillna(method='ffill', inplace=True)
    
    return ts_data

def check_stationarity(timeseries, title):
    adf_result = adfuller(timeseries.dropna())
    
    if adf_result[1] <= 0.05:
        adf_stationary = True
    else:
        adf_stationary = False
    
    kpss_result = kpss(timeseries.dropna())
    
    if kpss_result[1] <= 0.05:
        kpss_stationary = False
    else:
        kpss_stationary = True
    
    return {
        'adf_stationary': adf_stationary,
        'kpss_stationary': kpss_stationary,
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'kpss_statistic': kpss_result[0],
        'kpss_pvalue': kpss_result[1]
    }

def plot_time_series_analysis(ts_data, title_prefix=""):
    fig = plt.figure(figsize=(20, 15))
    
    plt.subplot(3, 2, 1)
    plt.plot(ts_data.index, ts_data['Weekly_Sales'])
    plt.title(f'{title_prefix} Original Time Series')
    plt.ylabel('Weekly Sales')
    plt.grid(True, alpha=0.3)
    
    try:
        decomposition = seasonal_decompose(ts_data['Weekly_Sales'], period=52, model='additive')
        
        plt.subplot(3, 2, 2)
        plt.plot(decomposition.trend.index, decomposition.trend)
        plt.title(f'{title_prefix} Trend Component')
        plt.ylabel('Trend')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 2, 3)
        plt.plot(decomposition.seasonal.index, decomposition.seasonal)
        plt.title(f'{title_prefix} Seasonal Component')
        plt.ylabel('Seasonal')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 2, 4)
        plt.plot(decomposition.resid.index, decomposition.resid)
        plt.title(f'{title_prefix} Residual Component')
        plt.ylabel('Residual')
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"Could not perform seasonal decomposition: {e}")
    
    plt.subplot(3, 2, 5)
    plot_acf(ts_data['Weekly_Sales'].dropna(), lags=40, ax=plt.gca())
    plt.title(f'{title_prefix} Autocorrelation Function')
    
    plt.subplot(3, 2, 6)
    plot_pacf(ts_data['Weekly_Sales'].dropna(), lags=40, ax=plt.gca())
    plt.title(f'{title_prefix} Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def time_series_train_test_split(ts_data, test_size=0.2):
    n_obs = len(ts_data)
    split_idx = int(n_obs * (1 - test_size))
    
    train_data = ts_data.iloc[:split_idx]
    test_data = ts_data.iloc[split_idx:]
    
    return train_data, test_data

def plot_forecast_results(train_data, test_data, forecast, title="Forecast Results"):
    plt.figure(figsize=(15, 8))
    
    plt.plot(train_data.index, train_data['Weekly_Sales'], 
             label='Training Data', color='blue', alpha=0.7)
    
    plt.plot(test_data.index, test_data['Weekly_Sales'], 
             label='Actual Test Data', color='green', linewidth=2)
    
    if isinstance(forecast, pd.Series):
        forecast_index = test_data.index[:len(forecast)]
    else:
        forecast_index = test_data.index[:len(forecast)]
    
    plt.plot(forecast_index, forecast[:len(forecast_index)], 
             label='Forecast', color='red', linewidth=2, linestyle='--')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_model_results(model_name, best_params, metrics, results_df, artifacts_path="./"):
    params_df = pd.DataFrame([{
        'model_name': model_name,
        'best_params': str(best_params),
        **metrics
    }])
    
    params_path = f"{artifacts_path}{model_name}_best_params.csv"
    params_df.to_csv(params_path, index=False)
    
    results_path = f"{artifacts_path}{model_name}_grid_search_results.csv"
    results_df.to_csv(results_path, index=False)
    
    return params_path, results_path

class WalmartTimeSeriesPipeline(Pipeline):
    def __init__(self, preprocessor, model, store_id=1, dept_id=1):
        self.store_id = store_id
        self.dept_id = dept_id
        super().__init__([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    
    def fit(self, X, y=None):
        X_processed = self.named_steps['preprocessor'].transform(X)
        
        self.named_steps['model'].fit(X_processed)
        
        return self
    
    def predict(self, X=None, steps=None):
        if steps is None:
            steps = 1
        
        return self.named_steps['model'].predict(steps)

def perform_cross_validation(ts_data, model_class, params, n_splits=5, test_size=0.1):
    scores = []
    n_obs = len(ts_data)
    
    for i in range(n_splits):
        test_start = int(n_obs * (0.9 - i * test_size))
        test_end = int(n_obs * (0.9 - (i-1) * test_size)) if i > 0 else n_obs
        
        if test_start <= 0 or test_end <= test_start:
            continue
        
        train_data = ts_data.iloc[:test_start]
        test_data = ts_data.iloc[test_start:test_end]
        
        if len(train_data) < 10 or len(test_data) < 1:
            continue
        
        try:
            model = model_class(**params)
            fitted_model = model.fit(train_data)
            
            if fitted_model is not None:
                predictions = fitted_model.predict(len(test_data))
                
                metrics = calculate_metrics(test_data['Weekly_Sales'].values, predictions)
                scores.append(metrics)
                
        except Exception as e:
            print(f"Error in CV fold {i}: {e}")
            continue
    
    if not scores:
        return None
    
    avg_scores = {}
    for metric in scores[0].keys():
        avg_scores[f'cv_{metric.lower()}'] = np.mean([score[metric] for score in scores])
        avg_scores[f'cv_{metric.lower()}_std'] = np.std([score[metric] for score in scores])
    
    return avg_scores

def evaluate_model_performance(model, train_data, test_data, model_name):
    try:
        forecast = model.predict(len(test_data))
        
        metrics = calculate_metrics(test_data['Weekly_Sales'].values, forecast)
        
        metrics['model_name'] = model_name
        
        return metrics, forecast
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None, None

def log_model_artifacts(model_name, metrics, params, results_df, forecast=None):
    for key, value in params.items():
        mlflow.log_param(f"{model_name}_{key}", value)
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"{model_name}_{key}", value)
    
    results_path = f"{model_name}_results.csv"
    results_df.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path)
    
    if forecast is not None:
        forecast_df = pd.DataFrame({'forecast': forecast})
        forecast_path = f"{model_name}_forecast.csv"
        forecast_df.to_csv(forecast_path, index=False)
        mlflow.log_artifact(forecast_path)