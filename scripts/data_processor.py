from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class DataPreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self,stores_df,features_df,lag_features=[1,2,4,8,12],rolling_windows=[4,8,12]):
        self.stores_df=stores_df
        self.features_df=features_df
        self.lag_features=lag_features
        self.rolling_windows=rolling_windows
        
        self.label_encoder=None
        self.store_dept_rolling_stats=None
        self.overall_rolling_averages=None
        self.lag_means={}
        self.rolling_means={}

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Target variable y is required")
        X_fit=X.copy()
        
        X_processed=self._apply_basic_preprocessing(X_fit)
        
        self.label_encoder=LabelEncoder()
        self.label_encoder.fit(X_processed['Type'])
        
        X_processed['Type_le']=self.label_encoder.transform(X_processed['Type'])
        
        X_processed=self._additional_time_feat(X_processed)
        
        X_processed['Weekly_Sales']=y
        X_processed=X_processed.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
        
        self._compute_lag_statistics(X_processed)
        self._compute_rolling_statistics(X_processed)

        return self

    def transform(self, X):
        X_processed=X.copy()
        
        X_processed=self._apply_basic_preprocessing(X_processed)
        
        X_processed['Type_le']=self.label_encoder.transform(X_processed['Type'])
        
        X_processed=self._additional_time_feat(X_processed)
        
        X_processed=X_processed.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
        
        has_target='Weekly_Sales' in X_processed.columns
        
        if has_target:
            X_processed=self._create_real_lag_features(X_processed)
            X_processed=self._create_real_rolling_features(X_processed)
        else:
            X_processed=self._create_approx_lag_features(X_processed)
            X_processed=self._create_approx_rolling_features(X_processed)
        
        X_processed=self._add_enhanced_features(X_processed)
        
        cols_to_remove=['Date', 'Type']
        if 'Weekly_Sales' in X_processed.columns:
            cols_to_remove.append('Weekly_Sales')
        X_processed=X_processed.drop([col for col in cols_to_remove if col in X_processed.columns], axis=1)
        
        X_processed=X_processed.fillna(0)
        
        return X_processed

    def _apply_basic_preprocessing(self, X_processed):
        X_processed=X_processed.merge(self.stores_df, on='Store', how='left')
        X_processed=X_processed.merge(self.features_df, on=['Store', 'Date'], how='left')

        X_processed['Date']=pd.to_datetime(X_processed['Date'])

        if 'IsHoliday_x' in X_processed.columns and 'IsHoliday_y' in X_processed.columns:
            X_processed['IsHoliday'] = X_processed['IsHoliday_y'].fillna(X_processed['IsHoliday_x'])
            X_processed = X_processed.drop(['IsHoliday_x', 'IsHoliday_y'], axis=1)

        num_cols=['CPI', 'Fuel_Price', 'Unemployment', 'Temperature']
        self._fill_numericals(num_cols, X_processed)

        markdowns=[col for col in X_processed.columns if 'MarkDown' in col]
        for col in markdowns:
            X_processed[col]=X_processed[col].fillna(0)

        X_processed['IsHoliday']=X_processed['IsHoliday'].fillna(False)

        removed_cols=['MarkDown1', 'MarkDown3', 'MarkDown5']    #bevri missing value
        X_processed=X_processed.drop([col for col in removed_cols if col in X_processed.columns], axis=1)

        return X_processed


    def _fill_numericals(self, num_cols, data):
        for col in num_cols:
            if col in data.columns:
                missing_count=data[col].isnull().sum()
                if missing_count > 0:
                    # data[col]=data.groupby('Store')[col].ffill()
                    data[col]=data[col].fillna(data[col].median())

    def _additional_time_feat(self, input_df):
        input_df=input_df.copy()
        
        input_df['Day']=input_df['Date'].dt.day.astype('int64')
        input_df['Week']=input_df['Date'].dt.isocalendar().week.astype('int64')
        input_df['Month']=input_df['Date'].dt.month.astype('int64')
        input_df['Year']=input_df['Date'].dt.year.astype('int64')

        input_df['Quarter']=input_df['Date'].dt.quarter.astype('int64')
        input_df['DayOfYear']=input_df['Date'].dt.dayofyear.astype('int64')
        input_df['WeekOfMonth']=((input_df['Date'].dt.day - 1) // 7+1).astype('int64')
        
        input_df['IsDecember']=(input_df['Month'] == 12).astype(int)
        input_df['IsQ4']=(input_df['Quarter'] == 4).astype(int)

        return input_df

    def _compute_lag_statistics(self, X_processed):
        for lag in self.lag_features:
            lag_means=X_processed.groupby(['Store', 'Dept'])['Weekly_Sales'].mean()
            self.lag_means[f'lag_{lag}']=lag_means.to_dict()

    def _compute_rolling_statistics(self, X_processed):
        for window in self.rolling_windows:
            rolling_values=(X_processed.groupby(['Store', 'Dept'])['Weekly_Sales'].rolling(window=window, min_periods=1).mean().groupby(['Store', 'Dept']).last())
            
            self.rolling_means[f'rolling_mean_{window}']=rolling_values.to_dict()
            
            rolling_std=(X_processed.groupby(['Store', 'Dept'])['Weekly_Sales'].rolling(window=window, min_periods=1).std().groupby(['Store', 'Dept']).last())
            self.rolling_means[f'rolling_std_{window}']=rolling_std.to_dict()

    def _create_real_lag_features(self, X_processed):
        for lag in self.lag_features:
            lag_series=X_processed.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
            group_mean_series=X_processed.groupby(['Store', 'Dept'])['Weekly_Sales'].transform('mean')

            lag_series=lag_series.fillna(group_mean_series)
            
            X_processed[f'lag_{lag}']=lag_series            
            X_processed[f'growth_{lag}w']=X_processed['Weekly_Sales']/(X_processed[f'lag_{lag}']+1)
            # X_processed[f'lag_diff_{lag}']=X_processed['Weekly_Sales'] - X_processed[f'lag_{lag}']
            
        return X_processed

    def _create_real_rolling_features(self, X_processed):        
        for window in self.rolling_windows:
            rolling_mean=(X_processed.groupby(['Store', 'Dept'])['Weekly_Sales'].rolling(window=window, min_periods=1).mean().reset_index(level=[0, 1], drop=True))
            rolling_std=(X_processed.groupby(['Store', 'Dept'])['Weekly_Sales'].rolling(window=window, min_periods=1).std().reset_index(level=[0, 1], drop=True).fillna(0))

            X_processed[f'rolling_mean_{window}']=rolling_mean.values
            X_processed[f'rolling_std_{window}']=rolling_std.values
            X_processed[f'rolling_cv_{window}']=X_processed[f'rolling_std_{window}']/(X_processed[f'rolling_mean_{window}']+1)
            X_processed[f'sales_vs_mean_{window}']=X_processed['Weekly_Sales']/(X_processed[f'rolling_mean_{window}']+1)
            
        return X_processed

    def _create_approx_lag_features(self, X_processed):        
        for lag in self.lag_features:
            if f'lag_{lag}' in self.lag_means:
                lag_values=[]
                for _, row in X_processed.iterrows():
                    store_dept_key=(row['Store'], row['Dept'])
                    lag_val=self.lag_means[f'lag_{lag}'].get(store_dept_key, 15000)
                    lag_values.append(lag_val)
                
                X_processed[f'lag_{lag}']=lag_values
                X_processed[f'growth_{lag}w']=1.0
                # X_processed[f'lag_diff_{lag}']=0.0
            else:
                X_processed[f'lag_{lag}']=10000
                X_processed[f'growth_{lag}w']=1.0
                # X_processed[f'lag_diff_{lag}']=0.0
                
        return X_processed

    def _create_approx_rolling_features(self, X_processed):
        for window in self.rolling_windows:
            if f'rolling_mean_{window}' in self.rolling_means:
                rolling_values=[]
                for _, row in X_processed.iterrows():
                    store_dept_key=(row['Store'], row['Dept'])
                    rolling_val=self.rolling_means[f'rolling_mean_{window}'].get(store_dept_key, 15000)
                    rolling_values.append(rolling_val)
                
                X_processed[f'rolling_mean_{window}']=rolling_values
                X_processed[f'rolling_std_{window}']=0
                X_processed[f'rolling_cv_{window}']=0
                X_processed[f'sales_vs_mean_{window}']=1.0
            else:
                X_processed[f'rolling_mean_{window}']=10000
                X_processed[f'rolling_std_{window}']=0
                X_processed[f'rolling_cv_{window}']=0
                X_processed[f'sales_vs_mean_{window}']=1.0
                
        return X_processed

    def _add_enhanced_features(self, X_processed):
        markdown_cols=[col for col in X_processed.columns if 'MarkDown' in col]
        if markdown_cols:
            X_processed['total_markdown']=X_processed[markdown_cols].sum(axis=1)
            X_processed['markdown_count']=(X_processed[markdown_cols] > 0).sum(axis=1)
            
            X_processed['markdown_per_dept']=X_processed['total_markdown']/(X_processed['Dept']+1)

        if 'Store' in X_processed.columns and 'Dept' in X_processed.columns:
            store_dept_combined=X_processed['Store'].astype(str)+'_'+X_processed['Dept'].astype(str)
            store_dept_encoder=LabelEncoder()
            X_processed['Store_Dept_encoded']=store_dept_encoder.fit_transform(store_dept_combined.fillna('NaN'))
            
            if 'Size' in X_processed.columns:
                X_processed['store_dept_size_ratio']=X_processed['Size']/(X_processed['Dept']+1)

        if 'IsHoliday' in X_processed.columns:
            X_processed['holiday_weight']=X_processed['IsHoliday'].map({True: 5, False: 1})
            X_processed['holiday_dept_interaction']=X_processed['IsHoliday'].astype(int)*X_processed['Dept']
            
            if 'Type_le' in X_processed.columns:
                X_processed['holiday_type_interaction']=X_processed['IsHoliday'].astype(int)*X_processed['Type_le']


        if 'CPI' in X_processed.columns and 'Unemployment' in X_processed.columns:
            X_processed['economic_pressure']=X_processed['CPI']*X_processed['Unemployment']
            X_processed['cpi_unemployment_ratio']=X_processed['CPI']/(X_processed['Unemployment']+0.1)


        if 'Temperature' in X_processed.columns:
            X_processed['temp_squared']=X_processed['Temperature'] ** 2
            X_processed['is_cold']=(X_processed['Temperature'] < 50).astype(int)
            X_processed['is_hot']=(X_processed['Temperature'] > 80).astype(int)


        if 'Fuel_Price' in X_processed.columns and 'Size' in X_processed.columns:
            X_processed['fuel_cost_impact']=X_processed['Fuel_Price']*X_processed['Size']/100000


        if 'Type_le' in X_processed.columns and 'Size' in X_processed.columns:
            X_processed['type_size_interaction']=X_processed['Type_le']*X_processed['Size']/1000

        return X_processed