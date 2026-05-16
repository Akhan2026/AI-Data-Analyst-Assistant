import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, df):
        self.df = df.copy()
        self.created_features = []
        self.removed_features = []
    
    def extract_datetime_features(self, drop_original=True):
        for col in self.df.columns:
            try:
                dt = pd.to_datetime(self.df[col])
                self.df[f'{col}_year'] = dt.dt.year
                self.df[f'{col}_month'] = dt.dt.month
                self.df[f'{col}_day'] = dt.dt.day
                self.df[f'{col}_dayofweek'] = dt.dt.dayofweek
                self.created_features.extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'])
                if drop_original:
                    self.df.drop(columns=[col], inplace=True)
                    self.removed_features.append(col)
            except:
                pass
        return self.df
    
    def create_interaction_features(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4]
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    self.df[f'{numeric_cols[i]}_x_{numeric_cols[j]}'] = self.df[numeric_cols[i]] * self.df[numeric_cols[j]]
                    self.created_features.append(f'{numeric_cols[i]}_x_{numeric_cols[j]}')
        return self.df
    
    def encode_categorical(self, max_categories=10):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if self.df[col].nunique() <= max_categories:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.created_features.extend(dummies.columns.tolist())
                self.df.drop(columns=[col], inplace=True)
                self.removed_features.append(col)
        return self.df
    
    def get_summary(self):
        return {
            "created_features": self.created_features[:10],
            "total_created": len(self.created_features),
            "total_removed": len(self.removed_features),
            "original_shape": self.df.shape,
        }