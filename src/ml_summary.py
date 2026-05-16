import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

class MLSummary:
    def __init__(self, df, target_col, problem_type):
        self.df = df
        self.target_col = target_col
        self.problem_type = problem_type
        self.metrics = None
        self.model = None
    
    def train_and_evaluate(self, test_size=0.2, random_state=42):
        """Обучает модель и возвращает метрики"""
        try:
            # Подготовка данных
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col]
            
            # Кодируем категориальные признаки
            X = pd.get_dummies(X, drop_first=True)
            
            # Кодируем целевую колонку если это классификация и текст
            if self.problem_type == "classification" and y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                y = LabelEncoder().fit_transform(y)
            
            # Разделяем на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Обучаем модель
            if self.problem_type == "classification":
                self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                
                self.metrics = {
                    "accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    "f1_score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
                }
            
            elif self.problem_type == "regression":
                self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                
                self.metrics = {
                    "mae": round(mean_absolute_error(y_test, y_pred), 4),
                    "mse": round(mean_squared_error(y_test, y_pred), 4),
                    "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    "r2": round(r2_score(y_test, y_pred), 4)
                }
            
            return self.metrics
        
        except Exception as e:
            self.metrics = {"error": str(e)}
            return self.metrics
    
    def get_metrics_download_link(self):
        """Возвращает JSON строку для скачивания"""
        if not self.metrics:
            return None
        
        if "error" in self.metrics:
            return None
        
        data = {
            "target_column": self.target_col,
            "problem_type": self.problem_type,
            "metrics": self.metrics
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def get_metrics_markdown(self):
        """Возвращает метрики для отчёта"""
        if not self.metrics or "error" in self.metrics:
            return "⚠️ Метрики не рассчитаны"
        
        if self.problem_type == "classification":
            return f"""
| Метрика | Значение |
|---------|----------|
| Accuracy | {self.metrics['accuracy']} |
| Precision | {self.metrics['precision']} |
| Recall | {self.metrics['recall']} |
| F1-score | {self.metrics['f1_score']} |
"""
        else:
            return f"""
| Метрика | Значение |
|---------|----------|
| MAE | {self.metrics['mae']} |
| MSE | {self.metrics['mse']} |
| RMSE | {self.metrics['rmse']} |
| R² | {self.metrics['r2']} |
"""