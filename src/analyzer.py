import pandas as pd
import numpy as np

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.analysis = {}

    def run_full_analysis(self):
        basic_info = self._basic_info()
        missing_values = self._missing_analysis()
        numeric_stats = self._numeric_stats()
        categorical_stats = self._categorical_stats()
        correlations = self._correlation_analysis()
        outliers = self._outlier_analysis()

        self.analysis = {
            "basic_info": basic_info,
            "missing_values": missing_values,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "correlations": correlations,
            "outliers": outliers
        }
        self.analysis["insights"] = self._generate_insights()
        return self.analysis

    def _basic_info(self):
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2,
            "duplicates": self.df.duplicated().sum()
        }

    def _missing_analysis(self):
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        result = {}
        for col in self.df.columns:
            if missing[col] > 0:
                result[col] = {
                    "count": int(missing[col]),
                    "percentage": round(missing_pct[col], 2),
                    "type": str(self.df[col].dtype)
                }
        return result

    def _numeric_stats(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = self.df[numeric_cols].describe().to_dict()
            for col in numeric_cols:
                stats[col]['median'] = self.df[col].median()
            return stats
        return {}

    def _categorical_stats(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        result = {}
        for col in categorical_cols:
            value_counts = self.df[col].value_counts().head(10).to_dict()
            result[col] = {
                "unique_values": self.df[col].nunique(),
                "top_values": value_counts
            }
        return result

    def _correlation_analysis(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        strong_corr.append({
                            "col1": corr_matrix.columns[i],
                            "col2": corr_matrix.columns[j],
                            "correlation": round(corr_value, 3),
                            "type": "positive" if corr_value > 0 else "negative"
                        })
            return {"strong_correlations": strong_corr}
        return {"strong_correlations": []}

    def _outlier_analysis(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_count = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].shape[0]
            if outliers_count > 0:
                outliers[col] = {
                    "count": outliers_count,
                    "percentage": round(outliers_count / len(self.df) * 100, 2)
                }
        return outliers

    def _generate_insights(self):
        insights = []
        if len(self.analysis["missing_values"]) > 0:
            insights.append(f"📌 Обнаружено {len(self.analysis['missing_values'])} колонок с пропущенными значениями")
        if self.analysis["basic_info"]["duplicates"] > 0:
            insights.append(f"📌 Найдено {self.analysis['basic_info']['duplicates']} дубликатов строк")
        if self.analysis["outliers"]:
            insights.append(f"📌 Обнаружены выбросы в {len(self.analysis['outliers'])} колонках")
        strong_corr = self.analysis["correlations"].get("strong_correlations", [])
        if strong_corr:
            for corr in strong_corr[:3]:
                direction = "положительная" if corr["type"] == "positive" else "отрицательная"
                insights.append(f"📌 Сильная {direction} корреляция между '{corr['col1']}' и '{corr['col2']}' ({corr['correlation']})")
        if not insights:
            insights.append("✅ Данные выглядят чистыми. Явных проблем не обнаружено.")
        return insights

    def clean_data(self):
        df_clean = self.df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                if df_clean[col].isnull().any():
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col].fillna(mode_val[0], inplace=True)
        df_clean = df_clean.drop_duplicates()
        return df_clean

    def suggest_target_column(self):
        best_target = None
        best_score = -1
        best_type = None
        best_reason = ""

        excluded_keywords = ['id', 'идентификатор', 'номер', 'index', 'uuid', 'ключ']

        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in excluded_keywords):
                continue
            if self.df[col].isnull().sum() / len(self.df) > 0.5:
                continue
            unique_count = self.df[col].nunique()
            dtype = self.df[col].dtype
            if unique_count == len(self.df):
                continue
            if dtype in ['object', 'category', 'bool'] or (dtype in ['int64', 'float64'] and unique_count <= 20):
                score = 10 if unique_count == 2 else 8 if unique_count <= 10 else 5
                problem_type = "classification"
                reason = f"категориальная колонка с {unique_count} уникальными значениями"
            elif dtype in ['int64', 'float64'] and unique_count > 20:
                score = 9
                problem_type = "regression"
                reason = f"числовая колонка с {unique_count} уникальными значениями"
            else:
                continue
            name_lower = col.lower()
            if name_lower in ['target', 'label', 'class', 'y']:
                score += 5
            elif name_lower in ['price', 'value', 'amount', 'age', 'salary', 'цена', 'возраст', 'количество']:
                score += 3 if problem_type == "regression" else 1
            if score > best_score:
                best_score = score
                best_target = col
                best_type = problem_type
                best_reason = reason

        if best_target is None:
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                col_lower = col.lower()
                if not any(keyword in col_lower for keyword in excluded_keywords):
                    best_target = col
                    best_type = "regression"
                    best_reason = "числовая колонка (выбрана как запасной вариант)"
                    break

        return best_target, best_type, best_reason

    def display_ml_suggestion(self):
        target, p_type, reason = self.suggest_target_column()
        if target is None:
            return '<div style="background:#fff3cd;padding:12px;border-radius:8px;">⚠️ Не удалось определить целевую колонку</div>'
        icon = "🏷️" if p_type == "classification" else "📈"
        return f'<div style="background:#d1ecf1;padding:12px;border-radius:8px;">{icon} <strong>{p_type.upper()}</strong><br>🎯 Целевая колонка: <code>{target}</code></div>'