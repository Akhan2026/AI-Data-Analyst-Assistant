import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

class Visualizer:
    def __init__(self):
        pass

    def create_all_plots(self, df):
        plots = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in numeric_cols[:4]:
            plots[f"dist_{col}"] = self._plot_distribution(df, col)

        for col in categorical_cols[:3]:
            if df[col].nunique() <= 15:
                plots[f"count_{col}"] = self._plot_counts(df, col)

        if len(numeric_cols) >= 2:
            plots["correlation"] = self._plot_correlation_matrix(df, numeric_cols)

        return plots

    def _plot_distribution(self, df, column):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Гистограмма — через matplotlib, а не через df[col].hist
        axes[0].hist(df[column].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].set_title(f'Distribution of {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')

        # Boxplot
        sns.boxplot(y=df[column], ax=axes[1], color='skyblue')
        axes[1].set_title(f'Boxplot of {column}')

        plt.tight_layout()
        return fig

    def _plot_counts(self, df, column):
        fig, ax = plt.subplots(figsize=(10, 5))

        counts = df[column].value_counts().head(15)
        ax.bar(range(len(counts)), counts.values, color='skyblue', edgecolor='black')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        ax.set_title(f'Top values in {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')

        plt.tight_layout()
        return fig

    def _plot_correlation_matrix(self, df, numeric_cols):
        fig, ax = plt.subplots(figsize=(10, 8))

        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')

        plt.tight_layout()
        return fig