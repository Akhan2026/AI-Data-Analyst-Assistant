from datetime import datetime

class ReportGenerator:
    def __init__(self, df, analysis):
        self.df = df
        self.analysis = analysis
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_markdown(self):
        basic_info = self.analysis.get("basic_info", {})
        missing_values = self.analysis.get("missing_values", {})
        insights = self.analysis.get("insights", [])
        
        report = f"""# 📊 Отчёт об анализе данных

**Дата:** {self.timestamp}

## 📌 Основная информация

| Показатель | Значение |
|------------|----------|
| Строки | {basic_info.get('rows', 0):,} |
| Колонки | {basic_info.get('columns', 0)} |
| Дубликаты | {basic_info.get('duplicates', 0):,} |

## 📋 Колонки

| Колонка | Тип | Пропуски |
|---------|-----|----------|
"""
        for col in self.df.columns:
            missing_count = missing_values.get(col, {}).get('count', 0)
            report += f"| {col} | {self.df[col].dtype} | {missing_count} |\n"
        
        report += f"\n## 💡 Инсайты\n"
        for insight in insights:
            report += f"- {insight}\n"
        
        return report
    
    def generate_html(self):
        md = self.generate_markdown()
        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Отчёт</title></head>
<body style="font-family: Arial; margin: 40px; background:#f5f5f5;">
<div style="max-width:1000px;margin:0 auto;background:white;padding:30px;border-radius:10px;">
<pre style="white-space:pre-wrap;font-family:Arial;">{md}</pre>
</div>
</body></html>"""