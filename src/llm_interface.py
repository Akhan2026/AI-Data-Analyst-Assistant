import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class LLMInterface:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = "llama-3.1-8b-instant"
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        if self.api_key:
            print("✅ Groq API готов")
        else:
            print("⚠️ API ключ не найден. Добавьте GROQ_API_KEY в .env")

    def ask_question(self, question, df, analysis):
        if not self.api_key:
            return "⚠️ Нет API ключа. Добавьте GROQ_API_KEY в файл .env"

        # Специальная обработка популярных вопросов
        q_lower = question.lower()

        # Вопрос про выручку по категориям
        if ("выручк" in q_lower or "категори" in q_lower) and "Цена" in df.columns and "Количество" in df.columns and "Категория_товара" in df.columns:
            df_temp = df.copy()
            df_temp['Выручка'] = df_temp['Цена'] * df_temp['Количество']
            revenue_by_cat = df_temp.groupby('Категория_товара')['Выручка'].sum().sort_values(ascending=False)
            if len(revenue_by_cat) > 0:
                top_cat = revenue_by_cat.index[0]
                top_val = revenue_by_cat.iloc[0]
                total = revenue_by_cat.sum()
                percent = (top_val / total) * 100
                return f"📊 **{top_cat}** приносит больше всего выручки: **{top_val:,.2f}** (это {percent:.1f}% от всей выручки)."

        # Вопрос про корреляцию
        if ("корреляц" in q_lower or "связь" in q_lower) and "Возраст" in df.columns and "Цена" in df.columns:
            corr = df['Возраст'].corr(df['Цена'])
            if abs(corr) < 0.2:
                return f"🔗 Корреляция между возрастом и ценой: **{corr:.3f}** — связи практически нет."
            elif corr > 0:
                return f"🔗 Корреляция между возрастом и ценой: **{corr:.3f}** — умеренная положительная связь."
            else:
                return f"🔗 Корреляция между возрастом и ценой: **{corr:.3f}** — умеренная отрицательная связь."

        # Вопрос про возраст и цену
        if "возраст" in q_lower and "цен" in q_lower and "Возраст" in df.columns and "Цена" in df.columns:
            bins = [0, 25, 35, 45, 100]
            labels = ['до 25', '25-35', '35-45', '45+']
            df['Возраст_группа'] = pd.cut(df['Возраст'], bins=bins, labels=labels)
            avg_by_age = df.groupby('Возраст_группа')['Цена'].mean()
            max_age_group = avg_by_age.idxmax()
            max_value = avg_by_age.max()
            return f"📊 Больше всего тратят покупатели в возрасте **{max_age_group}** лет (средняя цена: **{max_value:.2f}**)."

        # Стандартный запрос к Groq
        sample = df.head(15).to_string()
        prompt = f"""Ты AI Data Analyst Assistant. Отвечай на вопросы ТОЛЬКО на основе предоставленных данных.

📊 ДАННЫЕ:
- Строк: {len(df):,}
- Колонки: {', '.join(df.columns)}

📋 ПРИМЕР ДАННЫХ (первые 15 строк):
{sample}

❓ ВОПРОС: {question}

⚠️ ВАЖНО:
- НЕ пиши код
- НЕ предлагай, как посчитать
- Дай КОНКРЕТНЫЙ ответ на основе данных
- Если данных не хватает — так и скажи
- Ответ должен быть кратким (1-3 предложения)

КОНКРЕТНЫЙ ОТВЕТ:"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }

        try:
            response = requests.post(self.url, headers=headers, json=data)
            result = response.json()

            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = result.get("error", {}).get("message", "Неизвестная ошибка")
                return f"⚠️ Ошибка API: {error_msg}"
        except Exception as e:
            return f"⚠️ Ошибка соединения: {e}"

    def get_model_info(self):
        if self.api_key:
            return f"Groq: {self.model}"
        return "Groq API не подключён"