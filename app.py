import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from src.analyzer import DataAnalyzer
from src.visualizer import Visualizer
from src.llm_interface import LLMInterface
from src.report_generator import ReportGenerator
from src.ml_summary import MLSummary
from src.feature_engineering import FeatureEngineering

st.set_page_config(page_title="AI Data Analyst Assistant", page_icon="🤖", layout="wide")

def main():
    st.title("🤖 AI Data Analyst Assistant")
    st.markdown("**Загрузите CSV / Excel — я проанализирую, отвечу на вопросы и помогу с ML**")
    
    uploaded_file = st.file_uploader("📁 Загрузите файл", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"✅ Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
        except Exception as e:
            st.error(f"Ошибка: {e}")
            return
        
        analyzer = DataAnalyzer(df)
        visualizer = Visualizer()
        llm = LLMInterface()
        
        with st.spinner("🔄 Анализирую данные..."):
            analysis = analyzer.run_full_analysis()
        
        df_clean = analyzer.clean_data()
        
        tabs = st.tabs(["📊 Обзор", "📈 Визуализации", "🤖 AI Вопросы", "🛠️ Feature Engineering", "🤖 ML Задача", "📋 Отчёт"])
        
        # TAB 0: Обзор
        with tabs[0]:
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("📊 Строки", f"{analysis['basic_info']['rows']:,}")
            with col2: st.metric("📋 Колонки", analysis['basic_info']['columns'])
            with col3: st.metric("⚠️ Пропуски", len(analysis['missing_values']))
            with col4: st.metric("🔄 Дубликаты", analysis['basic_info']['duplicates'])
            
            st.divider()
            st.subheader("🎯 ML задача (автоопределение)")
            st.markdown(analyzer.display_ml_suggestion(), unsafe_allow_html=True)
            
            if analysis['insights']:
                st.subheader("💡 Ключевые инсайты")
                for insight in analysis['insights']:
                    st.markdown(f"- {insight}")
            
            if st.checkbox("📄 Показать пример данных"):
                st.dataframe(df.head(20), use_container_width=True)
        
        # TAB 1: Визуализации
        with tabs[1]:
            st.subheader("📈 Графики")
            plots = visualizer.create_all_plots(df)
            if plots:
                cols = st.columns(2)
                for i, (name, fig) in enumerate(plots.items()):
                    with cols[i % 2]:
                        st.pyplot(fig)
            else:
                st.warning("⚠️ Недостаточно данных для построения графиков")
        
        # TAB 2: AI вопросы
        with tabs[2]:
            st.subheader("🤖 Задайте вопрос о данных")
            st.caption(f"🧠 Используемая AI модель: **{llm.get_model_info()}** (работает локально, без API)")
            question = st.text_input("💬 Ваш вопрос:", placeholder="Пример: сколько строк в датасете? какие есть колонки?")
            if st.button("🔍 Отправить вопрос"):
                if question:
                    with st.spinner("🤔 Анализирую..."):
                        answer = llm.ask_question(question, df, analysis)
                    st.markdown(f"**🤖 Ответ:** {answer}")
                else:
                    st.warning("Введите вопрос")
        
       # TAB 3: Feature Engineering
        with tabs[3]:
            st.subheader("🛠️ Feature Engineering (создание новых признаков)")

            # Инициализируем состояние
            if "df_clean" not in st.session_state:
                st.session_state.df_clean = df_clean.copy()
                st.session_state.fe_created = []
                st.session_state.fe_history = []

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📅 Извлечение из дат**")
                if st.button("📆 Извлечь из дат"):
                    fe = FeatureEngineering(st.session_state.df_clean)
                    st.session_state.df_clean = fe.extract_datetime_features()
                    st.session_state.fe_created.extend(fe.created_features)
                    st.session_state.fe_history.append({"action": "Извлечение из дат", "features": fe.created_features.copy()})
                    st.success(f"✅ Добавлено {len(fe.created_features)} признаков")

            with col2:
                st.markdown("**🔄 Комбинации признаков**")
                if st.button("🧮 Комбинации (×)"):
                    fe = FeatureEngineering(st.session_state.df_clean)
                    st.session_state.df_clean = fe.create_interaction_features()
                    st.session_state.fe_created.extend(fe.created_features)
                    st.session_state.fe_history.append({"action": "Комбинации", "features": fe.created_features.copy()})
                    st.success(f"✅ Добавлено {len(fe.created_features)} комбинаций")

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**🔤 Кодирование категорий**")
                if st.button("🏷️ One‑Hot Encoding"):
                    fe = FeatureEngineering(st.session_state.df_clean)
                    st.session_state.df_clean = fe.encode_categorical()
                    st.session_state.fe_created.extend(fe.created_features)
                    st.session_state.fe_history.append({"action": "One‑Hot Encoding", "features": fe.created_features.copy()})
                    st.success(f"✅ Закодировано {len(fe.created_features)} новых колонок")

            with col4:
                st.markdown("**⚡ Действия**")
                if st.button("🔄 Сбросить всё"):
                    st.session_state.df_clean = df_clean.copy()
                    st.session_state.fe_created = []
                    st.session_state.fe_history = []
                    st.success("✅ Все изменения отменены, данные сброшены к исходным")
                    st.rerun()

            st.divider()

            # Показать результат
            if st.button("📊 Показать результат"):
                st.info(f"📌 Создано новых признаков: {len(st.session_state.fe_created)}")
                st.dataframe(st.session_state.df_clean.head(10), use_container_width=True)

                if st.session_state.fe_created:
                    with st.expander("📋 Список всех созданных признаков"):
                        st.write(st.session_state.fe_created)

            # История изменений
            if st.session_state.fe_history:
                with st.expander("📜 История изменений"):
                    for i, h in enumerate(st.session_state.fe_history):
                        st.caption(f"{i+1}. {h['action']}: добавлено {len(h['features'])} признаков")

            st.divider()
            st.caption("💡 **Советы:**")
            st.caption("- Извлечение из дат работает только для колонок с датами")
            st.caption("- Комбинации создают произведения числовых признаков")
            st.caption("- One‑Hot Encoding превращает текст в числа")
        
        
        # TAB 4: ML Задача
        with tabs[4]:
            st.subheader("🤖 Машинное обучение")

            target_col, problem_type, __ = analyzer.suggest_target_column()

            if target_col is None:
                st.warning("⚠️ Не удалось автоматически определить целевую колонку")
            else:
                st.info(f"🎯 **Рекомендуемая целевая колонка:** `{target_col}` (тип: `{problem_type}`)")

                all_cols = df_clean.columns.tolist()
                default_idx = all_cols.index(target_col) if target_col in all_cols else 0
                selected_target = st.selectbox("Выберите целевую колонку", all_cols, index=default_idx)

                # Определяем тип задачи для выбранной колонки
                unique_count = df_clean[selected_target].nunique()
                dtype = df_clean[selected_target].dtype

                if dtype in ['object', 'category', 'bool'] or (dtype in ['int64', 'float64'] and unique_count <= 20):
                    default_type = "classification"
                elif dtype in ['int64', 'float64'] and unique_count > 20:
                    default_type = "regression"
                else:
                    default_type = "classification"

                selected_type = st.selectbox("Выберите тип задачи", ["classification", "regression"], index=0 if default_type == "classification" else 1)

                # Кнопка обучения
                if st.button("🚀 Обучить модель и показать метрики"):
                    with st.spinner("🔄 Обучение модели..."):
                        ml = MLSummary(df_clean, selected_target, selected_type)
                        metrics = ml.train_and_evaluate()

                        if metrics and "error" not in metrics:
                            st.success("✅ Модель успешно обучена!")

                            if selected_type == "classification":
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("📈 Accuracy", f"{metrics['accuracy']:.4f}")
                                    st.metric("🎯 F1-score", f"{metrics['f1_score']:.4f}")
                                with col2:
                                    st.metric("🎯 Precision", f"{metrics['precision']:.4f}")
                                    st.metric("📊 Recall", f"{metrics['recall']:.4f}")
                            else:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("📈 R²", f"{metrics['r2']:.4f}")
                                    st.metric("📉 RMSE", f"{metrics['rmse']:.4f}")
                                with col2:
                                    st.metric("📊 MAE", f"{metrics['mae']:.4f}")
                                    st.metric("📉 MSE", f"{metrics['mse']:.4f}")

                            # Кнопка скачивания метрик
                            metrics_json = ml.get_metrics_download_link()
                            if metrics_json:
                                st.download_button(
                                    label="📥 Скачать метрики (JSON)",
                                    data=metrics_json,
                                    file_name=f"metrics_{selected_target}_{selected_type}.json",
                                    mime="application/json"
                                )

                            # Сохраняем модель в session_state для предсказаний
                            st.session_state.ml_model = ml
                            st.session_state.selected_target = selected_target
                        else:
                            error_msg = metrics.get("error", "Неизвестная ошибка") if metrics else "Ошибка"
                            st.error(f"❌ Ошибка: {error_msg}")

                # ─────────────────────────────────────────────────────────────
                # ПРЕДСКАЗАНИЯ (дополнительно)
                # ─────────────────────────────────────────────────────────────
                if "ml_model" in st.session_state and st.session_state.ml_model:
                    with st.expander("📈 Дополнительно: сохранить предсказания"):
                        st.markdown("Добавить колонку с предсказаниями модели в скачиваемый файл.")
                        if st.button("🔮 Добавить предсказания в данные", key="add_predictions"):
                            ml = st.session_state.ml_model
                            target = st.session_state.selected_target

                            # Подготовка данных для предсказаний
                            X_all = df_clean.drop(columns=[target])
                            X_all = pd.get_dummies(X_all, drop_first=True)

                            # Выравниваем колонки с моделью
                            if hasattr(ml.model, 'feature_names_in_'):
                                missing_cols = set(ml.model.feature_names_in_) - set(X_all.columns)
                                for col in missing_cols:
                                    X_all[col] = 0
                                X_all = X_all[ml.model.feature_names_in_]

                                predictions = ml.model.predict(X_all)

                                # Добавляем предсказания в df_clean
                                if "prediction" in df_clean.columns:
                                    df_clean.drop(columns=["prediction"], inplace=True)
                                df_clean['prediction'] = predictions

                                st.success("✅ Предсказания добавлены в датафрейм!")
                                st.info("📥 Теперь нажмите «Скачать обработанные данные» внизу страницы.")
                            else:
                                st.warning("⚠️ Не удалось определить колонки модели.")

                # ─────────────────────────────────────────────────────────────
                # Экспорт данных
                # ─────────────────────────────────────────────────────────────
                st.divider()
                st.subheader("📥 Экспорт данных")

                csv_data = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Скачать обработанные данные (CSV)",
                    data=csv_data,
                    file_name="processed_data.csv",
                    mime="text/csv",
                    help="Экспорт текущего датафрейма после всех преобразований (FE, очистка, предсказания)"
                )
        
        # TAB 5: Отчёт
        with tabs[5]:
            st.subheader("📋 Генерация отчёта")
            report_gen = ReportGenerator(df_clean, analysis)
            if st.button("📄 Создать Markdown отчёт"):
                md = report_gen.generate_markdown()
                st.download_button("📥 Скачать Markdown", md, "report.md", "text/markdown")
                st.markdown(md)
            if st.button("🌐 Создать HTML отчёт"):
                html = report_gen.generate_html()
                st.download_button("📥 Скачать HTML", html, "report.html", "text/html")
        
        # Кнопка скачивания данных
        st.divider()
        csv_data = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Скачать обработанные данные (CSV)", csv_data, "processed_data.csv", "text/csv")

if __name__ == "__main__":
    main()