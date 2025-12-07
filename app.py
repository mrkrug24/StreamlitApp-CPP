import re
import io
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Предсказание цен на автомобили",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
        padding: 1rem;
        background-color: #e8f5e9;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #27ae60;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def extract_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    numbers = re.findall(r'\d+\.?\d*', str(value))
    return float(numbers[0]) if numbers else np.nan

def extract_torque(torque_str):
    if pd.isna(torque_str):
        return np.nan
    try:
        match = re.search(r'(\d+\.?\d*)', str(torque_str))
        if match:
            value = float(match.group(1))
            if 'kg' in str(torque_str).lower():
                value = value * 9.80665
            return value
        return np.nan
    except:
        return np.nan

class FeatureExtractor:
    def __init__(self):
        self.numeric_medians = {}
        self.categorical_modes = {}
        self.torque_pattern = re.compile(r'(\d+\.?\d*)\s*(Nm|kgm)', re.IGNORECASE)
        
    def extract_number(self, value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return value
        numbers = re.findall(r'\d+\.?\d*', str(value))
        return float(numbers[0]) if numbers else np.nan
    
    def extract_torque(self, torque_str):
        if pd.isna(torque_str):
            return np.nan
        try:
            match = self.torque_pattern.search(str(torque_str))
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                if 'kg' in unit:
                    value = value * 9.80665  # Преобразуем в Nm
                return value
            return np.nan
        except:
            return np.nan
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        
        for col in ['mileage', 'engine', 'max_power']:
            X_copy[col] = X_copy[col].apply(self.extract_number)
        
        if 'torque' in X_copy.columns:
            X_copy['torque'] = X_copy['torque'].apply(self.extract_torque)
        
        X_copy['seats'] = X_copy['seats'].astype(str)
        
        numeric_cols = ['mileage', 'engine', 'max_power', 'torque', 'year', 'km_driven']
        for col in numeric_cols:
            if col in X_copy.columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
                self.numeric_medians[col] = X_copy[col].median()
        
        categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
        for col in categorical_cols:
            if col in X_copy.columns:
                mode = X_copy[col].mode()
                self.categorical_modes[col] = mode.iloc[0] if not mode.empty else X_copy[col].iloc[0]
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in ['mileage', 'engine', 'max_power']:
            X_copy[col] = X_copy[col].apply(self.extract_number)
        
        if 'torque' in X_copy.columns:
            X_copy['torque'] = X_copy['torque'].apply(self.extract_torque)
        
        for col, median_value in self.numeric_medians.items():
            if col in X_copy.columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
                X_copy[col] = X_copy[col].fillna(median_value)
        
        for col, mode_value in self.categorical_modes.items():
            if col in X_copy.columns:
                if col == 'seats':
                    X_copy[col] = X_copy[col].astype(str)
                X_copy[col] = X_copy[col].fillna(mode_value)
        
        return X_copy

def safe_load_pickle(filepath):
    try:
        with open(filepath, 'rb') as f:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name == 'FeatureExtractor':
                        return FeatureExtractor
                    return super().find_class(module, name)
            
            return CustomUnpickler(f).load()
    except Exception as e:
        st.error(f"Ошибка при загрузке файла {filepath}: {str(e)}")
        return None

@st.cache_resource
def load_model():
    try:
        return safe_load_pickle('pipe.pkl')
    except FileNotFoundError:
        st.error("❌ Файл модели не найден. Сначала запустите save_model.py")
        return None

@st.cache_resource
def load_model_info():
    try:
        with open('pipe_info.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_train_data():
    df = pd.read_csv(io.StringIO(requests.get('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv', verify=False).text))
    df['brand'] = df['name'].str.split().str[0]
    df['age'] = 2025 - df['year']
    return df

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

pipeline = load_model()
model_info = load_model_info()
df_train = load_train_data()

with st.sidebar:
    st.title("🔧 Навигация")
    selected_tab = st.radio(
        "Выберите раздел:",
        ["🏠 Обзор", "📊 EDA Анализ", "📁 Загрузить CSV", "⌨ Ручной ввод", "⚖ Модель и Веса", "📈 Прогнозы"]
    )
    
    st.markdown("---")
    st.subheader("ℹ️ О модели")
    if model_info:
        st.metric("Метрика R²", f"{model_info['model_metrics']['test_r2']:.4f}")
        st.metric("Алгоритм", model_info['model_metrics']['model_type'])
        st.metric("Параметр alpha", model_info['model_metrics']['alpha'])
    
    st.markdown("---")

if selected_tab == "🏠 Обзор":
    st.markdown('<h1 class="main-header">🚗 Прогнозирование цен на автомобили</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown("""
        ### 📊 О проекте
        
        Это интерактивное веб-приложение для прогнозирования цен на подержанные автомобили 
        с использованием машинного обучения.
        
        **Основные возможности:**
        - 📈 Анализ данных (EDA)
        - 🤖 Прогнозирование цен
        - 🔍 Интерпретация модели
        - 📊 Визуализация результатов
        """)
    
    with col3:
        if model_info:
            metrics = model_info['model_metrics']
            st.metric("Точность на тренировочных данных", f"{metrics['train_r2']:.4f}")
            st.metric("Точность на тестовых данных", f"{metrics['test_r2']:.4f}")
            st.metric("Количество признаков", len(model_info['all_features']))
    
    st.markdown("---")
    
    st.subheader("📈 Быстрая статистика данных")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Всего автомобилей", f"{len(df_train):,}")
    with col2:
        st.metric("Средняя цена", f"${df_train['selling_price'].mean():,.0f}")
    with col3:
        st.metric("Медианная цена", f"${df_train['selling_price'].median():,.0f}")
    
    st.subheader("📋 Пример данных")
    st.dataframe(df_train.head(10), use_container_width=True)
    
    st.subheader("💡 Быстрые инсайты")
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        **🔝 Топ-5 брендов по количеству:**
        """)
        top_brands = df_train['brand'].value_counts().head(5)
        for brand, count in top_brands.items():
            st.write(f"- {brand}: {count} авто")
    
    with insights_col2:
        st.markdown("""
        **🏆 Топ-5 брендов по средней цене:**
        """)
        top_priced = df_train.groupby('brand')['selling_price'].mean().sort_values(ascending=False).head(5)
        for brand, price in top_priced.items():
            st.write(f"- {brand}: ${price:,.0f}")

elif selected_tab == "📊 EDA Анализ":
    st.markdown('<h1 class="sub-header">📊 Анализ данных (EDA)</h1>', unsafe_allow_html=True)
    
    eda_section = st.selectbox(
        "Выберите раздел анализа:",
        ["📈 Распределения", "🔗 Корреляции", "🚗 По брендам", "⛽ По типам", "📅 По годам", "📊 Общая статистика"]
    )
    
    if eda_section == "📈 Распределения":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_train, x='selling_price', nbins=50,
                             title='Распределение цен на автомобили',
                             labels={'selling_price': 'Цена ($)'},
                             color_discrete_sequence=['#1f77b4'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df_train, x=np.log(df_train['selling_price']), nbins=50,
                             title='Распределение логарифма цен',
                             labels={'x': 'log(Цена)'},
                             color_discrete_sequence=['#2ca02c'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Распределение числовых признаков")
        numeric_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
        selected_numeric = st.selectbox("Выберите признак:", numeric_cols)
        
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=[f'Распределение {selected_numeric}', f'{selected_numeric} vs Цена'])
        
        fig.add_trace(
            go.Histogram(x=df_train[selected_numeric], nbinsx=30, name='Распределение',
                        marker_color='#1f77b4'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_train[selected_numeric], y=df_train['selling_price'],
                      mode='markers', marker=dict(size=5, opacity=0.5, color='#ff7f0e'),
                      name='Зависимость от цены'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    elif eda_section == "🔗 Корреляции":
        numeric_df = df_train[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'selling_price']].copy()
        for col in ['mileage', 'engine', 'max_power']:
            numeric_df[col] = numeric_df[col].apply(extract_number)
        
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix,
                       title='Матрица корреляций',
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1,
                       text_auto='.2f')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔝 Наибольшие корреляции с ценой")
        price_corr = corr_matrix['selling_price'].sort_values(ascending=False)[1:6]
        fig = px.bar(x=price_corr.values, y=price_corr.index,
                    orientation='h',
                    title='Топ-5 признаков по корреляции с ценой',
                    labels={'x': 'Коэффициент корреляции', 'y': 'Признак'},
                    color=price_corr.values,
                    color_continuous_scale='viridis')
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    elif eda_section == "🚗 По брендам":
        top_brands = df_train['brand'].value_counts().head(10).index
        
        col1, col2 = st.columns(2)
        
        with col1:
            brand_counts = df_train['brand'].value_counts().head(10)
            fig = px.bar(x=brand_counts.values, y=brand_counts.index,
                        orientation='h',
                        title='Топ-10 брендов по количеству',
                        labels={'x': 'Количество', 'y': 'Бренд'},
                        color=brand_counts.values,
                        color_continuous_scale='blues')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            brand_prices = df_train.groupby('brand')['selling_price'].mean().loc[top_brands]
            fig = px.bar(x=brand_prices.values, y=brand_prices.index,
                        orientation='h',
                        title='Средняя цена по брендам (топ-10)',
                        labels={'x': 'Средняя цена ($)', 'y': 'Бренд'},
                        color=brand_prices.values,
                        color_continuous_scale='reds')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        fig = px.box(df_train[df_train['brand'].isin(top_brands)], 
                    x='brand', y='selling_price',
                    title='Распределение цен по брендам',
                    color='brand')
        fig.update_layout(height=500, xaxis_title='Бренд', yaxis_title='Цена ($)',
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    elif eda_section == "⛽ По типам":
        cat_features = ['fuel', 'transmission', 'seller_type', 'owner']
        
        for feature in cat_features:
            st.subheader(f"📊 Анализ по признаку: {feature}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                value_counts = df_train[feature].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f'Распределение по {feature}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                avg_prices = df_train.groupby(feature)['selling_price'].mean().sort_values()
                fig = px.bar(x=avg_prices.values, y=avg_prices.index,
                           orientation='h',
                           title=f'Средняя цена по {feature}',
                           labels={'x': 'Средняя цена ($)', 'y': feature},
                           color=avg_prices.values,
                           color_continuous_scale='greens')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    elif eda_section == "📅 По годам":
        year_stats = df_train.groupby('year').agg({
            'selling_price': ['mean', 'count']
        }).round(2)
        year_stats.columns = ['avg_price', 'count']
        
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=['Средняя цена по годам', 'Количество автомобилей по годам'])
        
        fig.add_trace(
            go.Scatter(x=year_stats.index, y=year_stats['avg_price'],
                      mode='lines+markers', name='Средняя цена',
                      line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=year_stats.index, y=year_stats['count'],
                  name='Количество', marker_color='#ff7f0e'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📈 Зависимость цены от возраста автомобиля")
        fig = px.scatter(df_train, x='age', y='selling_price',
                        trendline="lowess",
                        title='Цена vs Возраст автомобиля',
                        labels={'age': 'Возраст (лет)', 'selling_price': 'Цена ($)'},
                        opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    elif eda_section == "📊 Общая статистика":
        st.subheader("📋 Описательная статистика")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Числовые признаки:**")
            numeric_stats = df_train[['year', 'km_driven', 'selling_price']].describe()
            st.dataframe(numeric_stats, use_container_width=True)
        
        with col2:
            st.write("**Категориальные признаки:**")
            for col in ['fuel', 'transmission', 'owner']:
                counts = df_train[col].value_counts()
                st.write(f"**{col}:**")
                for val, count in counts.head(3).items():
                    st.write(f"  - {val}: {count} ({count/len(df_train)*100:.1f}%)")
        
        st.subheader("🔍 Пропущенные значения")
        missing = df_train.isnull().sum()
        missing_pct = (missing / len(df_train) * 100).round(2)
        missing_df = pd.DataFrame({
            'Колонка': missing.index,
            'Пропущено': missing.values,
            'Процент': missing_pct.values
        })
        missing_df = missing_df[missing_df['Пропущено'] > 0]
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Колонка', y='Процент',
                        title='Процент пропущенных значений по колонкам',
                        color='Процент',
                        color_continuous_scale='reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ В данных нет пропущенных значений!")

elif selected_tab == "📁 Загрузить CSV":
    st.markdown('<h1 class="sub-header">📁 Прогнозирование по CSV файлу</h1>', unsafe_allow_html=True)
    
    st.info("""
    **Инструкция:**
    1. Загрузите CSV файл с данными об автомобилях
    2. Файл должен содержать те же колонки, что и тренировочные данные
    3. Нажмите кнопку "Прогнозировать" для получения результатов
    """)
    
    uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"✅ Файл успешно загружен! Записей: {len(df_input)}")
            
            with st.expander("👁️ Просмотр данных"):
                st.dataframe(df_input.head(), use_container_width=True)
                st.write(f"**Размер данных:** {df_input.shape[0]} строк, {df_input.shape[1]} колонок")
            
            required_cols = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 
                           'transmission', 'owner', 'mileage', 'engine', 
                           'max_power', 'torque', 'seats']
            
            missing_cols = [col for col in required_cols if col not in df_input.columns]
            
            if missing_cols:
                st.error(f"❌ Отсутствуют обязательные колонки: {missing_cols}")
                st.info("""
                **Требуемые колонки:**
                - name, year, km_driven, fuel, seller_type
                - transmission, owner, mileage, engine
                - max_power, torque, seats
                """)
            else:
                if st.button("🚀 Прогнозировать цены", type="primary", use_container_width=True):
                    with st.spinner("⏳ Обработка данных и прогнозирование..."):
                        try:
                            predictions_log = pipeline.predict(df_input)
                            predictions = np.exp(predictions_log)
                            
                            df_result = df_input.copy()
                            df_result['predicted_price'] = predictions.round(2)
                            df_result['predicted_price_log'] = predictions_log.round(4)
                            
                            st.success(f"✅ Прогнозирование завершено для {len(df_result)} автомобилей!")
                            
                            st.markdown('<div class="prediction-result">📊 Результаты прогнозирования</div>', unsafe_allow_html=True)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Средняя цена", f"${predictions.mean():,.0f}")
                            with col2:
                                st.metric("Минимальная цена", f"${predictions.min():,.0f}")
                            with col3:
                                st.metric("Максимальная цена", f"${predictions.max():,.0f}")
                            with col4:
                                st.metric("Медианная цена", f"${np.median(predictions):,.0f}")
                            
                            st.subheader("📋 Таблица прогнозов")
                            st.dataframe(df_result[['name', 'year', 'fuel', 'transmission', 
                                                  'km_driven', 'predicted_price']].head(20), 
                                       use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.histogram(df_result, x='predicted_price', nbins=30,
                                                 title='Распределение прогнозируемых цен',
                                                 labels={'predicted_price': 'Прогнозируемая цена ($)'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                if 'fuel' in df_result.columns:
                                    fig = px.box(df_result, x='fuel', y='predicted_price',
                                               title='Прогнозируемая цена по типу топлива',
                                               labels={'predicted_price': 'Цена ($)'})
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("📥 Скачать результаты")
                            csv = df_result.to_csv(index=False)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="💾 Скачать CSV с прогнозами",
                                    data=csv,
                                    file_name="car_price_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                summary_stats = df_result['predicted_price'].describe()
                                summary_csv = summary_stats.to_csv()
                                st.download_button(
                                    label="📊 Скачать статистику",
                                    data=summary_csv,
                                    file_name="prediction_summary.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                        except Exception as e:
                            st.error(f"❌ Ошибка при прогнозировании: {str(e)}")
                            st.info("Проверьте формат данных в CSV файле.")
        
        except Exception as e:
            st.error(f"❌ Ошибка при чтении файла: {str(e)}")
    
    with st.expander("📋 Пример формата CSV файла"):
        example_data = """name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,torque,seats
Maruti Swift VXI,2018,25000,Petrol,Individual,Manual,First Owner,22.0 kmpl,1197 CC,81.80 bhp,113Nm@ 4200rpm,5
Hyundai i20 Asta,2017,35000,Petrol,Dealer,Manual,First Owner,18.5 kmpl,1197 CC,82.85 bhp,113.7Nm@ 4000rpm,5
Honda City VX,2019,15000,Diesel,Individual,Automatic,Second Owner,25.1 kmpl,1498 CC,98.6 bhp,200Nm@ 1750rpm,5"""
        
        st.code(example_data, language='csv')
        st.download_button(
            label="📥 Скачать шаблон CSV",
            data=example_data,
            file_name="car_data_template.csv",
            mime="text/csv",
            use_container_width=True
        )

elif selected_tab == "⌨ Ручной ввод":
    st.markdown('<h1 class="sub-header">⌨ Прогнозирование для одного автомобиля</h1>', unsafe_allow_html=True)
    
    st.info("Заполните информацию об автомобиле для получения прогноза цены")
    
    with st.form("car_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Основная информация")
            name = st.text_input("Название модели", "Maruti Swift VXI", help="Например: Maruti Swift VXI")
            year = st.slider("Год выпуска", 1990, 2025, 2018, help="Год выпуска автомобиля")
            km_driven = st.number_input("Пробег (км)", 0, 1000000, 25000, step=1000, 
                                       help="Общий пробег автомобиля в километрах")
            fuel = st.selectbox("Тип топлива", ["Petrol", "Diesel", "CNG", "LPG", "Electric"],
                              help="Тип используемого топлива")
            seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"],
                                     help="Индивидуальный продавец или дилер")
        
        with col2:
            st.subheader("⚙️ Технические характеристики")
            transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"],
                                      help="Тип трансмиссии")
            owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", 
                                            "Fourth & Above Owner", "Test Drive Car"],
                               help="Количество предыдущих владельцев")
            mileage = st.text_input("Расход топлива", "22.0 kmpl", 
                                  help="Например: 22.0 kmpl или 15.5 km/kg для CNG")
            engine = st.text_input("Объем двигателя", "1197 CC", 
                                 help="Например: 1197 CC или 1498 CC")
            max_power = st.text_input("Мощность", "81.80 bhp", 
                                    help="Например: 81.80 bhp или 98.6 bhp")
            torque = st.text_input("Крутящий момент", "113Nm@ 4200rpm",
                                 help="Например: 113Nm@ 4200rpm или 200Nm@ 1750rpm")
            seats = st.slider("Количество мест", 2, 10, 5, 
                            help="Количество пассажирских мест")
        
        submitted = st.form_submit_button("🎯 Получить прогноз цены", use_container_width=True)
    
    if submitted and pipeline is not None:
        input_data = pd.DataFrame([{
            'name': name,
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'torque': torque,
            'seats': seats
        }])
        
        try:
            with st.spinner("⏳ Расчет прогноза..."):
                prediction_log = pipeline.predict(input_data)[0]
                prediction = np.exp(prediction_log)
                st.markdown(f'<div class="prediction-result">💰 Прогнозируемая цена: ${prediction:,.2f}</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Цена в логарифмической шкале", f"{prediction_log:.4f}")
                
                with col2:
                    avg_price = df_train['selling_price'].mean()
                    diff = prediction - avg_price
                    diff_pct = (diff / avg_price) * 100
                    st.metric("Отклонение от среднего", f"{diff_pct:+.1f}%")
                
                with col3:
                    st.metric("Возраст автомобиля", f"{2025 - year} лет")
                
                with st.expander("📊 Детальный анализ"):
                    st.write("**Введенные данные:**")
                    st.dataframe(input_data, use_container_width=True)
                    
                    st.write("**Похожие автомобили в данных:**")
                    similar_cars = df_train[
                        (df_train['fuel'] == fuel) & 
                        (df_train['transmission'] == transmission) &
                        (abs(df_train['year'] - year) <= 3) &
                        (abs(df_train['km_driven'] - km_driven) <= 20000)
                    ].head(5)
                    
                    if len(similar_cars) > 0:
                        st.dataframe(similar_cars[['name', 'year', 'km_driven', 'fuel', 'selling_price']], 
                                   use_container_width=True)
                    else:
                        st.info("Похожих автомобилей не найдено в тренировочных данных.")
        
        except Exception as e:
            st.error(f"❌ Ошибка при прогнозировании: {str(e)}")
            st.info("Проверьте правильность введенных данных.")

elif selected_tab == "⚖ Модель и Веса":
    st.markdown('<h1 class="sub-header">⚖ Анализ модели и важность признаков</h1>', unsafe_allow_html=True)
    
    if pipeline is not None and model_info is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Тип модели", model_info['model_metrics']['model_type'])
        with col2:
            st.metric("R² на тесте", f"{model_info['model_metrics']['test_r2']:.4f}")
        with col3:
            st.metric("Количество признаков", len(model_info['all_features']))
        
        model = pipeline.named_steps['model']
        
        if len(model_info['all_features']) >= len(model.coef_):
            feature_names = model_info['all_features'][:len(model.coef_)]
        else:
            feature_names = model_info['all_features'] + [f'feature_{i}' for i in range(len(model_info['all_features']), len(model.coef_))]
        
        feature_importance = pd.DataFrame({
            'Признак': feature_names,
            'Коэффициент': model.coef_,
            'Важность': np.abs(model.coef_)
        }).sort_values('Важность', ascending=False)
        
        st.subheader("📊 Коэффициенты модели")
        
        fig = make_subplots(rows=1, cols=2,
                          subplot_titles=['Топ-10 положительных признаков', 'Топ-10 отрицательных признаков'])
        
        top_positive = feature_importance[feature_importance['Коэффициент'] > 0].head(10)
        fig.add_trace(
            go.Bar(x=top_positive['Коэффициент'], y=top_positive['Признак'],
                  orientation='h', name='Положительные',
                  marker_color='#2ca02c'),
            row=1, col=1
        )
        
        top_negative = feature_importance[feature_importance['Коэффициент'] < 0].head(10)
        fig.add_trace(
            go.Bar(x=top_negative['Коэффициент'], y=top_negative['Признак'],
                  orientation='h', name='Отрицательные',
                  marker_color='#d62728'),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🌳 Важность признаков (Treemap)")
        
        feature_importance['Категория'] = feature_importance['Признак'].apply(
            lambda x: 'Числовой' if x in model_info['numeric_features'] else 'Категориальный'
        )
        
        fig = px.treemap(feature_importance.head(30),
                        path=['Категория', 'Признак'],
                        values='Важность',
                        color='Коэффициент',
                        color_continuous_scale='RdBu',
                        title='Распределение важности признаков')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Таблица коэффициентов")
        
        feature_importance['Интерпретация'] = feature_importance.apply(
            lambda row: 'Увеличивает цену' if row['Коэффициент'] > 0 else 'Уменьшает цену',
            axis=1
        )
        
        display_df = feature_importance.copy()
        display_df['Коэффициент'] = display_df['Коэффициент'].apply(lambda x: f'{x:.6f}')
        display_df['Важность'] = display_df['Важность'].apply(lambda x: f'{x:.6f}')
        
        st.dataframe(display_df, use_container_width=True)
        
        with st.expander("📚 Как интерпретировать коэффициенты"):
            st.markdown("""
            ### Интерпретация коэффициентов Ridge Regression:
            
            #### Положительные коэффициенты (увеличивают цену):
            - **year**: Более новые автомобили стоят дороже
            - **engine**: Больший объем двигателя → выше цена
            - **max_power**: Большая мощность → выше цена
            - **transmission_Automatic**: Автоматическая коробка передач дороже
            - **fuel_Diesel**: Дизельные автомобили обычно дороже бензиновых
            
            #### Отрицательные коэффициенты (уменьшают цену):
            - **km_driven**: Больший пробег → ниже цена
            - **owner_Second Owner и др.**: Больше владельцев → ниже цена
            - **seller_type_Individual**: Частные продавцы обычно дешевле дилеров
            
            #### Важные замечания:
            1. Признаки стандартизированы, поэтому коэффициенты сравнимы
            2. Целевая переменная: log(price), поэтому изменения в коэффициентах интерпретируются в процентном отношении к цене
            3. Ridge regression уменьшает коэффициенты менее важных признаков (регуляризация)
            """)
    
    else:
        st.error("❌ Модель не загружена. Сначала запустите save_model.py")

elif selected_tab == "📈 Прогнозы":
    st.markdown('<h1 class="sub-header">📈 Анализ прогнозов модели</h1>', unsafe_allow_html=True)
    
    if pipeline is not None:
        df_test = pd.read_csv(io.StringIO(requests.get('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv',  verify=False).text))
        y_test = np.log(df_test['selling_price'])
        
        y_pred_log = pipeline.predict(df_test.drop('selling_price', axis=1))
        y_pred = np.exp(y_pred_log)
        y_true = np.exp(y_test)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"${mae:,.0f}")
        with col2:
            st.metric("RMSE", f"${rmse:,.0f}")
        with col3:
            st.metric("R² Score", f"{r2:.4f}")
        with col4:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            st.metric("MAPE", f"{mape:.2f}%")
        
        st.subheader("📊 Прогнозы vs Фактические значения")
        
        fig = make_subplots(rows=1, cols=2,
                          subplot_titles=['Сравнение прогнозов и фактических цен', 'Ошибки прогнозирования'])
        
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred,
                      mode='markers',
                      marker=dict(size=8, opacity=0.6, color='#1f77b4'),
                      name='Прогнозы'),
            row=1, col=1
        )
        
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[0, max_val], y=[0, max_val],
                      mode='lines',
                      line=dict(color='red', dash='dash'),
                      name='Идеальный прогноз'),
            row=1, col=1
        )
        
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals,
                      mode='markers',
                      marker=dict(size=8, opacity=0.6, color='#ff7f0e'),
                      name='Ошибки'),
            row=1, col=2
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=True)
        fig.update_xaxes(title_text="Фактическая цена ($)", row=1, col=1)
        fig.update_yaxes(title_text="Прогнозируемая цена ($)", row=1, col=1)
        fig.update_xaxes(title_text="Прогнозируемая цена ($)", row=1, col=2)
        fig.update_yaxes(title_text="Ошибка ($)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📈 Распределение ошибок прогнозирования")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(x=residuals, nbins=50,
                             title='Распределение ошибок',
                             labels={'x': 'Ошибка ($)'},
                             color_discrete_sequence=['#d62728'])
            fig.add_vline(x=0, line_dash="dash", line_color="green")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(y=residuals,
                        title='Boxplot ошибок',
                        labels={'y': 'Ошибка ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🎯 Примеры прогнозов")
        
        df_results = pd.DataFrame({
            'Фактическая цена': y_true,
            'Прогнозируемая цена': y_pred,
            'Ошибка': residuals,
            'Относительная ошибка (%)': (residuals / y_true * 100).abs()
        })
        
        worst_predictions = df_results.nlargest(5, 'Относительная ошибка (%)')
        best_predictions = df_results.nsmallest(5, 'Относительная ошибка (%)')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Худшие прогнозы (наибольшая ошибка):**")
            st.dataframe(worst_predictions, use_container_width=True)
        
        with col2:
            st.write("**Лучшие прогнозы (наименьшая ошибка):**")
            st.dataframe(best_predictions, use_container_width=True)
    
    else:
        st.error("❌ Модель не загружена")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🚗 Car Price Prediction Model • Built with Streamlit • Ridge Regression</p>
    <p>Для корректной работы убедитесь, что файлы модели (pipe.pkl, pipe_info.pkl) созданы c помощью save_pipe.py</p>
</div>
""", unsafe_allow_html=True)