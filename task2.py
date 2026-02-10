import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from sklearn.cluster import KMeans

# Подключение к БД
CONN = 'mysql+pymysql://root:root@localhost:3306/competition'

def load():
    # 2.1 Загрузка данных одной командой
    return pd.read_sql("SELECT * FROM track_points", create_engine(CONN))

def process(df):
    # 2.2 и 2.3 ML и Риски
    if df.empty: return df
    
    # Кластеризация (KMeans) - группируем похожие точки
    # Заполняем пропуски нулями, чтобы ML не падал
    features = df[['lat', 'lon', 'elevation']].fillna(0)
    df['cluster'] = KMeans(n_clusters=5).fit_predict(features)
    
    # Определение рисков (простая логика)
    df['risk'] = 'Safe'
    df.loc[(df['elevation'] < 50) & (df['environment'] == 'swamp'), 'risk'] = 'Flood'
    df.loc[(df['temperature'] > 30) & (df['environment'] == 'forest'), 'risk'] = 'Fire'
    
    return df

# --- ИНТЕРФЕЙС ---
st.title("Аналитическая Система (Модуль Б)")

try:
    df = process(load())
except:
    st.error("Ошибка БД! Запусти сначала main.py")
    st.stop()

# Сайдбар для фильтров (2.1)
env_filter = st.sidebar.multiselect("Местность", df['environment'].unique(), default=df['environment'].unique())
df = df[df['environment'].isin(env_filter)]

# Метрики (2.1)
st.write(f"**Точек:** {len(df)} | **Ср. Температура:** {df['temperature'].mean():.1f}°C | **Опасных зон:** {len(df[df['risk']!='Safe'])}")

# Графики (2.2)
c1, c2 = st.columns(2)
# График 1: Зависимость Температура-Высота
c1.plotly_chart(px.scatter(df, x='elevation', y='temperature', color='environment', title="Температура vs Высота"))
# График 2: Распределение активности
c2.plotly_chart(px.bar(df.groupby('environment')['cadence'].mean().reset_index(), x='environment', y='cadence', title="Шаги по местности"))

# Карта (2.4)
map_type = st.radio("Раскраска карты:", ["Кластеры (ML)", "Риски"])
color_col = 'cluster' if map_type == "Кластеры (ML)" else 'risk'

st.plotly_chart(px.scatter_mapbox(df, lat="lat", lon="lon", color=color_col, zoom=10, height=500).update_layout(mapbox_style="open-street-map"))