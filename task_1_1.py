
import os
import requests
import gpxpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import json
import random
from sqlalchemy import create_engine, text

# --- НАСТРОЙКИ ---
DB_CONN = 'mysql+pymysql://root:root@localhost:3306/competition'

# !!! СЮДА ВСТАВЛЯЙ СВОИ ССЫЛКИ !!!
# Формат: ("Имя_файла.gpx", "Ссылка на файл")
TARGET_FILES = [
    ("track_1.gpx", "https://raw.githubusercontent.com/stevenvandorpe/testdata/master/gps/gpx/1.gpx"), 
    ("track_2.gpx", "https://raw.githubusercontent.com/gps-touring/sample-gpx/master/Hamburg/Alster.gpx"),
    ("track_3.gpx", "https://raw.githubusercontent.com/jmontane/gpx-parser/master/src/test/resources/ashland.gpx"),
    ("track_4.gpx", "https://raw.githubusercontent.com/gps-touring/sample-gpx/master/Wales/Snowdon.gpx"),
    ("track_5.gpx", "https://raw.githubusercontent.com/stevenvandorpe/testdata/master/gps/gpx/2.gpx")
]

class DownloadAgent:
    def __init__(self):
        self.engine = create_engine(DB_CONN)
        if not os.path.exists('tracks'):
            os.makedirs('tracks')

    def run_cycle(self):
        """Запускает полный цикл: Скачивание -> Обработка -> БД -> Карта"""
        for filename, url in TARGET_FILES:
            print(f"\n--- Работа с файлом: {filename} ---")
            
            # 1. Скачивание
            local_path = self.download_file(filename, url)
            if not local_path: continue # Если ошибка, пропускаем
            
            # 2. Парсинг
            df = self.parse_gpx(local_path)
            if df.empty: 
                print("Файл пустой или битый, пропускаем.")
                continue

            # 3. Сохранение в БД
            self.save_to_db(df, filename)

            # 4. Рисование карты
            self.create_map(df, filename)

    def download_file(self, filename, url):
        """Скачивает файл по ссылке"""
        path = f"tracks/{filename}"
        print(f"--> Скачиваю с: {url}")
        try:
            # Fake User-Agent чтобы сайты не блокировали скрипт
            headers = {'User-Agent': 'Mozilla/5.0'} 
            r = requests.get(url, headers=headers, timeout=10)
            
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                print(f"--> Файл сохранен: {path}")
                return path
            else:
                print(f"ОШИБКА: Сайт вернул код {r.status_code}")
                return None
        except Exception as e:
            print(f"ОШИБКА скачивания: {e}")
            return None

    def parse_gpx(self, filepath):
        """Чтение GPX и расчет физики"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
        except Exception as e:
            print(f"Ошибка чтения GPX: {e}")
            return pd.DataFrame()

        data = []
        for t in gpx.tracks:
            for s in t.segments:
                for p in s.points:
                    ele = p.elevation if p.elevation else 0
                    data.append({
                        'time': p.time,
                        'date': p.time if p.time else pd.Timestamp.now(),
                        'lat': p.latitude,
                        'lon': p.longitude,
                        'ele': ele,
                        'temp': round(22 - (ele/100)*0.6, 1) # Имитация температуры
                    })
        
        df = pd.DataFrame(data)
        if df.empty: return df

        # Расчет скорости и шагов (упрощенно)
        # Если время None, заполняем секундами по порядку
        if df['time'].isnull().all():
            df['time_diff'] = 1 # 1 секунда между точками
        else:
            df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(1)

        df['dist'] = np.sqrt((df['lat'].diff()*111000)**2 + (df['lon'].diff()*111000)**2).fillna(0)
        df['speed'] = (df['dist'] / df['time_diff']).fillna(0)
        
        # Шаги (Cadence): если быстро - бежим, если медленно - стоим
        df['cadence'] = (df['speed'] * 30 + 60).fillna(0).astype(int)
        df.loc[df['speed'] < 0.5, 'cadence'] = 0

        # Упрощенное окружение (1.2)
        df['environment'] = df['ele'].apply(lambda x: 'Mountains' if x > 500 else 'Forest')
        df['poi'] = [json.dumps(['Shop'] if random.random() < 0.05 else []) for _ in range(len(df))]
        
        return df

    def save_to_db(self, df, filename):
        """Запись в SQL"""
        with self.engine.connect() as conn:
            conn.execute(text("INSERT INTO tracks (filename) VALUES (:f)"), {"f": filename})
            conn.commit()
            tid = conn.execute(text("SELECT MAX(id) FROM tracks")).scalar()

        df['track_id'] = tid
        # Переименование для БД
        save_df = df.rename(columns={'ele': 'elevation', 'temp': 'temperature'})
        # Заполняем пропуски в дате
        save_df['date'] = save_df

DROP DATABASE IF EXISTS competition;
CREATE DATABASE competition;
USE competition;

CREATE TABLE tracks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255),
    region VARCHAR(100) DEFAULT 'Russia',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE track_points (
    id INT AUTO_INCREMENT PRIMARY KEY,
    track_id INT,
    date DATETIME,
    lat DOUBLE,
    lon DOUBLE,
    elevation DOUBLE,
    speed DOUBLE,
    cadence INT,
    temperature DOUBLE,
    environment VARCHAR(50),
    poi TEXT,
    FOREIGN KEY (track_id) REFERENCES tracks(id)
);
