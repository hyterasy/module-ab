import os
import json
import random
import cv2
import gpxpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from sqlalchemy import create_engine, text
from scipy import stats

# Настройки БД (Пользователь:Пароль@Хост:Порт/База)
DB_CONN = 'mysql+pymysql://root:root@localhost:3306/competition'

class CollectorAgent:
    def __init__(self):
        self.engine = create_engine(DB_CONN)
        print("[Collector] Агент готов к работе.")

    def process_gpx(self, filename):
        """Чтение GPX и расчет физических параметров"""
        print(f"--> Читаю файл: {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            gpx = gpxpy.parse(f)

        data = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    ele = point.elevation if point.elevation is not None else 0
                    
                    # Физика: Температура падает на 0.6 градуса каждые 100м
                    temp = round(25 - (ele / 100) * 0.6, 1)
                    
                    data.append({
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'ele': ele,
                        'time': point.time,
                        'date': point.time.date(),
                        'temp': temp
                    })

        df = pd.DataFrame(data)
        
        # 1. Считаем дистанцию и скорость (векторно, через Pandas)
        df['dist'] = np.sqrt((df['lat'].diff()*111000)**2 + (df['lon'].diff()*111000*np.cos(np.radians(df['lat'])))**2)
        df['speed'] = (df['dist'] / df['time'].diff().dt.total_seconds()).fillna(0)
        
        # 2. Считаем частоту шагов (Cadence)
        df['cadence'] = (df['speed'] * 40 + 80).astype(int)
        df.loc[df['speed'] < 0.5, 'cadence'] = 0 # Если стоим, шагов нет

        # 3. Определяем местность и POI (через обычные функции)
        df['environment'] = df['ele'].apply(self._get_environment_type)
        df['poi'] = df.apply(self._get_poi_json, axis=1)
        
        return df

    def _get_environment_type(self, elevation):
        """Определяет тип местности по высоте (вместо lambda)"""
        if elevation > 1000: return 'mountains'
        elif elevation > 300: return 'forest'
        elif elevation < 50: return 'swamp'
        return 'plain'

    def _get_poi_json(self, row):
        """Генерирует список объектов JSON (вместо lambda)"""
        # С вероятностью 10% находим пещеру
        if random.random() < 0.1:
            return json.dumps(['cave'])
        return json.dumps([])

    def save_to_db(self, df, filename):
        """Сохранение в MariaDB"""
        with self.engine.connect() as conn:
            # 1. Создаем запись о треке
            conn.execute(text("INSERT INTO tracks (filename, region) VALUES (:f, 'Region_1')"), {"f": filename})
            conn.commit()
            # 2. Получаем его ID
            track_id = conn.execute(text("SELECT MAX(id) FROM tracks")).scalar()

        df['track_id'] = track_id
        
        # Готовим имена колонок как в таблице БД
        df = df.rename(columns={'ele': 'elevation', 'temp': 'temperature'})
        cols = ['track_id', 'date', 'lat', 'lon', 'elevation', 'speed', 'cadence', 'temperature', 'environment', 'poi']
        
        df[cols].to_sql('track_points', self.engine, if_exists='append', index=False)
        print(f"--> Сохранено точек: {len(df)}")

    def create_map(self, df, filename):
        """Рисуем карту"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(df['lon'], df['lat'], c='red', lw=3)
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
        except: pass
        
        ax.axis('off')
        img_name = f"map_{filename}.png"
        plt.savefig(img_name, bbox_inches='tight')
        plt.close()
        return img_name

class AnalystAgent:
    def analyze(self, df):
        print("\n--- Отчет Аналитика ---")
        # Корреляция
        corr = df[['temp', 'ele']].corr().iloc[0, 1]
        print(f"Корреляция Температура/Высота: {corr:.2f}")
        # Нормальность
        stat, p = stats.shapiro(df['cadence'].dropna()[:500])
        result = "Нормальное" if p > 0.05 else "НЕ нормальное"
        print(f"Распределение шагов: {result} (p={p:.5f})")

class AugmentorAgent:
    def augment(self, img_path):
        if not img_path: return
        img = cv2.imread(img_path)
        if img is None: return

        print(f"--> Аугментация для: {img_path}")
        cv2.imwrite(f"aug_rot_{img_path}", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        cv2.imwrite(f"aug_col_{img_path}", cv2.convertScaleAbs(img, alpha=1.2, beta=30))
        M = np.float32([[1, 0, 50], [0, 1, 50]])
        cv2.imwrite(f"aug_mov_{img_path}", cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))

if __name__ == "__main__":
    col = CollectorAgent()
    ana = AnalystAgent()
    aug = AugmentorAgent()

    files = [f for f in os.listdir('.') if f.endswith('.gpx')]
    if not files:
        print("Нет файлов .gpx!")
    else:
        for f in files:
            df = col.process_gpx(f)
            col.save_to_db(df, f)
            img = col.create_map(df, f)
            ana.analyze(df)
            aug.augment(img)
            print("-" * 30)