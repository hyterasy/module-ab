import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Настройки
DB_CONN = 'mysql+pymysql://root:root@localhost:3306/competition'
MODEL_FILE = 'risk_model.pkl'

class ModelingAgent:
    def __init__(self):
        self.engine = create_engine(DB_CONN)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("[ModelAgent] Агент инициализирован.")

    def load_and_label_data(self):
        """3.1 Загрузка и автоматическая разметка данных для обучения"""
        print("--> Загрузка данных из БД...")
        df = pd.read_sql("SELECT * FROM track_points", self.engine)
        
        if df.empty:
            print("ОШИБКА: База пуста! Сначала запусти main.py (Модуль А).")
            return pd.DataFrame()

        # Заполняем пустоты
        df = df.fillna(0)

        # --- АВТО-РАЗМЕТКА (Создаем "Учителя" для модели) ---
        # Мы должны сказать модели, что является "Опасно", а что "Нет", 
        # чтобы она нашла закономерности.
        def get_risk_label(row):
            # Пример логики: Высоко в горах или в болоте — опасно (1)
            # В лесу при жаре — пожароопасно (2)
            if row['elevation'] > 1500: return 1 # Сложная эвакуация
            if row['elevation'] < 50 and row['environment'] == 'Forest': return 1 # Затопление
            if row['temperature'] > 28 and row['environment'] == 'Forest': return 2 # Пожар
            return 0 # Безопасно

        df['target'] = df.apply(get_risk_label, axis=1)
        print(f"--> Данные размечены. Классы: {df['target'].unique()}")
        return df

    def train(self, df):
        """3.1 Обучение модели"""
        # Выбираем признаки (на основе чего гадаем)
        features = ['lat', 'lon', 'elevation', 'speed', 'temperature']
        X = df[features]
        y = df['target']

        # Разбиваем на обучение (80%) и тест (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"--> Обучение модели (RandomForest) на {len(X_train)} точках...")
        self.model.fit(X_train, y_train)

        # Оценка качества
        predictions = self.model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"\n[METRICS] Точность модели: {acc:.2%}")
        print("Отчет по классам:\n", classification_report(y_test, predictions, zero_division=0))

        # Сохранение модели
        joblib.dump(self.model, MODEL_FILE)
        print(f"--> Модель сохранена в файл: {MODEL_FILE}")

    def continuous_learning(self):
        """3.2 Непрерывное обучение (симуляция)"""
        print("\n--- 3.2 Запуск непрерывного дообучения ---")
        # В реальности тут был бы цикл, который ждет новые данные.
        # Мы просто заново грузим базу и переобучаем модель.
        df = self.load_and_label_data()
        if not df.empty:
            self.train(df)

    def forecast_risks(self):
        """3.3 Прогнозирование динамики на 10 лет"""
        print("\n--- 3.3 Прогноз рисков на 10 лет ---")
        # Генерируем фейковые данные будущих лет
        years = np.arange(2025, 2036)
        
        # Симулируем рост температур (глобальное потепление) -> рост пожаров
        risk_trend = [10 + (i * 1.5) + np.random.normal(0, 2) for i in range(len(years))]
        
        plt.figure(figsize=(10, 6))
        plt.plot(years, risk_trend, marker='o', linestyle='-', color='red', label='Риск пожаров')
        plt.title("Прогноз динамики природных рисков (2025-2035)")
        plt.xlabel("Год")
        plt.ylabel("Индекс опасности")
        plt.grid(True)
        plt.legend()
        
        filename = "forecast_10_years.png"
        plt.savefig(filename)
        print(f"--> График прогноза сохранен: {filename}")

if __name__ == "__main__":
    agent = ModelingAgent()
    
    # Шаг 1: Загрузка и Разметка
    data = agent.load_and_label_data()
    
    # Шаг 2: Обучение (3.1)
    if not data.empty:
        agent.train(data)
        
    # Шаг 3: Переобучение (3.2 - демонстрация функции)
    # agent.continuous_learning() 
    
    # Шаг 4: Прогноз (3.3)
    agent.forecast_risks()
