import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def download_data():
    df = pd.read_csv('/home/meshkov/airflow/dags/ndtv_data_final.csv', delimiter = ',')
    df.to_csv("/home/meshkov/airflow/dags/phones.csv", index = False)
    print("df: ", df.shape)
    return df

def clear_data():
    # Загружаем данные
    df = pd.read_csv("/home/meshkov/airflow/dags/phones.csv", index_col=0)
    
    # Определяем типы колонок
    num_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 'Resolution x', 'Resolution y', 
                   'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera', 
                   'Number of SIMs', 'Price']
    binary_columns = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    
    # Колонки для удаления
    columns_to_drop = ['Name', 'Model', 'Brand', 'Processor', 'Operating system']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Преобразуем Yes/No → 1/0
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
    
    initial_shape = df.shape[0]
    
    # Базовые фильтры
    df = df[df["Battery capacity (mAh)"].between(500, 10000)]
    df = df[df["Screen size (inches)"].between(3.0, 7.5)]
    df = df[df["Resolution x"].between(240, 4000)]
    df = df[df["Resolution y"].between(320, 4000)]
    df = df[df["RAM (MB)"].between(256, 16384)]
    df = df[df["Internal storage (GB)"].between(1, 1024)]
    df = df[df["Rear camera"].between(0, 200)]
    df = df[df["Front camera"].between(0, 100)]
    df = df[df["Number of SIMs"].between(1, 3)]
    df = df[df["Price"].between(1000, 500000)]
    
    # Удаляем дубликаты
    df = df.drop_duplicates()
    df = df.dropna()
    
    # === НОВЫЕ УЛУЧШЕНИЯ ===
    
    # 1. One-Hot Encoding для Number of SIMs
    if 'Number of SIMs' in df.columns:
        sim_dummies = pd.get_dummies(df['Number of SIMs'], prefix='sim', dtype=int)
        df = pd.concat([df, sim_dummies], axis=1)
        df = df.drop(columns=['Number of SIMs'])
        # Обновляем списки колонок
        if 'Number of SIMs' in num_columns:
            num_columns.remove('Number of SIMs')
        binary_columns.extend(['sim_1', 'sim_2', 'sim_3'])
    
    # 2. Создаем признак PPI
    df['PPI'] = (np.sqrt(df['Resolution x']**2 + df['Resolution y']**2) / df['Screen size (inches)']).round(1)
    num_columns.append('PPI')
    
    # 3. Создаем признак общей памяти
    df['Total_Memory_GB'] = (df['RAM (MB)'] / 1024) + df['Internal storage (GB)']
    num_columns.append('Total_Memory_GB')
    
    # 4. Логарифмические преобразования
    df['Log_RAM'] = np.log1p(df['RAM (MB)'])
    df['Log_Storage'] = np.log1p(df['Internal storage (GB)'])
    df['Log_Battery'] = np.log1p(df['Battery capacity (mAh)'])
    
    num_columns.extend(['Log_RAM', 'Log_Storage', 'Log_Battery'])
    
    # 5. Признаки взаимодействия
    df['Camera_Power'] = df['Rear camera'] * df['Front camera']
    df['Storage_per_RAM'] = df['Internal storage (GB)'] / (df['RAM (MB)'] / 1024 + 0.1)
    df['Battery_per_Inch'] = df['Battery capacity (mAh)'] / df['Screen size (inches)']
    
    num_columns.extend(['Camera_Power', 'Storage_per_RAM', 'Battery_per_Inch'])
    
    # 6. Удаляем исходные Resolution колонки
    df = df.drop(columns=['Resolution x', 'Resolution y'])
    if 'Resolution x' in num_columns:
        num_columns.remove('Resolution x')
    if 'Resolution y' in num_columns:
        num_columns.remove('Resolution y')
    
    # 7. Проверка и удаление бесконечных значений
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 8. ИТОГОВЫЙ НАБОР КОЛОНОК - только числовые признаки!
    # Собираем все финальные колонки для модели
    feature_columns = num_columns + binary_columns
    # Убеждаемся, что Price в списке
    if 'Price' not in feature_columns:
        feature_columns.append('Price')
    
    # Оставляем только нужные колонки
    available_columns = [col for col in feature_columns if col in df.columns]
    df = df[available_columns]
    
    print(f"\n=== РЕЗУЛЬТАТЫ ОЧИСТКИ ===")
    print(f"Размер до очистки: {initial_shape}")
    print(f"Размер после очистки: {df.shape}")
    print(f"Удалено записей: {initial_shape - df.shape[0]}")
    print(f"Процент удаленных: {(initial_shape - df.shape[0]) / initial_shape * 100:.2f}%")
    
    print(f"\nЦелевая переменная (Price):")
    print(f"  Мин: {df['Price'].min():.0f}")
    print(f"  Макс: {df['Price'].max():.0f}")
    print(f"  Среднее: {df['Price'].mean():.0f}")
    print(f"  Медиана: {df['Price'].median():.0f}")
    
    print(f"\nКоличество признаков: {len(df.columns) - 1}")
    print(f"Типы данных:")
    print(df.dtypes)
    
    # Проверяем, что все колонки числовые
    non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        print(f"⚠️  ВНИМАНИЕ: Найдены нечисловые колонки: {non_numeric}")
        # Удаляем их если есть
        df = df.drop(columns=non_numeric)
        print(f"   Удалены колонки: {non_numeric}")
    
    # Сохраняем
    df = df.reset_index(drop=True)
    df.to_csv('/home/meshkov/airflow/dags/df_clear.csv', index=False)
    print(f"\n✅ Очищенный датасет сохранён в df_clear.csv")
    print(f"Финальные колонки: {df.columns.tolist()}")
    
    return True

    
dag_phones = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule=timedelta(minutes=5),
    #schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
download_task = PythonOperator(python_callable=download_data, task_id = "download_phones", dag = dag_phones)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_phones", dag = dag_phones)
train_task = PythonOperator(python_callable=train, task_id = "train_phones", dag = dag_phones)
download_task >> clear_task >> train_task
