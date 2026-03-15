from datetime import datetime, timedelta
import json
import logging
import os
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from hooks import CarsHook

# Настройка логгера
logger = logging.getLogger(__name__)

# Аргументы по умолчанию для DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2026, 3, 15),
}

def fetch_cars_data(**context):
    """
    Task 1: Получение данных из cars-api и сохранение в сыром виде
    """
    logger.info("=" * 60)
    logger.info("TASK 1: FETCHING CARS DATA FROM API")
    logger.info("=" * 60)
    
    # Используем существующий хук
    hook = CarsHook(conn_id='carsapi')
    
    # Получаем все автомобили (метод уже реализован в hooks.py)
    cars = list(hook.get_cars(batch_size=100))
    logger.info(f"Fetched {len(cars)} car records from API")
    
    # Создаем директорию для сырых данных, если её нет
    raw_dir = '/data/cars/raw'
    os.makedirs(raw_dir, exist_ok=True)
    
    # Генерируем имя файла с таймстемпом
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_file = f'{raw_dir}/cars_raw_{timestamp}.json'
    
    # Сохраняем данные через хук
    hook.save_raw_data(cars, raw_file)
    logger.info(f"Raw data saved to: {raw_file}")
    
    # Передаем имя файла в следующий таск через XCom
    context['task_instance'].xcom_push(key='raw_file', value=raw_file)
    
    # Логируем статистику
    logger.info(f"Records count: {len(cars)}")
    
    return f"Successfully fetched {len(cars)} records"

def clean_cars_data(**context):
    """Task 2: Предобработка данных"""
    logger.info("=" * 60)
    logger.info("TASK 2: CLEANING AND PROCESSING CARS DATA")
    logger.info("=" * 60)
    
    try:
        # Получаем имя файла из предыдущего таска
        raw_file = context['task_instance'].xcom_pull(key='raw_file', task_ids='fetch_cars_data')
        
        if not raw_file or not os.path.exists(raw_file):
            raise Exception(f"❌ Raw file not found: {raw_file}")
        
        logger.info(f"📂 Loading raw data from: {raw_file}")
        
        # Загружаем данные
        with open(raw_file, 'r') as f:
            cars_data = json.load(f)
        
        # Преобразуем в DataFrame
        df = pd.DataFrame(cars_data)
        
        # ДИАГНОСТИКА: выводим информацию о колонках
        logger.info("=" * 60)
        logger.info("🔍 ДОСТУПНЫЕ КОЛОНКИ:")
        logger.info(f"{df.columns.tolist()}")
        logger.info("=" * 60)
        
        # Статистика до очистки
        logger.info(f"📊 Initial data shape: {df.shape}")
        logger.info(f"📊 Initial duplicates: {df.duplicated().sum()}")
        
        # 1. Удаление дубликатов
        df = df.drop_duplicates()
        
        # 2. Удаление пропусков
        df = df.dropna()
        
        logger.info(f"✅ After removing duplicates and NA: {df.shape}")
        
        # 3. Кодирование категориальных признаков - ИСПРАВЛЕНО
        # Fuel_type encoding
        fuel_mapping = {
            'Petrol': 1, 
            'Diesel': 2, 
            'Electric': 3, 
            'Hybrid': 4,
            'Plug-in Hybrid': 5,
            'Metan/Propan': 6
        }
        df['fuel_encoded'] = df['Fuel_type'].map(fuel_mapping)
        
        # Transmission encoding
        transmission_mapping = {
            'Manual': 1, 
            'Automatic': 2, 
            'CVT': 3
        }
        df['transmission_encoded'] = df['Transmission'].map(transmission_mapping)
        
        # Brand encoding (Label encoding)
        if 'Make' in df.columns:
            brand_mapping = {brand: idx for idx, brand in enumerate(df['Make'].unique())}
            df['brand_encoded'] = df['Make'].map(brand_mapping)
        
        # Style encoding
        if 'Style' in df.columns:
            style_mapping = {style: idx for idx, style in enumerate(df['Style'].unique())}
            df['style_encoded'] = df['Style'].map(style_mapping)
        
        # Логируем результаты кодирования
        logger.info("🔢 Результаты кодирования:")
        logger.info(f"   Fuel types distribution:\n{df['fuel_encoded'].value_counts().sort_index()}")
        logger.info(f"   Transmission distribution:\n{df['transmission_encoded'].value_counts().sort_index()}")
        
        if 'brand_encoded' in df.columns:
            logger.info(f"   Unique brands: {df['Make'].nunique()}")
        
        # Создаем директорию для очищенных данных
        cleaned_dir = '/opt/airflow/data/cleaned'
        os.makedirs(cleaned_dir, exist_ok=True)
        
        # Сохраняем результаты
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cleaned_json = f'{cleaned_dir}/cars_cleaned_{timestamp}.json'
        cleaned_csv = f'{cleaned_dir}/cars_cleaned_{timestamp}.csv'
        
        df.to_json(cleaned_json, orient='records', indent=2)
        df.to_csv(cleaned_csv, index=False)
        
        logger.info(f"💾 Cleaned JSON saved to: {cleaned_json}")
        logger.info(f"💾 Cleaned CSV saved to: {cleaned_csv}")
        
        # Логируем финальную статистику
        logger.info(f"📊 Final data shape: {df.shape}")
        logger.info(f"📊 Final columns: {df.columns.tolist()}")
        logger.info(f"📊 Records processed: {len(df)}")
        
        return f"Successfully processed {len(df)} records"
    except Exception as e:
        logger.error(f"❌ Error cleaning data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# Создаем DAG
with DAG(
    dag_id='cars_data_pipeline',
    default_args=default_args,
    description='Pipeline for fetching and cleaning car data',
    schedule='*/10 * * * *',  # <--- ЗАМЕНИТЕ НА schedule
    catchup=False,
    tags=['cars', 'etl', 'cleaning'],
    max_active_runs=1,
) as dag:
    
    # Task 1: Fetch data from API
    fetch_task = PythonOperator(
        task_id='fetch_cars_data',
        python_callable=fetch_cars_data
    )
    
    # Task 2: Clean and process data
    clean_task = PythonOperator(
        task_id='clean_cars_data',
        python_callable=clean_cars_data
    )
    
    # Set dependencies
    fetch_task >> clean_task