import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder

def download_data():

    url = 'https://raw.githubusercontent.com/Martin-Li-45/MLops_Meshkov-main/refs/heads/main/Lab4_Jenkins/healthcare-dataset-stroke-data.csv'
    
    print(f"Скачивание данных из {url}...")
    
    try:
        df = pd.read_csv(url)
        print(f"Данные загружены успешно. Размер: {len(df)} строк")
        print(f"Колонки: {list(df.columns)}")
        
        # Сохраняем локальную копию
        df.to_csv("stroke_raw.csv", index=False)
        print("Файл сохранен как stroke_raw.csv")
        
        return df
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        print("Проверьте интернет-соединение и доступность URL")
        raise

def clear_data(path2df):
    """Очистка и предобработка данных"""
    
    print(f"Загрузка данных из {path2df}...")
    df = pd.read_csv(path2df)
    print(f"Исходный размер датасета: {len(df)}")
    
    # Правильные названия колонок
    cat_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    # 1. Удаляем ID
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
        print("Колонка 'id' удалена")
    
    # 2. Обработка пропусков в bmi
    bmi_nulls = df['bmi'].isna().sum()
    if bmi_nulls > 0:
        print(f"Пропуски в bmi: {bmi_nulls}")
        df.dropna(subset=['bmi'], inplace=True)
        print(f"Размер после удаления пропусков bmi: {len(df)}")
    
    # 3. Удаляем выбросы
    df = df[(df['age'] >= 0) & (df['age'] <= 120)]
    df = df[(df['avg_glucose_level'] > 0) & (df['bmi'] > 0)]
    print(f"Размер после удаления выбросов: {len(df)}")
    
    # 4. Кодируем категориальные признаки
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    print("Категориальные признаки закодированы")
    
    # 5. Сохраняем очищенный датасет
    df.to_csv('df_clear.csv', index=False)
    print(f"Очищенный датасет сохранен в df_clear.csv (размер: {len(df)} строк)")
    
    return True

if __name__ == "__main__":
    print("=== Загрузка и очистка данных Stroke Prediction ===")
    print(f"Текущая директория: {os.getcwd()}")
    print(f"Файлы в директории: {os.listdir('.')}")
    
    # Проверяем, есть ли уже файл stroke_raw.csv
    if os.path.exists("stroke_raw.csv"):
        print("Файл stroke_raw.csv уже существует, использую его")
    else:
        print("Файл stroke_raw.csv не найден, начинаю загрузку...")
        # Скачиваем данные
        download_data()
    
    # Проверяем, что файл теперь существует
    if not os.path.exists("stroke_raw.csv"):
        print("ОШИБКА: Не удалось загрузить или создать файл stroke_raw.csv!")
        exit(1)
    
    # Очищаем данные
    clear_data("stroke_raw.csv")
    
    print("=== Процесс завершен успешно ===")