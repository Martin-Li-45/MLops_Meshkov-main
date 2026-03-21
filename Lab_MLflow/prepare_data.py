import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

# Переходим в директорию скрипта
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

print(f"Текущая директория: {os.getcwd()}")

# Проверяем наличие файла датасета
if not os.path.exists('healthcare-dataset-stroke-data.csv'):
    print("ОШИБКА: Файл healthcare-dataset-stroke-data.csv не найден!")
    print("Пожалуйста, скачайте датасет с Kaggle и поместите его в папку Lab_MLflow")
    exit(1)

# Загрузка данных
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

print("Исходные данные:")
print(f"Размер датасета: {df.shape}")
print(f"Количество пропусков:\n{df.isnull().sum()}")
print(f"\nРаспределение целевой переменной:\n{df['stroke'].value_counts()}")

# Удаляем id
df.drop('id', axis=1, inplace=True)

# Кодируем категориальные признаки
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Обработка пропусков bmi
imputer = SimpleImputer(strategy='median')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# Разделение на признаки и целевую переменную
X = df.drop('stroke', axis=1)
y = df['stroke']

# Разбиение с учетом стратификации
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# Масштабирование числовых признаков
scaler = StandardScaler()
num_cols = ['age', 'avg_glucose_level', 'bmi']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Сохраняем обработанные данные
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Сохраняем скейлер и энкодеры
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("\n✅ Данные успешно подготовлены!")
print("Созданы файлы: X_train.csv, X_test.csv, y_train.csv, y_test.csv")