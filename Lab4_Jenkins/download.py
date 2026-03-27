import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    url = 'https://raw.githubusercontent.com/Martin-Li-45/MLops_Meshkov-main/refs/heads/main/Lab4_Jenkins/healthcare-dataset-stroke-data.csv'

    df = pd.read_csv(url)
    df.to_csv("stroke_raw.csv", index=False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    # Определяем категориальные и числовые колонки
    cat_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    num_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    
    # 1. Удаляем ID, так как он не несет информации
    df.drop(columns=['id'], inplace=True)
    
    # 2. Очистка: удаляем строки с пропусками в bmi (или заполняем медианой)
    # Для простоты удалим строки с пропущенным bmi
    df.dropna(subset=['bmi'], inplace=True)
    
    # 3. Удаляем выбросы (здравый смысл)
    # Возраст не может быть > 120 или < 0 (в датасете все корректно, но для примера)
    df = df[(df['age'] >= 0) & (df['age'] <= 120)]
    # Уровень глюкозы и ИМТ не могут быть отрицательными
    df = df[(df['avg_glucose_level'] > 0) & (df['bmi'] > 0)]
    
    # 4. Кодируем категориальные признаки
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    # 5. Сохраняем очищенный датасет
    df.to_csv('df_clear.csv', index=False)
    print("Data cleared and saved to df_clear.csv")
    return True

if __name__ == "__main__":
    clear_data("stroke_raw.csv")