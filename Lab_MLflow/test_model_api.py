import requests
import json
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


# Загружаем тестовые данные
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"\n📊 Тестовая выборка: {len(y_test)} примеров")
print(f"   Из них с инсультом: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")

# Тестируем первые 10 примеров
print("\n🔍 Тестирование API:")
print("-" * 60)

correct = 0
detected_strokes = 0
total_strokes = sum(y_test[:10])

for i in range(10):
    sample = X_test.iloc[i:i+1].values.tolist()
    columns = X_test.columns.tolist()
    
    request_data = {
        "dataframe_split": {
            "columns": columns,
            "data": sample
        }
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:5001/invocations",
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data),
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['predictions'][0]
            true_value = y_test[i]
            
            if prediction == true_value:
                correct += 1
            
            if true_value == 1 and prediction == 1:
                detected_strokes += 1
            
            status = "✓" if prediction == true_value else "✗"
            print(f"  Пример {i+1}: Предсказание={prediction}, Истина={true_value} {status}")
        else:
            print(f"  Ошибка {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Ошибка: Сервер не запущен!")
        print("   Запустите команду в отдельном терминале:")
        print("   mlflow models serve -m runs:/<run_id>/model -p 5001 --no-conda")
        break

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
print("=" * 60)
print(f"✅ Точность на первых 10 примерах: {correct}/10 ({correct*10}%)")
print(f"🎯 Обнаружено инсультов: {detected_strokes}/{total_strokes}")