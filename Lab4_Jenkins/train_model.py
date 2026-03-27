import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import numpy as np

def eval_metrics(y_true, y_pred, y_pred_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
    return accuracy, precision, recall, f1, roc_auc

if __name__ == "__main__":
    # Загружаем очищенные данные
    df = pd.read_csv("./df_clear.csv")
    
    # Целевая переменная
    target = 'stroke'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Масштабируем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Разделяем данные
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=y)  # стратификация для несбалансированных классов
    
    # Настраиваем MLflow
    mlflow.set_experiment("stroke_prediction_model")
    with mlflow.start_run():
        # Обучаем модель RandomForest (можно и другие)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Метрики
        accuracy, precision, recall, f1, roc_auc = eval_metrics(y_val, y_pred, y_pred_proba)
        
        # Логируем параметры и метрики
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("class_weight", "balanced")
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Логируем модель
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        # Сохраняем scaler для последующего использования в сервисе
        joblib.dump(scaler, "scaler.pkl")
        
        # Сохраняем модель локально
        joblib.dump(model, "stroke_model.pkl")
        
        print("Model trained and logged.")
    
    # Находим лучшую модель по roc_auc (или другой метрике)
    df_runs = mlflow.search_runs()
    # Если экспериментов нет, обрабатываем ошибку
    if len(df_runs) > 0:
        # Сортируем по roc_auc
        best_run = df_runs.sort_values("metrics.roc_auc", ascending=False).iloc[0]
        path2model = best_run['artifact_uri'].replace("file://", "") + '/model'
        print(path2model)
        with open("best_model.txt", "w") as f:
            f.write(path2model)
    else:
        # Если логов нет, используем локальную модель
        with open("best_model.txt", "w") as f:
            f.write("./stroke_model.pkl")
        print("No MLflow runs found, using local model path.")