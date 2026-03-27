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
import os

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
                                                      stratify=y)
    
    # Настраиваем MLflow
    mlflow.set_experiment("stroke_prediction_model")
    with mlflow.start_run():
        # Обучаем модель
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
        
        # Сохраняем scaler и модель локально
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(model, "stroke_model.pkl")
        
        # Получаем URI текущего run
        current_run_id = mlflow.active_run().info.run_id
        current_experiment_id = mlflow.active_run().info.experiment_id
        
        # Формируем путь к артефактам
        mlflow_artifacts_path = f"./mlruns/{current_experiment_id}/{current_run_id}/artifacts/model"
        absolute_path = os.path.abspath(mlflow_artifacts_path)
        
        # Записываем путь в файл
        with open("best_model.txt", "w") as f:
            f.write(absolute_path)
        
        # Также создаем файл в директории download для надежности
        download_path = "/var/lib/jenkins/workspace/download/best_model.txt"
        with open(download_path, "w") as f:
            f.write(absolute_path)
        
        print(f"Model path saved to best_model.txt: {absolute_path}")
    
    print("Training completed successfully")