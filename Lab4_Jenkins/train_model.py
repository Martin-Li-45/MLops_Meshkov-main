import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import os
import sys

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
    
    print(f"Loaded {len(df)} rows", file=sys.stderr)
    print("Columns:", list(df.columns), file=sys.stderr)
    
    # Целевая переменная
    target = 'stroke'
    if target not in df.columns:
        print(f"ERROR: '{target}' not found in columns!", file=sys.stderr)
        print("Available columns:", list(df.columns), file=sys.stderr)
        exit(1)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # Масштабируем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Разделяем данные
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}", file=sys.stderr)
    print(f"Positive class ratio: {y.mean():.4f}", file=sys.stderr)
    
    # Настраиваем MLflow
    mlflow.set_experiment("stroke_prediction_model")
    
    with mlflow.start_run() as run:
        # Обучаем модель
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Метрики
        accuracy, precision, recall, f1, roc_auc = eval_metrics(y_val, y_pred, y_pred_proba)
        
        # Логируем параметры
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("class_weight", "balanced")
        
        # Логируем метрики
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"Metrics - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}", file=sys.stderr)
        
        # ========== ИСПРАВЛЕННАЯ ЧАСТЬ ==========
        # Создаем сигнатуру
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Логируем модель ПРАВИЛЬНО - с artifact_path="model"
        # Это создаст структуру artifacts/model/ с необходимыми файлами
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # ВАЖНО: именно "model", а не что-то другое
            signature=signature,
            registered_model_name="stroke_predictor"
        )
        
        # Сохраняем scaler как артефакт (чтобы он был в MLflow)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path, artifact_path="model")
        
        # Сохраняем модель локально (как fallback)
        joblib.dump(model, "stroke_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        
        # Получаем правильный URI для serving
        model_uri = f"runs:/{run.info.run_id}/model"
        
        # Сохраняем путь в best_model.txt
        with open("best_model.txt", "w") as f:
            f.write(model_uri)
        
        # Выводим путь для логирования
        print(model_uri)
        
        print("✅ Model saved successfully", file=sys.stderr)
        print(f"Model URI: {model_uri}", file=sys.stderr)
    
    print("=== Training completed ===", file=sys.stderr)