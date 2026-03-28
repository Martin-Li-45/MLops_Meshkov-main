import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from mlflow.models import infer_signature
import mlflow
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

def scale_frame(frame):
    """
    Масштабирование признаков для задачи классификации
    """
    df = frame.copy()
    target = 'stroke'
    X = df.drop(columns=[target])
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values, scaler

def eval_metrics(actual, pred, pred_proba=None):
    """
    Расчет метрик для задачи классификации
    """
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    
    roc_auc = None
    if pred_proba is not None:
        roc_auc = roc_auc_score(actual, pred_proba)
    
    return accuracy, precision, recall, f1, roc_auc

if __name__ == "__main__":
    # Загружаем очищенные данные
    print("Loading data...")
    df = pd.read_csv("./df_clear.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['stroke'].value_counts()}")
    
    # Масштабируем признаки
    X_scaled, y, scaler = scale_frame(df)
    
    # Разделяем данные на train и validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
    print(f"Train positive class ratio: {y_train.mean():.4f}")
    print(f"Validation positive class ratio: {y_val.mean():.4f}")
    
    # ========== ИЗМЕНЕНИЕ 1: Добавляем SMOTE для балансировки классов ==========
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Train size: {X_train_balanced.shape}")
    print(f"Balanced positive class ratio: {y_train_balanced.mean():.4f}")
    
    # ========== ИЗМЕНЕНИЕ 2: Улучшенные параметры для GridSearch ==========
    params = {
        'n_estimators': [150, 200, 300],
        'max_depth': [8, 10, 12],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None],  # Оставляем None, так как SMOTE уже сбалансировал данные
        'max_features': ['sqrt', 'log2']
    }
    
    # Настройка MLflow эксперимента
    mlflow.set_experiment("stroke_prediction_model")
    
    with mlflow.start_run():
        print("\nStarting GridSearchCV...")
        
        # Создаем модель RandomForestClassifier
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # GridSearch с кросс-валидацией на сбалансированных данных
        clf = GridSearchCV(
            rf, 
            params, 
            cv=5,
            n_jobs=4,
            scoring='roc_auc',
            verbose=1
        )
        
        # ========== ИЗМЕНЕНИЕ 3: Обучаем на сбалансированных данных ==========
        clf.fit(X_train_balanced, y_train_balanced)
        
        # Получаем лучшую модель
        best_model = clf.best_estimator_
        
        # ========== ИЗМЕНЕНИЕ 4: Предсказания с обычным порогом 0.5 ==========
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Расчет метрик
        accuracy, precision, recall, f1, roc_auc = eval_metrics(y_val, y_pred, y_pred_proba)
        
        # Логируем параметры
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("best_params", str(clf.best_params_))
        mlflow.log_param("n_estimators", best_model.n_estimators)
        mlflow.log_param("max_depth", best_model.max_depth)
        mlflow.log_param("min_samples_split", best_model.min_samples_split)
        mlflow.log_param("min_samples_leaf", best_model.min_samples_leaf)
        mlflow.log_param("class_weight", best_model.class_weight)
        mlflow.log_param("max_features", best_model.max_features)
        mlflow.log_param("random_state", 42)
        
        # Логируем метрики
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Логируем матрицу ошибок
        cm = confusion_matrix(y_val, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Создаем сигнатуру для модели
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        # Логируем модель в MLflow
        mlflow.sklearn.log_model(
            best_model, 
            "model", 
            signature=signature,
            registered_model_name="stroke_predictor"
        )
        
        print("\n" + "="*50)
        print("Model Training Completed!")
        print(f"Best parameters: {clf.best_params_}")
        print(f"Best CV score: {clf.best_score_:.4f}")
        print(f"Validation metrics (threshold=0.5):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print("="*50)
    
    # Сохраняем модель и scaler локально
    print("\n" + "="*50)
    print("Saving model locally...")
    
    joblib.dump(best_model, "stroke_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    # Записываем путь в best_model.txt
    with open("best_model.txt", "w") as f:
        f.write("stroke_model.pkl")
    
    print("✓ Model saved to: stroke_model.pkl")
    print("✓ Scaler saved to: scaler.pkl")
    print("✓ Path saved to: best_model.txt")
    print("="*50)

# Сохраняем информацию о признаках
feature_info = pd.DataFrame({
    'feature': ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'],
    'description': [
        'Пол пациента', 'Возраст', 'Гипертония (0/1)', 'Болезни сердца (0/1)', 'Был в браке',
        'Тип работы', 'Тип проживания', 'Средний уровень глюкозы', 'Индекс массы тела', 'Статус курения'
    ]
})
feature_info.to_csv('feature_info.csv', index=False)
print("\n✓ Feature info saved to feature_info.csv")