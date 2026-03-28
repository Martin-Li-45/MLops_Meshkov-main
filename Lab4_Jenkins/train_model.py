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
from imblearn.over_sampling import SMOTE  # ДОБАВИТЬ: библиотека для балансировки
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
    
    # ========== ДОБАВЛЕНО: SMOTE для балансировки классов ==========
    print("\nApplying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced train size: {X_train_balanced.shape}")
    print(f"Balanced positive class ratio: {y_train_balanced.mean():.4f}")
    
    # Параметры для GridSearch
    params = {
        'n_estimators': [100, 150, 200],  # немного увеличил
        'max_depth': [10, 15, 20],  # убрал None для стабильности
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None]  # SMOTE уже балансирует, поэтому class_weight=None
    }
    
    # Настройка MLflow эксперимента
    mlflow.set_experiment("stroke_prediction_model")
    
    with mlflow.start_run():
        print("\nStarting GridSearchCV...")
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        clf = GridSearchCV(
            rf, 
            params, 
            cv=3, 
            n_jobs=4,
            scoring='roc_auc',
            verbose=1
        )
        
        # Используем СБАЛАНСИРОВАННЫЕ данные для обучения
        clf.fit(X_train_balanced, y_train_balanced)
        
        best_model = clf.best_estimator_
        
        # Предсказания на валидационной выборке
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        # ========== ДОБАВЛЕНО: Оптимизация порога ==========
        # Находим оптимальный порог по ROC-AUC
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
        # Оптимальный порог - максимизирующий Youden's J statistic
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Применяем оптимальный порог
        y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Расчет метрик для обычного порога (0.5)
        accuracy, precision, recall, f1, roc_auc = eval_metrics(y_val, y_pred, y_pred_proba)
        
        # Расчет метрик для оптимизированного порога
        accuracy_opt, precision_opt, recall_opt, f1_opt, _ = eval_metrics(y_val, y_pred_optimized, y_pred_proba)
        
        # Логируем параметры
        mlflow.log_param("best_params", str(clf.best_params_))
        mlflow.log_param("n_estimators", best_model.n_estimators)
        mlflow.log_param("max_depth", best_model.max_depth)
        mlflow.log_param("min_samples_split", best_model.min_samples_split)
        mlflow.log_param("min_samples_leaf", best_model.min_samples_leaf)
        mlflow.log_param("class_weight", best_model.class_weight)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("optimal_threshold", optimal_threshold)  # ДОБАВЛЕНО
        
        # Логируем метрики для обычного порога
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # ДОБАВЛЕНО: логируем метрики для оптимизированного порога
        mlflow.log_metric("accuracy_optimized", accuracy_opt)
        mlflow.log_metric("precision_optimized", precision_opt)
        mlflow.log_metric("recall_optimized", recall_opt)
        mlflow.log_metric("f1_score_optimized", f1_opt)
        
        # Матрица ошибок
        cm = confusion_matrix(y_val, y_pred_optimized)
        print(f"\nConfusion Matrix (optimized threshold={optimal_threshold:.3f}):\n{cm}")
        
        signature = infer_signature(X_train_balanced, best_model.predict(X_train_balanced))
        
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
        print(f"\nMetrics (default threshold=0.5):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"\nMetrics (optimized threshold={optimal_threshold:.3f}):")
        print(f"  Accuracy:  {accuracy_opt:.4f}")
        print(f"  Precision: {precision_opt:.4f}")
        print(f"  Recall:    {recall_opt:.4f}")
        print(f"  F1-Score:  {f1_opt:.4f}")
        print("="*50)
    
    # Сохраняем модель и порог
    print("\n" + "="*50)
    print("Saving model locally...")
    
    # Сохраняем модель, scaler и порог
    joblib.dump(best_model, "stroke_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(optimal_threshold, "threshold.pkl")  # ДОБАВЛЕНО: сохраняем порог
    
    # Записываем путь в best_model.txt
    with open("best_model.txt", "w") as f:
        f.write("stroke_model.pkl")
    
    print("✓ Model saved to: stroke_model.pkl")
    print("✓ Scaler saved to: scaler.pkl")
    print("✓ Threshold saved to: threshold.pkl")
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