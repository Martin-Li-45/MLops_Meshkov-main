import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from mlflow.models import infer_signature
import mlflow
import joblib
import warnings
import os
import sys
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("✓ SMOTE imported successfully")
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠ WARNING: imbalanced-learn not installed")

def scale_frame(frame):
    df = frame.copy()
    target = 'stroke'
    X = df.drop(columns=[target])
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, scaler

def eval_metrics(actual, pred, pred_proba=None):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    roc_auc = None
    if pred_proba is not None:
        roc_auc = roc_auc_score(actual, pred_proba)
    return accuracy, precision, recall, f1, roc_auc

if __name__ == "__main__":
    print("="*50)
    print("Starting train_model.py")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    print("="*50)
    
    # Проверяем наличие файла
    if not os.path.exists("./df_clear.csv"):
        print("ERROR: df_clear.csv not found!")
        sys.exit(1)
    
    # Загружаем очищенные данные
    print("\nLoading data...")
    df = pd.read_csv("./df_clear.csv")
    print(f"Dataset shape: {df.shape}")
    
    # ========== ДОБАВЛЯЕМ: проверка и удаление пропусков ==========
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")
    
    if df.isnull().sum().sum() > 0:
        print("Removing rows with missing values...")
        df = df.dropna()
        print(f"Dataset shape after cleaning: {df.shape}")
    # ============================================================
    
    print(f"Target distribution:\n{df['stroke'].value_counts()}")
    
    # Масштабируем признаки
    X_scaled, y, scaler = scale_frame(df)
    
    # Разделяем данные
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
    print(f"Train positive class ratio: {y_train.mean():.4f}")
    print(f"Validation positive class ratio: {y_val.mean():.4f}")
    
    # ========== ДОБАВЛЯЕМ: проверка на NaN перед SMOTE ==========
    print(f"\nChecking for NaN in X_train: {np.isnan(X_train).sum()}")
    if np.isnan(X_train).sum() > 0:
        print("ERROR: X_train still contains NaN values!")
        print("Columns with NaN:")
        nan_cols = np.where(np.isnan(X_train).any(axis=0))[0]
        print(f"Indices: {nan_cols}")
        sys.exit(1)
    # ============================================================
    
    # Применяем SMOTE если доступен и нет NaN
    if SMOTE_AVAILABLE and np.isnan(X_train).sum() == 0:
        print("\nApplying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Train size: {X_train_balanced.shape}")
        print(f"Balanced positive class ratio: {y_train_balanced.mean():.4f}")
        train_data = (X_train_balanced, y_train_balanced)
    else:
        print("\n⚠ Training without SMOTE")
        train_data = (X_train, y_train)
    
    # Параметры для GridSearch
    params = {
        'n_estimators': [100, 150, 200],
        'max_depth': [8, 10, 12],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    # Настройка MLflow эксперимента
    mlflow.set_experiment("stroke_prediction_model")
    
    with mlflow.start_run():
        print("\nStarting GridSearchCV...")
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        
        clf = GridSearchCV(
            rf, 
            params, 
            cv=3,
            n_jobs=4,
            scoring='roc_auc',
            verbose=1
        )
        
        X_train_used, y_train_used = train_data
        clf.fit(X_train_used, y_train_used)
        
        best_model = clf.best_estimator_
        
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        accuracy, precision, recall, f1, roc_auc = eval_metrics(y_val, y_pred, y_pred_proba)
        
        # Логируем параметры и метрики
        mlflow.log_param("smote_applied", SMOTE_AVAILABLE and np.isnan(X_train).sum() == 0)
        mlflow.log_param("best_params", str(clf.best_params_))
        mlflow.log_param("n_estimators", best_model.n_estimators)
        mlflow.log_param("max_depth", best_model.max_depth)
        mlflow.log_param("class_weight", "balanced")
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        cm = confusion_matrix(y_val, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        mlflow.sklearn.log_model(
            best_model, 
            "model", 
            signature=signature,
            registered_model_name="stroke_predictor"
        )
        
        print("\n" + "="*50)
        print("Model Training Completed!")
        print(f"Best parameters: {clf.best_params_}")
        print(f"Validation metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print("="*50)
    
    # Сохраняем модель
    joblib.dump(best_model, "stroke_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    with open("best_model.txt", "w") as f:
        f.write("stroke_model.pkl")
    
    print("\n✓ Model saved to: stroke_model.pkl")
    print("✓ Scaler saved to: scaler.pkl")