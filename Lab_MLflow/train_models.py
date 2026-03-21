import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import os

warnings.filterwarnings('ignore')

# Переходим в директорию скрипта
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

print(f"Текущая директория: {os.getcwd()}")

# Проверяем наличие файлов
required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
for file in required_files:
    if not os.path.exists(file):
        print(f"ОШИБКА: Файл {file} не найден!")
        print("Сначала запустите prepare_data.py")
        exit(1)

# Загрузка данных
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")
print(f"y_test distribution: {np.bincount(y_test)}")

# Настройка MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Stroke_Prediction_Experiment")

# Модели с балансировкой классов
models_config = {
    'LogisticRegression': {
        'class': LogisticRegression,
        'params_list': [
            {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced'}
        ]
    },
    'RandomForest': {
        'class': RandomForestClassifier,
        'params_list': [
            {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}
        ]
    },
    'XGBoost': {
        'class': XGBClassifier,
        'params_list': [
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 
             'eval_metric': 'logloss', 'scale_pos_weight': 19.5}
        ]
    }
}

results = []

for model_name, config in models_config.items():
    for params in config['params_list']:
        model = config['class'](**params)
        run_name = f"{model_name}"
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.sklearn.log_model(model, artifact_path="model")
            
            results.append({
                'model': model_name,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'roc_auc': roc_auc
            })
            
            print(f"✅ {model_name}: ROC_AUC={roc_auc:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

# Результаты
print("\n" + "="*50)
results_df = pd.DataFrame(results)
print(results_df.to_string())

best = results_df.loc[results_df['f1_score'].idxmax()]
print(f"\nЛучшая модель: {best['model']}")
print(f"F1-Score: {best['f1_score']:.4f}")
print(f"Recall: {best['recall']:.4f}")
print(f"ROC-AUC: {best['roc_auc']:.4f}")