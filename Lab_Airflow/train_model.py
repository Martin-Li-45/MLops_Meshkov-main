import pandas as pd
from os import name
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib


def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=['Price']), df['Price']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1)).ravel() 
    return X_scale, Y_scale, power_trans


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    df = pd.read_csv("/home/meshkov/airflow/dags/df_clear.csv")
    
    # Используем Simple логарифмирование вместо PowerTransformer
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Масштабируем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    # Логарифмируем целевую переменную (просто и стабильно)
    y_log = np.log1p(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_log, test_size=0.3, random_state=42
    )

    params = {
        'alpha': [0.0001, 0.001, 0.01],
        'l1_ratio': [0.1, 0.15, 0.2],
        "penalty": ["elasticnet"],
        "loss": ['squared_error'],
        "fit_intercept": [True],
    }

    mlflow.set_experiment("linear model phones")
    
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42, max_iter=1000)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4, scoring='r2')
        clf.fit(X_train, y_train)
        
        best = clf.best_estimator_
        
        # Предсказание в логарифмическом масштабе
        y_pred_log = best.predict(X_val)
        
        # Обратное преобразование через expm1
        y_price_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_val)
        
        # Метрики на оригинальной шкале
        rmse = np.sqrt(mean_squared_error(y_true, y_price_pred))
        mae = mean_absolute_error(y_true, y_price_pred)
        r2 = r2_score(y_true, y_price_pred)

        # Логируем параметры
        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("cv_best_score", clf.best_score_)

        # Логируем модель
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        # Сохраняем модель и scaler
        with open("lr_phones.pkl", "wb") as file:
            joblib.dump(best, file)
        with open("scaler.pkl", "wb") as f:
            joblib.dump(scaler, f)
        
        print(f"\n✅ Model trained successfully!")
        print(f"R2: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"Best params: {clf.best_params_}")

if __name__ == "__main__":
    train()