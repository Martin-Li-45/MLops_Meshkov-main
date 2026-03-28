import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Загружаем модель и scaler при старте
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model loaded successfully!")

@app.route('/invocations', methods=['POST'])
def predict():
    data = request.get_json()
    X = np.array(data['inputs'])
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)