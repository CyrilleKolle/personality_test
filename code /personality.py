from flask import Flask, request, jsonify
import joblib
import numpy as np  
import os
import tensorflow as tf
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

script_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(script_dir, 'scaler.pkl')
encoder_path = os.path.join(script_dir, 'encoder_model.h5')
kmeans_path = os.path.join(script_dir, 'kmeans_model.pkl')

scaler = joblib.load(scaler_path)
km = joblib.load(kmeans_path)
encoder = tf.keras.models.load_model(encoder_path)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data = np.array(data["features"]).reshape(1, -1)
    scaled_data = scaler.transform(data)
    encoded_data = encoder.predict(scaled_data)
    prediction = km.predict(encoded_data)
    
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(port=5000, debug=True)