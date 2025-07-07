from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Load models and scaler
try:
    autoencoder = load_model('autoencoder_minmax_final_20250705_final.keras')
    clf = joblib.load('xgboost_model_minmax_final_20250705_final.pkl')
    scaler = joblib.load('scaler_minmax_final_20250705_final.pkl')
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Create encoder model from autoencoder
encoder = autoencoder.get_layer('encoder_bottleneck').output
encoder_model = load_model('autoencoder_minmax_final_20250705_final.keras', compile=False)
encoder_model = Model(inputs=autoencoder.input, outputs=encoder)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from JSON request
        data = request.json.get('data')
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to numpy array and validate shape
        data = np.array(data).reshape(1, -1)
        if data.shape[1] != scaler.n_features_in_:
            return jsonify({'error': f'Expected {scaler.n_features_in_} features, got {data.shape[1]}'}), 400
        
        # Preprocess: scale and encode
        data_scaled = scaler.transform(data)
        encoded_data = encoder_model.predict(data_scaled, verbose=0)
        
        # Predict fraud probability with XGBoost
        fraud_probability = clf.predict_proba(encoded_data)[:, 1][0]
        
        return jsonify({'fraud_probability': float(fraud_probability)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)