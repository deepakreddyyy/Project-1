import os
import sys
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import the preprocessing function from preprocess.py
try:
    from preprocess import preprocess_text
except ImportError:
    print("Error: preprocess.py not found in the current directory.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) so the frontend can easily make requests to it
CORS(app)

# Path to the pre-trained model
MODEL_PATH = "best_emotion_model.pkl"
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first by running 'python train.py'.")
        sys.exit(1)
    
    print(f"Loading trained emotion detection model ('{MODEL_PATH}')...")
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# Serve the static frontend files (HTML, CSS, JS)
@app.route('/')
def index():
    # Serves the index.html file from the static folder
    return send_from_directory('static', 'index.html')

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded on server"}), 500
        
    # Get JSON payload from request
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"status": "error", "message": "Missing 'text' key in request JSON"}), 400
        
    user_text = data['text'].strip()
    if not user_text:
        return jsonify({"status": "error", "message": "Text cannot be empty"}), 400

    try:
        # 1. Preprocess the input text using our stemming pipeline
        cleaned_text = preprocess_text(user_text, method='stem')
        
        # 2. Run prediction using the loaded model pipeline
        prediction = model.predict([cleaned_text])[0]
        
        # Return prediction result as JSON
        return jsonify({
            "status": "success",
            "original_text": user_text,
            "cleaned_text": cleaned_text,
            "emotion": prediction
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    # Load model once when the server starts
    load_model()
    # Start the server on port 5000 (accessible locally at http://127.0.0.1:5000)
    print("Starting Flask web server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
