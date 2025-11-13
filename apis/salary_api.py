from flask import Blueprint, request, jsonify
import os
import pickle
import pandas as pd

salary_bp = Blueprint('salary_bp', __name__)

# Load the trained model
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(SRC_DIR, 'models', 'salary_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    salary_model = pickle.load(f)

@salary_bp.route('/predict', methods=['POST'])
def predict_salary():
    data = request.json

    # Expected input fields
    input_df = pd.DataFrame([{
        "years_experience": data.get("years_experience", 0),
        "role": data.get("role", ""),
        "degree": data.get("degree", ""),
        "company_size": data.get("company_size", ""),
        "location": data.get("location", ""),
        "level": data.get("level", "")
    }])

    # Predict
    prediction = salary_model.predict(input_df)

    return jsonify({
        "prediction": float(prediction[0])
    })
