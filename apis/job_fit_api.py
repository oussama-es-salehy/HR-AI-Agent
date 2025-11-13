from flask import Blueprint, request, jsonify
import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

job_fit_bp = Blueprint('job_fit_bp', __name__)

# Load the trained model
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(SRC_DIR, 'models', 'job_fit_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    job_fit_model = pickle.load(f)

@job_fit_bp.route('/predict', methods=['POST'])
def predict_fit():
    data = request.json

    # Extract input fields
    required_skills = data.get("required_skills", "")
    candidate_skills = data.get("candidate_skills", "")
    degree = data.get("degree", "")
    years_experience = data.get("years_experience", 0)

    # 1. Compute overlap
    req_set = set([s.strip() for s in required_skills.split(",") if s.strip()])
    cand_set = set([s.strip() for s in candidate_skills.split(",") if s.strip()])
    overlap = len(req_set & cand_set)

    # 2. Number of required skills
    num_required_skills = len(req_set)

    # 3. Overlap ratio (avoid division by zero)
    overlap_ratio = overlap / num_required_skills if num_required_skills > 0 else 0

    # Prepare dataframe for the model
    df_input = pd.DataFrame([{
        "degree": degree,
        "years_experience": years_experience,
        "overlap": overlap,
        "num_required_skills": num_required_skills,
        "overlap_ratio": overlap_ratio
    }])

    # Predict
    prediction = job_fit_model.predict(df_input)
    proba = job_fit_model.predict_proba(df_input)[:, 1]

    return jsonify({
        "prediction": int(prediction[0]),
        "fit_probability": float(proba[0])
    })
