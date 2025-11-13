from flask import Blueprint, request, jsonify
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model

resume_bp = Blueprint('resume_bp', __name__)

# Load ANN model
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(SRC_DIR, 'models', 'resume_ann_model.h5')
VECT_DIR = os.path.join(SRC_DIR, 'vectorizers')
resume_model = load_model(MODEL_PATH)

# Load TF-IDF vectorizers
with open(os.path.join(VECT_DIR, 'tfidf_resume.pkl'), 'rb') as f:
    tfidf_resume = pickle.load(f)

with open(os.path.join(VECT_DIR, 'tfidf_jd.pkl'), 'rb') as f:
    tfidf_jd = pickle.load(f)

@resume_bp.route('/predict', methods=['POST'])
def predict_resume():
    data = request.json
    resume_text = data.get("resume_text", "")
    jd_text = data.get("jd_text", "")

    X_resume = tfidf_resume.transform([resume_text]).toarray()
    X_jd = tfidf_jd.transform([jd_text]).toarray()
    X = np.hstack((X_resume, X_jd))

    prediction = resume_model.predict(X)
    return jsonify({"advance_prediction": int(prediction[0] > 0.5)})
