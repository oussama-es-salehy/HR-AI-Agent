from flask import Blueprint, request, jsonify
import pandas as pd
import pickle
import os

hiring_bp = Blueprint("hiring_bp", __name__)

# Load the trained model
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(SRC_DIR, 'models', 'random_forest_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    hiring_model = pickle.load(f)

# Hard-coded training columns
X_train_columns = [
    'referral_flag',
    'years_exp_band_1-3',
    'years_exp_band_3-6',
    'years_exp_band_6+',
    'skills_coverage_band_Low',
    'skills_coverage_band_Medium',
    'english_level_B1',
    'english_level_B2',
    'english_level_C1',
    'english_level_NAN',
    'location_match_Relocate',
    'location_match_Remoteok'
]

@hiring_bp.route("/predict", methods=["POST"])
def predict_hiring():
    try:
        data = request.json  # Expect a single JSON object
        print(data)
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        df = pd.DataFrame([data])  # Wrap in list to make a single-row DataFrame

        # Columns to one-hot encode
        categorical_cols = ["years_exp_band", "skills_coverage_band", "english_level", "location_match"]

        # One-hot encode
        df_encoded = pd.get_dummies(df, columns=categorical_cols)

        # Add missing columns
        for col in X_train_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Reorder columns
        df_encoded = df_encoded[X_train_columns]

        # Predict
        prediction = hiring_model.predict(df_encoded)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
